"""K-FL: Kalman Filter-Based Clustering Federated Learning.

Implementation based on:
H. Kim et al., "K-FL: Kalman Filter-Based Clustering Federated Learning
Method", IEEE Access, vol. 11, pp. 36097-36105, 2023.

Key algorithm:
1. A device requests clustering → server broadcasts its local model to all devices.
2. Each device evaluates the requesting device's model on its local data
   → builds an accuracy matrix over rounds.
3. A Kalman filter tracks the requesting device's accuracy to determine
   when enough rounds have passed for reliable clustering.
4. Clustering decision (Eq.7): devices whose accuracy difference with the
   requesting device is continuously increasing are excluded.
5. After clustering, a separate cluster model is maintained alongside the
   global model. Cluster members train on the cluster model; others train
   on the global model. Both models are aggregated independently.
"""

from typing import Callable, Optional, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate

from src.kalman_filter import KalmanFilter


class KFL(FedAvg):

    def __init__(
        self,
        *,
        requesting_device_id: int = 0,
        clustering_threshold: float = 0.01,
        min_clustering_rounds: int = 3,
        max_clustering_rounds: int = 15,
        kalman_Q: float = 0.0179,
        kalman_R: float = 0.0003,
        cluster_evaluate_fn: Optional[
            Callable[[int, NDArrays], tuple[float, float]]
        ] = None,
        **kwargs,
    ):
        kwargs["fraction_fit"] = 1.0
        kwargs["fraction_evaluate"] = 1.0
        super().__init__(**kwargs)

        self.requesting_device_id = str(requesting_device_id)
        self.clustering_threshold = clustering_threshold
        self.min_clustering_rounds = min_clustering_rounds
        self.max_clustering_rounds = max_clustering_rounds
        self.kalman_Q = kalman_Q
        self.kalman_R = kalman_R
        self.cluster_evaluate_fn = cluster_evaluate_fn

        self.kf: Optional[KalmanFilter] = None
        self.innovations: list[float] = []

        self.accuracy_matrix: dict[int, dict[str, float]] = {}
        self.requesting_device_accuracy: dict[int, float] = {}

        self.requesting_device_params: Optional[Parameters] = None

        self.is_clustered = False
        self.cluster_members: set[str] = set()
        self.cluster_round: Optional[int] = None
        self.cluster_parameters: Optional[Parameters] = None
        self._cluster_needs_init = False

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        if self._cluster_needs_init:
            self.cluster_parameters = parameters
            self._cluster_needs_init = False

        config: dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        fit_configurations = []
        for client in clients:
            if (
                self.is_clustered
                and client.cid in self.cluster_members
                and self.cluster_parameters is not None
            ):
                fit_ins = FitIns(self.cluster_parameters, config)
            else:
                fit_ins = FitIns(parameters, config)
            fit_configurations.append((client, fit_ins))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        for client_proxy, fit_res in results:
            if client_proxy.cid == self.requesting_device_id:
                self.requesting_device_params = fit_res.parameters
                break

        if not self.is_clustered:
            return super().aggregate_fit(server_round, results, failures)

        # --- Post-clustering: dual aggregation ---

        # 1. Global model ← ALL devices
        weights_global = [
            (parameters_to_ndarrays(fr.parameters), fr.num_examples)
            for _, fr in results
        ]
        global_params = ndarrays_to_parameters(aggregate(weights_global))

        # 2. Cluster model ← cluster members only
        weights_cluster = [
            (parameters_to_ndarrays(fr.parameters), fr.num_examples)
            for cp, fr in results
            if cp.cid in self.cluster_members
        ]
        if weights_cluster:
            self.cluster_parameters = ndarrays_to_parameters(
                aggregate(weights_cluster)
            )
            if self.cluster_evaluate_fn is not None:
                cluster_ndarrays = parameters_to_ndarrays(self.cluster_parameters)
                self.cluster_evaluate_fn(server_round, cluster_ndarrays)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fr.num_examples, fr.metrics) for _, fr in results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return global_params, metrics_aggregated

    # ------------------------------------------------------------------
    # EVALUATE  (federated — used to build the accuracy matrix)
    # ------------------------------------------------------------------

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        if self.is_clustered or self.requesting_device_params is None:
            return []

        config: dict[str, Scalar] = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        evaluate_ins = EvaluateIns(self.requesting_device_params, config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        if not results or self.is_clustered:
            return None, {}

        round_accuracies: dict[str, float] = {}
        for client_proxy, eval_res in results:
            accuracy = eval_res.metrics.get("accuracy", 0.0)
            round_accuracies[client_proxy.cid] = accuracy

        self.accuracy_matrix[server_round] = round_accuracies

        req_acc = round_accuracies.get(self.requesting_device_id, 0.0)
        self.requesting_device_accuracy[server_round] = req_acc

        if len(self.requesting_device_accuracy) >= 2:
            self._apply_kalman_filter(server_round, req_acc)

        metrics_aggregated: dict[str, Scalar] = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [
                (eval_res.num_examples, eval_res.metrics)
                for _, eval_res in results
            ]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                eval_metrics
            )

        return None, metrics_aggregated

    # ------------------------------------------------------------------
    # Kalman filter
    # ------------------------------------------------------------------

    def _apply_kalman_filter(self, server_round: int, measurement: float):
        rounds = sorted(self.requesting_device_accuracy.keys())

        if self.kf is None and len(rounds) >= 2:
            x_hat_init = self.requesting_device_accuracy[rounds[1]]
            p_init = max(
                (
                    self.requesting_device_accuracy[rounds[1]]
                    - self.requesting_device_accuracy[rounds[0]]
                )
                ** 2,
                0.001,
            )
            self.kf = KalmanFilter(
                A=1.0,
                H=1.0,
                Q=self.kalman_Q,
                R=self.kalman_R,
                x_hat=x_hat_init,
                P=p_init,
            )
            return

        if self.kf is None:
            return

        predicted = self.kf.predict()
        innovation = abs(measurement - predicted)
        self.innovations.append(innovation)
        self.kf.update(measurement)

        num_eval_rounds = len(self.accuracy_matrix)

        if num_eval_rounds >= self.min_clustering_rounds:
            recent = (
                self.innovations[-2:]
                if len(self.innovations) >= 2
                else self.innovations
            )
            if all(inn < self.clustering_threshold for inn in recent):
                self._perform_clustering(server_round)
                return

        if num_eval_rounds >= self.max_clustering_rounds:
            self._perform_clustering(server_round)

    # ------------------------------------------------------------------
    # Clustering  (Equation 7)
    # ------------------------------------------------------------------

    def _perform_clustering(self, server_round: int):
        self.is_clustered = True
        self.cluster_round = server_round
        self._cluster_needs_init = True

        rounds = sorted(self.accuracy_matrix.keys())

        all_cids: set[str] = set()
        for r in rounds:
            all_cids.update(self.accuracy_matrix[r].keys())

        self.cluster_members = {self.requesting_device_id}

        for cid in all_cids:
            if cid == self.requesting_device_id:
                continue

            # Eq.7: exclude if accuracy difference continuously increases
            continuously_increasing = True

            for idx in range(1, len(rounds)):
                prev_r, curr_r = rounds[idx - 1], rounds[idx]

                req_prev = self.requesting_device_accuracy.get(prev_r, 0.0)
                req_curr = self.requesting_device_accuracy.get(curr_r, 0.0)

                dev_prev = self.accuracy_matrix[prev_r].get(cid, 0.0)
                dev_curr = self.accuracy_matrix[curr_r].get(cid, 0.0)

                theta_prev = abs(dev_prev - req_prev)
                theta_curr = abs(dev_curr - req_curr)

                if theta_curr <= theta_prev:
                    continuously_increasing = False
                    break

            if not continuously_increasing:
                self.cluster_members.add(cid)

        print(f"\n[K-FL] Clustering at round {server_round}")
        print(f"[K-FL] Cluster members: {sorted(self.cluster_members)}")
        print(
            f"[K-FL] Total: {len(all_cids)}, "
            f"Cluster size: {len(self.cluster_members)}"
        )
