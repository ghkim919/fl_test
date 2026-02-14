from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server.strategy import FedAdam, FedAvg, FedAvgM, FedProx

from src.kfl_strategy import KFL
from src.model import CNN, get_parameters


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def create_strategy(config, evaluate_fn=None, cluster_evaluate_fn=None):
    model = CNN()
    initial_parameters = ndarrays_to_parameters(get_parameters(model))

    num_clients = config["num_clients"]
    fraction_fit = config["fraction_fit"]
    min_fit_clients = max(2, int(num_clients * fraction_fit))

    def on_fit_config_fn(server_round):
        return {
            "local_epochs": config["local_epochs"],
            "learning_rate": config["learning_rate"],
        }

    common_kwargs = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_fit_clients=min_fit_clients,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    strategy_name = config["strategy"].lower()

    if strategy_name == "fedavg":
        return FedAvg(**common_kwargs)

    elif strategy_name == "fedprox":
        proximal_mu = config["fedprox"]["proximal_mu"]
        return FedProx(proximal_mu=proximal_mu, **common_kwargs)

    elif strategy_name == "fedadam":
        fc = config["fedadam"]
        return FedAdam(
            eta=fc["eta"],
            tau=fc["tau"],
            beta_1=fc["beta_1"],
            beta_2=fc["beta_2"],
            **common_kwargs,
        )

    elif strategy_name == "fedavgm":
        fc = config["fedavgm"]
        return FedAvgM(
            server_momentum=fc["server_momentum"],
            **common_kwargs,
        )

    elif strategy_name == "kfl":
        fc = config.get("kfl", {})
        return KFL(
            requesting_device_id=fc.get("requesting_device_id", 0),
            clustering_threshold=fc.get("clustering_threshold", 0.01),
            min_clustering_rounds=fc.get("min_clustering_rounds", 3),
            max_clustering_rounds=fc.get("max_clustering_rounds", 15),
            kalman_Q=fc.get("kalman_Q", 0.0179),
            kalman_R=fc.get("kalman_R", 0.0003),
            cluster_evaluate_fn=cluster_evaluate_fn,
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
