import torch
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from src.dataset import load_test_data
from src.model import CNN, set_parameters, test
from src.strategy import create_strategy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_evaluate_fn(config, results_collector):
    test_loader = load_test_data(batch_size=config["batch_size"])

    def evaluate(server_round, parameters_ndarrays, config_dict):
        model = CNN()
        set_parameters(model, parameters_ndarrays)
        loss, accuracy = test(model, test_loader, DEVICE)

        results_collector.append(
            {
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
            }
        )

        print(f"  Round {server_round}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        return loss, {"accuracy": accuracy}

    return evaluate


def get_cluster_evaluate_fn(config, cluster_results_collector):
    test_loader = load_test_data(batch_size=config["batch_size"])

    def evaluate(server_round, parameters_ndarrays):
        model = CNN()
        set_parameters(model, parameters_ndarrays)
        loss, accuracy = test(model, test_loader, DEVICE)

        cluster_results_collector.append(
            {
                "round": server_round,
                "loss": loss,
                "accuracy": accuracy,
            }
        )

        print(
            f"  [Cluster] Round {server_round}: "
            f"loss={loss:.4f}, accuracy={accuracy:.4f}"
        )
        return loss, accuracy

    return evaluate


def create_server_app(config, results_collector, cluster_results_collector=None):
    def server_fn(context: Context):
        evaluate_fn = get_evaluate_fn(config, results_collector)

        cluster_evaluate_fn = None
        if (
            config["strategy"].lower() == "kfl"
            and cluster_results_collector is not None
        ):
            cluster_evaluate_fn = get_cluster_evaluate_fn(
                config, cluster_results_collector
            )

        strategy = create_strategy(
            config,
            evaluate_fn=evaluate_fn,
            cluster_evaluate_fn=cluster_evaluate_fn,
        )
        server_config = ServerConfig(num_rounds=config["num_rounds"])
        return ServerAppComponents(strategy=strategy, config=server_config)

    return ServerApp(server_fn=server_fn)
