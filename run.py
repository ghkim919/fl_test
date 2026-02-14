import argparse

from flwr.simulation import run_simulation

from src.client import create_client_app
from src.dataset import init_fds
from src.server import create_server_app
from src.utils import load_config, save_results, set_seed


def main():
    parser = argparse.ArgumentParser(description="Run a single FL experiment")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument(
        "--partition", type=str, default=None, choices=["iid", "dirichlet"]
    )
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.strategy:
        config["strategy"] = args.strategy
    if args.num_rounds:
        config["num_rounds"] = args.num_rounds
    if args.num_clients:
        config["num_clients"] = args.num_clients
    if args.partition:
        config["partition"]["type"] = args.partition
    if args.alpha:
        config["partition"]["alpha"] = args.alpha
    if args.seed is not None:
        config["seed"] = args.seed

    set_seed(config["seed"])
    init_fds(config["num_clients"], config["partition"], config["seed"])

    results_collector = []
    cluster_results_collector = []

    client_app = create_client_app(config)
    server_app = create_server_app(
        config, results_collector, cluster_results_collector
    )

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=config["num_clients"],
    )

    results = {
        "strategy": config["strategy"],
        "config": config,
        "rounds": results_collector,
    }

    if cluster_results_collector:
        results["cluster_rounds"] = cluster_results_collector

    partition_type = config["partition"]["type"]
    output_path = f"results/{config['strategy']}_{partition_type}.json"
    save_results(results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
