import argparse
import copy

from flwr.simulation import run_simulation

from src.client import create_client_app
from src.dataset import init_fds
from src.server import create_server_app
from src.utils import load_config, save_results, set_seed

STRATEGIES = ["fedavg", "fedprox", "fedadam", "fedavgm"]


def run_single(config):
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

    result = {
        "strategy": config["strategy"],
        "rounds": results_collector,
    }

    if cluster_results_collector:
        result["cluster_rounds"] = cluster_results_collector

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare FL strategies")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument("--partitions", nargs="+", default=["iid"])
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    base_config = load_config(args.config)

    if args.alpha:
        base_config["partition"]["alpha"] = args.alpha

    all_results = {}

    for partition_type in args.partitions:
        partition_results = []

        for strategy in args.strategies:
            config = copy.deepcopy(base_config)
            config["strategy"] = strategy
            config["partition"]["type"] = partition_type

            print(f"\n{'=' * 60}")
            print(f"Running {strategy} with {partition_type} partition")
            print(f"{'=' * 60}")

            result = run_single(config)
            partition_results.append(result)

            save_data = {
                "strategy": strategy,
                "config": config,
                "rounds": result["rounds"],
            }
            if "cluster_rounds" in result:
                save_data["cluster_rounds"] = result["cluster_rounds"]

            save_results(save_data, f"results/{strategy}_{partition_type}.json")

        all_results[partition_type] = partition_results

    comparison = {
        "config": {k: v for k, v in base_config.items() if k != "strategy"},
        "results": all_results,
    }

    partitions_str = "_".join(args.partitions)
    output_path = f"results/comparison_{partitions_str}.json"
    save_results(comparison, output_path)
    print(f"\nComparison results saved to {output_path}")


if __name__ == "__main__":
    main()
