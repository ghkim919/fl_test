import argparse
import os

import matplotlib.pyplot as plt

from src.utils import load_results


def plot_comparison(comparison_data, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    for partition_type, strategies in comparison_data["results"].items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy_data in strategies:
            rounds = [r["round"] for r in strategy_data["rounds"]]
            accuracies = [r["accuracy"] for r in strategy_data["rounds"]]
            ax.plot(rounds, accuracies, marker="o", markersize=4, label=strategy_data["strategy"])
        ax.set_xlabel("Round")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Test Accuracy Comparison ({partition_type.upper()})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, f"accuracy_{partition_type}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy_data in strategies:
            rounds = [r["round"] for r in strategy_data["rounds"]]
            losses = [r["loss"] for r in strategy_data["rounds"]]
            ax.plot(rounds, losses, marker="o", markersize=4, label=strategy_data["strategy"])
        ax.set_xlabel("Round")
        ax.set_ylabel("Test Loss")
        ax.set_title(f"Test Loss Comparison ({partition_type.upper()})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(output_dir, f"loss_{partition_type}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_iid_vs_noniid(output_dir="results"):
    strategies = ["fedavg", "fedprox", "fedadam", "fedavgm"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]

        for partition_type, style in [("iid", "-"), ("dirichlet", "--")]:
            path = os.path.join(output_dir, f"{strategy}_{partition_type}.json")
            if not os.path.exists(path):
                continue

            data = load_results(path)
            rounds = [r["round"] for r in data["rounds"]]
            accuracies = [r["accuracy"] for r in data["rounds"]]
            ax.plot(
                rounds,
                accuracies,
                style,
                marker="o",
                markersize=3,
                label=partition_type.upper(),
            )

        ax.set_xlabel("Round")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(strategy.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("IID vs Non-IID (Dirichlet) Comparison", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "iid_vs_noniid.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize FL experiment results")
    parser.add_argument("--input", type=str, default=None, help="Path to comparison JSON")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument(
        "--iid-vs-noniid",
        action="store_true",
        help="Generate IID vs Non-IID comparison",
    )
    args = parser.parse_args()

    if args.input:
        data = load_results(args.input)
        plot_comparison(data, args.output_dir)

    if args.iid_vs_noniid:
        plot_iid_vs_noniid(args.output_dir)

    if not args.input and not args.iid_vs_noniid:
        for f in sorted(os.listdir(args.output_dir)):
            if f.startswith("comparison_") and f.endswith(".json"):
                print(f"Processing {f}")
                data = load_results(os.path.join(args.output_dir, f))
                plot_comparison(data, args.output_dir)
        plot_iid_vs_noniid(args.output_dir)


if __name__ == "__main__":
    main()
