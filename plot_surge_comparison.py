import argparse
import csv
from itertools import accumulate
from pathlib import Path

import matplotlib.pyplot as plt


def _read_series(path):
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            orr = float(row["orr"])
            gmv = float(row["gmv"])
            gap = float(row["gap"])
            data[step] = {"orr": orr, "gmv": gmv, "gap": gap}
    return data


def _align_series(control, experiment, zeroshot):
    common_steps = sorted(set(control.keys()) & set(experiment.keys()) & set(zeroshot.keys()))
    steps = []
    ctrl_orr, exp_orr, zero_orr = [], [], []
    ctrl_gmv, exp_gmv, zero_gmv = [], [], []
    for step in common_steps:
        steps.append(step)
        ctrl_orr.append(control[step]["orr"])
        exp_orr.append(experiment[step]["orr"])
        zero_orr.append(zeroshot[step]["orr"])
        ctrl_gmv.append(control[step]["gmv"])
        exp_gmv.append(experiment[step]["gmv"])
        zero_gmv.append(zeroshot[step]["gmv"])
    return steps, ctrl_orr, exp_orr, zero_orr, ctrl_gmv, exp_gmv, zero_gmv


def _align_two(a, b):
    common_steps = sorted(set(a.keys()) & set(b.keys()))
    steps = []
    a_orr, b_orr = [], []
    a_gmv, b_gmv = [], []
    for step in common_steps:
        steps.append(step)
        a_orr.append(a[step]["orr"])
        b_orr.append(b[step]["orr"])
        a_gmv.append(a[step]["gmv"])
        b_gmv.append(b[step]["gmv"])
    return steps, a_orr, b_orr, a_gmv, b_gmv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control",
        default="surge_v4_control.csv",
        help="Path to control CSV.",
    )
    parser.add_argument(
        "--experiment",
        default="surge_v4_experiment.csv",
        help="Path to experiment CSV.",
    )
    parser.add_argument(
        "--zeroshot",
        default="surge_v4_zeroshot.csv",
        help="Path to zero-shot CSV.",
    )
    parser.add_argument(
        "--coopride",
        default=None,
        help="Path to CoopRide CSV (LLM-A-HDRL vs CoopRide mode).",
    )
    parser.add_argument(
        "--llm",
        default=None,
        help="Path to LLM-A-HDRL CSV (LLM-A-HDRL vs CoopRide mode).",
    )
    parser.add_argument(
        "--coopride-label",
        default="CoopRide",
        help="Label for CoopRide curve.",
    )
    parser.add_argument(
        "--llm-label",
        default="LLM-A-HDRL",
        help="Label for LLM-A-HDRL curve.",
    )
    parser.add_argument(
        "--output",
        default="surge_v4_comparison.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--peak-start",
        type=int,
        default=90,
        help="Blizzard peak start step.",
    )
    parser.add_argument(
        "--peak-end",
        type=int,
        default=120,
        help="Blizzard peak end step.",
    )
    args = parser.parse_args()

    if args.coopride or args.llm:
        if not args.coopride or not args.llm:
            raise SystemExit("Both --coopride and --llm are required for LLM-A-HDRL vs CoopRide plots.")
        coopride = _read_series(args.coopride)
        llm = _read_series(args.llm)
        steps, coop_orr, llm_orr, coop_gmv, llm_gmv = _align_two(coopride, llm)
        mode = "llm_vs_coopride"
    else:
        control = _read_series(args.control)
        experiment = _read_series(args.experiment)
        zeroshot = _read_series(args.zeroshot)
        steps, ctrl_orr, exp_orr, zero_orr, ctrl_gmv, exp_gmv, zero_gmv = _align_series(
            control, experiment, zeroshot
        )
        mode = "baseline_triple"

    if not steps:
        raise SystemExit("No overlapping steps found between the two CSV files.")

    try:
        import seaborn as sns  # type: ignore

        sns.set_style("whitegrid")
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: ORR
    if mode == "llm_vs_coopride":
        ax1.plot(steps, coop_orr, label=args.coopride_label, color="#1f77b4")
        ax1.plot(steps, llm_orr, label=args.llm_label, color="#d62728")
    else:
        ax1.plot(steps, ctrl_orr, label="Control (Normal)", color="#1f77b4")
        ax1.plot(steps, exp_orr, label="Experiment (Urgent)", color="#d62728")
        ax1.plot(steps, zero_orr, label="Zero-shot (Blizzard)", color="#2ca02c")
    ax1.axvspan(args.peak_start, args.peak_end, color="gray", alpha=0.15)
    ax1.set_title("Scenario B ORR Over Time")
    ax1.set_ylabel("ORR")
    ax1.legend(loc="upper right")

    # Plot 2: Cumulative GMV
    if mode == "llm_vs_coopride":
        coop_cum = list(accumulate(coop_gmv))
        llm_cum = list(accumulate(llm_gmv))
        ax2.plot(steps, coop_cum, label=args.coopride_label, color="#1f77b4")
        ax2.plot(steps, llm_cum, label=args.llm_label, color="#d62728")
    else:
        ctrl_cum = list(accumulate(ctrl_gmv))
        exp_cum = list(accumulate(exp_gmv))
        zero_cum = list(accumulate(zero_gmv))
        ax2.plot(steps, ctrl_cum, label="Control (Normal)", color="#1f77b4")
        ax2.plot(steps, exp_cum, label="Experiment (Urgent)", color="#d62728")
        ax2.plot(steps, zero_cum, label="Zero-shot (Blizzard)", color="#2ca02c")
    ax2.axvspan(args.peak_start, args.peak_end, color="gray", alpha=0.15, label="Blizzard peak")
    ax2.set_title("Scenario B Cumulative GMV")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative GMV")
    ax2.legend(loc="upper left")

    ax2.set_xlim(min(steps), max(steps))

    output_path = Path(args.output)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
