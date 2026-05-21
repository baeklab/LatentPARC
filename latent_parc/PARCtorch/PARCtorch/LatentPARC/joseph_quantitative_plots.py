"""
plot_results.py

Usage:
    python plot_results.py \
        --results results/proposed.npz "Proposed" \
                  results/parcv1.npz   "PARCv1" \
                  results/parcv2.npz   "PARCv2" \
        --save_path figures/hotspot_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

MODEL_COLORS = [
    "mediumpurple",   # Proposed
    "forestgreen",    # PARCv1
    "darkorange",     # PARCv2
    "steelblue",      # any additional models
    "crimson",
    "pink",
    "green"
]

def load_results(path):
    return np.load(path)

def plot_all_models(model_results, save_path=None, dt=0.17, t_start=0.68):
    """
    model_results: list of (label, npz_data) tuples, e.g.
        [("Proposed", data1), ("PARCv1", data2), ...]
    Ground truth is taken from the first file (they share the same GT).
    """
    gt = model_results[0][1]  # GT arrays are identical across files
    
    n_ts      = len(gt["mean_T_hs"])
    n_ts_rate = len(gt["mean_dotT_hs"])
    t_main = np.array([t_start + i * dt for i in range(n_ts)])
    t_rate = np.array([t_start + i * dt for i in range(n_ts_rate)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    panels = [
        # (ax,          t,      gt_mean,           gt_p5,              gt_p95,             pred_mean_key,       pred_p5_key,          pred_p95_key,          title,                       ylabel)
        (axes[0, 0], t_main, "mean_A_hs",    "perc5_A_hs",    "perc95_A_hs",    "pred_mean_A_hs",    "pred_perc5_A_hs",    "pred_perc95_A_hs",    r"(a)  $A_{hs}$",            r"$\mu m^2$"),
        (axes[0, 1], t_rate, "mean_dotA_hs", "perc5_dotA_hs", "perc95_dotA_hs", "pred_mean_dotA_hs", "pred_perc5_dotA_hs", "pred_perc95_dotA_hs", r"(b)  $\dot{A}_{hs}$",      r"$\mu m^2 / ns$"),
        (axes[1, 0], t_main, "mean_T_hs",    "perc5_T_hs",    "perc95_T_hs",    "pred_mean_T_hs",    "pred_perc5_T_hs",    "pred_perc95_T_hs",    r"(c)  $T_{hs}$",            r"$K$"),
        (axes[1, 1], t_rate, "mean_dotT_hs", "perc5_dotT_hs", "perc95_dotT_hs", "pred_mean_dotT_hs", "pred_perc5_dotT_hs", "pred_perc95_dotT_hs", r"(d)  $\dot{T}_{hs}$",      r"$K / ns$"),
    ]

    for (ax, t, gt_mean_k, gt_p5_k, gt_p95_k,
         pred_mean_k, pred_p5_k, pred_p95_k, title, ylabel) in panels:

        # Ground truth (same for all models)
        ax.plot(t, gt[gt_mean_k], color="black", linewidth=2, label="Ground Truth")
        ax.fill_between(t, gt[gt_p5_k], gt[gt_p95_k], color="black", alpha=0.15)

        # Each model
        for i, (label, data) in enumerate(model_results):
            color = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax.plot(t, data[pred_mean_k], color=color, linewidth=2, label=label)
            ax.fill_between(t, data[pred_p5_k], data[pred_p95_k], color=color, alpha=0.15)

        # ax.set_title(title, fontsize=14, loc="left", fontweight="bold", pad=8)
        # ax.set_ylabel(ylabel, fontsize=11)
        # ax.set_xlabel("ns", fontsize=11)
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)

        ax.set_title(title, fontsize=18, loc="left", fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
        ax.set_xlabel("$ns$", fontsize=16, fontweight="bold")
        ax.tick_params(axis='both', labelsize=11)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color="black", label="Ground Truth")]
    handles += [
        mpatches.Patch(color=MODEL_COLORS[i % len(MODEL_COLORS)], label=label)
        for i, (label, _) in enumerate(model_results)
    ]
    # fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               # fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
    
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02),
               prop={"size": 18, "weight": "normal"})

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="Alternating path/label pairs: path1.npz Label1 path2.npz Label2 ...")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dt", type=float, default=0.17)
    parser.add_argument("--t_start", type=float, default=0.68)
    args = parser.parse_args()

    # Parse alternating path/label pairs
    assert len(args.results) % 2 == 0, "Provide path/label pairs: path1.npz Label1 path2.npz Label2"
    pairs = list(zip(args.results[0::2], args.results[1::2]))
    model_results = [(label, load_results(path)) for path, label in pairs]

    plot_all_models(model_results, save_path=args.save_path, dt=args.dt, t_start=args.t_start)