"""
Plot results from compare_methods.py

Reads results.csv and produces a publication-quality comparison chart.

Usage:
    python plot_results.py
"""

import csv
import matplotlib.pyplot as plt

# Config
INPUT_CSV = "results.csv"
OUTPUT_PNG = "method_comparison.png"

# Normalise the rewards (0-100) to make graph look nicer
SHIFT = 201
WORST_SHIFTED = 1
OPTIMAL_SHIFTED = 142

# plot appearance
FIGURE_SIZE = (9.5, 5.8)
TITLE_SIZE = 16
AXIS_LABEL_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 12

# one colour per method
STYLE_MAP = {
    "Tabular Q-Learning":    {"color": "#FF0000"},   # red
    "Vanilla DQN":           {"color": "#9932CC"},   # purple
    "DQN + Replay":          {"color": "#0080FF"},   # blue
    "DQN + Replay + Target": {"color": "#2ecc71"},   # green
}

PLOT_ORDER = [
    "Tabular Q-Learning",
    "Vanilla DQN",
    "DQN + Replay",
    "DQN + Replay + Target",
]


def load_results(path):
    """Read the CSV produced by compare_methods.py."""
    data = {}

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key == "episode":
                    continue

                data.setdefault(key, []).append(float(value))

    episodes = list(range(1, len(next(iter(data.values()))) + 1))
    return episodes, data


def normalise(values):
    """Map raw average rewards to a 0-100 scale."""

    return [(v + SHIFT - WORST_SHIFTED) / (OPTIMAL_SHIFTED - WORST_SHIFTED) * 100
            for v in values]



def main():
    episodes, data = load_results(INPUT_CSV)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # plot each method's learning curve
    for method in PLOT_ORDER:
        if method not in data:
            continue

        style = STYLE_MAP.get(method, {"color": "gray"})

        ax.plot(episodes, normalise(data[method]),
                color=style["color"], label=method,
                linewidth=2.5, alpha=0.9)

    # dashed line showing the theoretical best reward
    ax.axhline(y=100, color="black", linestyle="--",
               linewidth=2.0, label="Optimal (100)")

    # axis limits and labels
    ax.set_ylim(-0.5, 110)
    ax.set_xlim(0, max(episodes))
    ax.set_xlabel("Episode", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Total Reward", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("Method Comparison on 50 x 50 GridWorld", fontsize=TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    # make the legend lines match the plot lines in thickness
    legend = ax.legend(fontsize=LEGEND_SIZE, loc="lower right",
                       facecolor="white", edgecolor="gray",
                       handlelength=2.5, markerfirst=False)
    
    for line in legend.get_lines():
        lw = 2.0 if line.get_linestyle() == "--" else 2.5
        line.set_linewidth(lw)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # save copy for the poster and slides
    plt.savefig(OUTPUT_PNG, dpi=300, facecolor="white",
                edgecolor="none", bbox_inches="tight")
    print(f"Saved plot to {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
