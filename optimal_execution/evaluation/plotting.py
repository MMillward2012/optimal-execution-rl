from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def plot_inventory_trajectories(groups, output_path, title, show_examples=False, show_std=True, max_examples=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(groups))))

    for color, (label, episodes) in zip(colors, groups.items()):
        paths = [episode["inventory"] for episode in episodes]
        data = _matrix(paths)
        steps = np.arange(data.shape[1])
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        if show_examples:
            examples = data if max_examples is None else data[:max_examples]
            for path in examples:
                ax.plot(steps, path, color=color, alpha=0.18, linewidth=1.0)
        if show_std:
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2.4, label=label)

    ax.set_title(title)
    ax.set_xlabel("Decision step")
    ax.set_ylabel("Fraction of initial inventory")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path

def _matrix(paths):
    width = max(len(path) for path in paths)
    data = np.full((len(paths), width), np.nan, dtype=float)
    for row, path in enumerate(paths):
        data[row, :len(path)] = path
    return data