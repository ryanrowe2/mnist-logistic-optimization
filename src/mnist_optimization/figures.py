"""Plots derived exclusively from executed experiment arrays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .optimizers import OptimizationResult

plt.switch_backend("Agg")


def save_figures(
    result: OptimizationResult,
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    destination: str | Path,
    *,
    pair: tuple[int, int],
    provenance: str,
) -> list[str]:
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    title_prefix = f"{pair[0]} vs {pair[1]} | {provenance}"

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(result.iterations, result.losses, lw=1.8)
    ax.set(
        xlabel="Completed parameter updates",
        ylabel="Mean logistic objective",
        title=f"Convergence | {title_prefix}",
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = folder / "convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    from .evaluation import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=range(cm.shape[1]),
        yticks=(0, 1),
        xticklabels=([str(pair[0]), str(pair[1]), "tie"] if cm.shape[1] == 3 else [str(pair[0]), str(pair[1])]),
        yticklabels=(str(pair[0]), str(pair[1])),
        xlabel="Predicted digit",
        ylabel="True digit",
        title=f"Confusion matrix | {title_prefix}",
    )
    for (i, j), count in np.ndenumerate(cm):
        ax.text(j, i, str(count), ha="center", va="center", color="white" if count > cm.max() / 2 else "black")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = folder / "confusion.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    if X_test.shape[1] in (784, 785):
        pixel_max = 255 if X_test.shape[1] == 784 else 1
        # Genuine rows from this experiment; never substitute stock/synthetic digit art.
        selected = np.arange(min(8, len(X_test)))
        fig, axes = plt.subplots(2, 4, figsize=(9, 5))
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, selected, strict=False):
            ax.imshow(X_test[index, :784].reshape(28, 28), cmap="gray", vmin=0, vmax=pixel_max)
            true_name = pair[0] if y_true[index] == -1 else pair[1]
            predicted_name = "tie" if y_pred[index] == 0 else (pair[0] if y_pred[index] == -1 else pair[1])
            ax.set_title(f"true {true_name} / pred {predicted_name}", fontsize=9)
        fig.suptitle(f"Actual evaluation samples | {title_prefix}", fontsize=11)
        fig.tight_layout()
        path = folder / "examples.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
        wrong = np.flatnonzero(y_pred != y_true)[:8]
        if len(wrong):
            fig, axes = plt.subplots(2, 4, figsize=(9, 5))
            for ax in axes.flat:
                ax.axis("off")
            for ax, index in zip(axes.flat, wrong, strict=False):
                ax.imshow(X_test[index, :784].reshape(28, 28), cmap="gray", vmin=0, vmax=pixel_max)
                true_name = pair[0] if y_true[index] == -1 else pair[1]
                predicted_name = "tie" if y_pred[index] == 0 else (pair[0] if y_pred[index] == -1 else pair[1])
                ax.set_title(f"true {true_name} / pred {predicted_name}", fontsize=9)
            fig.suptitle(f"Misclassified evaluation examples | {title_prefix}", fontsize=11)
            fig.tight_layout()
            path = folder / "mistakes.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
        weights = result.weights[:784].reshape(28, 28)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        coefficient_limit = float(np.max(np.abs(weights)))
        if coefficient_limit == 0.0:
            coefficient_limit = 1.0  # Symmetric non-degenerate scale for an all-zero model.
        im = ax.imshow(weights, cmap="coolwarm", vmin=-coefficient_limit, vmax=coefficient_limit)
        ax.set(title=f"Learned pixel coefficients | {title_prefix}", xticks=[], yticks=[])
        fig.colorbar(im, ax=ax, label="Signed coefficient value (zero-centered)")
        fig.tight_layout()
        path = folder / "weights.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
        if result.method == "coordinate":
            fig, ax = plt.subplots(figsize=(5, 4.5))
            im = ax.imshow(result.update_counts[:784].reshape(28, 28), cmap="inferno")
            ax.set(title=f"Effective coordinate updates | {title_prefix}", xticks=[], yticks=[])
            fig.colorbar(im, ax=ax, label="Number of effective weight changes")
            fig.tight_layout()
            path = folder / "coordinate_updates.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
    return paths
