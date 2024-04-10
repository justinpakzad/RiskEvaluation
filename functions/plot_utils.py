from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def hide_spines(ax: plt.Axes) -> None:
    """
    Hides the lines around the graph.
    """
    for spine in ax.spines.values():
        spine.set_visible(False)


def calculate_total(ax: plt.Axes, orientation: str) -> float:
    """Calculate the total value of bars based on orientation."""
    if orientation == "vertical":
        return sum([p.get_height() for p in ax.patches])
    else:
        return sum([p.get_width() for p in ax.patches])


def plot_learning_curves(results):
    sns.lineplot(results["validation_0"]["auc"], label="Train")
    sns.lineplot(results["validation_1"]["auc"], label="Validation")

    plt.ylabel("AUC")
    plt.title("Training vs Validation AUC")
    plt.show()


def annotate_bar(
    ax: plt.Axes,
    bar: plt.Axes,
    total: float,
    offset_ratio: float,
    lim: float,
    percentage: bool,
    fontsize: int,
    orientation: str,
) -> None:
    """Annotate a single bar with text."""
    if orientation == "vertical":
        value = bar.get_height()
        offset = value * offset_ratio
        position = (
            bar.get_x() + bar.get_width() / 2,
            value - offset if (value / total) * 100 > lim else value + offset,
        )
        text = f"{(value / total) * 100:.2f}%" if percentage else f"{value:.3f}"
    else:
        value = bar.get_width()
        offset = value * offset_ratio
        position = (
            value - offset if (value / total) * 100 > lim else value + offset,
            bar.get_y() + bar.get_height() / 2,
        )
        text = f"{(value / total) * 100:.2f}%" if percentage else f"{value:.3f}"

    ax.text(
        *position,
        text,
        ha="center" if orientation == "vertical" else "left",
        va="center" if orientation != "vertical" else "bottom",
        color="w" if (value / total) * 100 > lim else "black",
        fontsize=fontsize,
    )


def annotate_bars(
    ax: plt.Axes,
    lim: float = 10,
    percentage: bool = False,
    offset_ratio: float = 0.05,
    total: Union[None, float] = None,
    orientation: str = "vertical",
    fontsize: int = 8,
) -> None:
    """Annotates Bar Chart"""
    if total is None:
        total = calculate_total(ax, orientation)

    for bar in ax.patches:
        annotate_bar(
            ax, bar, total, offset_ratio, lim, percentage, fontsize, orientation
        )


def plot_model_evaluation(
    fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, conf_matrix: np.ndarray
) -> None:
    """Plots ROC AUC & Confusion Matrix"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    sns.heatmap(conf_matrix, annot=True, fmt="g", cbar=False)
    plt.title("Confusion Matrix")
    plt.show()


def plot_double_conf_matrix(
    matrix_1: np.ndarray, matrix_2: np.ndarray, title_1: str, title_2: str
) -> None:
    """Plots 2 confusion matrices side by side"""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(title_1)
    sns.heatmap(matrix_1, annot=True, fmt="g", cbar=False)
    plt.subplot(1, 2, 2)
    plt.title(title_2)
    sns.heatmap(matrix_2, annot=True, fmt="g", cbar=False)
    plt.show()
