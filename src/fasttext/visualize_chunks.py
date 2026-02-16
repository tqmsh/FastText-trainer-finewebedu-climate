#!/usr/bin/env python3
"""Visualize FastText chunk probability distributions and threshold analysis."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATA = Path("Experiment_results/fasttext/data/fasttext_probability_all_chunks.json")
THRESHOLD_DATA = Path("Experiment_results/fasttext/data/fasttext_threshold_comparison.json")
OUT_DIR = Path("Experiment_results/fasttext/images")

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_probs(data):
    """Extract prob arrays for historical and modern (use first threshold - probs are identical across thresholds)."""
    hist_probs = []
    mod_probs = []
    hist_keywords = []
    mod_keywords = []

    for run in data:
        if run["threshold"] != 0.5:
            continue
        for article in run["articles"]:
            for chunk in article["chunks"]:
                p = chunk["prob"]
                has_kw = len(chunk["keywords"]) > 0
                if "historical" in article["file"]:
                    hist_probs.append(p)
                    hist_keywords.append(has_kw)
                else:
                    mod_probs.append(p)
                    mod_keywords.append(has_kw)

    return (np.array(hist_probs), np.array(hist_keywords),
            np.array(mod_probs), np.array(mod_keywords))


def plot_prob_distribution(hist_probs, mod_probs):
    """1. Probability distribution histogram - historical vs modern."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, probs, label, color in [
        (axes[0], hist_probs, "Historical", "#2196F3"),
        (axes[1], mod_probs, "Modern", "#FF9800"),
    ]:
        ax.hist(probs, bins=50, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_title(f"{label} ({len(probs)} chunks)", fontsize=13, fontweight="bold")
        ax.set_xlabel("FastText Climate Probability")
        ax.set_ylabel("Chunk Count")
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="0.5 threshold")
        ax.legend()

        # Stats box
        high = np.sum(probs >= 0.5)
        low = np.sum(probs < 0.5)
        stats = f"mean={np.mean(probs):.3f}\nmedian={np.median(probs):.3f}\n>=0.5: {high} ({high/len(probs)*100:.1f}%)\n<0.5: {low} ({low/len(probs)*100:.1f}%)"
        ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    fig.suptitle("Chunk Probability Distribution", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prob_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [1/6] prob_distribution.png")


def plot_pass_rate_vs_threshold(hist_probs, mod_probs):
    """2. Pass rate (% and count) at each threshold."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    hist_pcts = [np.sum(hist_probs >= t) / len(hist_probs) * 100 for t in THRESHOLDS]
    mod_pcts = [np.sum(mod_probs >= t) / len(mod_probs) * 100 for t in THRESHOLDS]
    hist_counts = [int(np.sum(hist_probs >= t)) for t in THRESHOLDS]
    mod_counts = [int(np.sum(mod_probs >= t)) for t in THRESHOLDS]

    # Percentage
    ax1.plot(THRESHOLDS, hist_pcts, "o-", color="#2196F3", linewidth=2, markersize=8, label=f"Historical ({len(hist_probs)} chunks)")
    ax1.plot(THRESHOLDS, mod_pcts, "s-", color="#FF9800", linewidth=2, markersize=8, label=f"Modern ({len(mod_probs)} chunks)")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Pass Rate (%)")
    ax1.set_title("Chunk Pass Rate by Threshold", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(THRESHOLDS)
    for i, t in enumerate(THRESHOLDS):
        ax1.annotate(f"{hist_pcts[i]:.1f}%", (t, hist_pcts[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=7, color="#2196F3")
        ax1.annotate(f"{mod_pcts[i]:.1f}%", (t, mod_pcts[i]), textcoords="offset points",
                     xytext=(0, -15), ha="center", fontsize=7, color="#FF9800")

    # Counts
    x = np.arange(len(THRESHOLDS))
    w = 0.35
    bars1 = ax2.bar(x - w/2, hist_counts, w, color="#2196F3", label="Historical", alpha=0.85)
    bars2 = ax2.bar(x + w/2, mod_counts, w, color="#FF9800", label="Modern", alpha=0.85)
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Passing Chunks")
    ax2.set_title("Passing Chunk Count by Threshold", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(THRESHOLDS)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(bar.get_height())), ha="center", fontsize=7, color="#2196F3")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(int(bar.get_height())), ha="center", fontsize=7, color="#FF9800")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pass_rate_vs_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [2/6] pass_rate_vs_threshold.png")


def plot_cdf(hist_probs, mod_probs):
    """3. Cumulative distribution function."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for probs, label, color in [
        (hist_probs, "Historical", "#2196F3"),
        (mod_probs, "Modern", "#FF9800"),
    ]:
        sorted_p = np.sort(probs)
        cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        ax.plot(sorted_p, cdf, color=color, linewidth=2, label=f"{label} ({len(probs)} chunks)")

    for t in [0.3, 0.5, 0.7, 0.9]:
        ax.axvline(t, color="gray", linestyle=":", alpha=0.4)
        ax.text(t, 0.02, f"{t}", ha="center", fontsize=8, color="gray")

    ax.set_xlabel("FastText Climate Probability")
    ax.set_ylabel("Cumulative Fraction of Chunks")
    ax.set_title("CDF of Chunk Probabilities", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "prob_cdf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [3/6] prob_cdf.png")


def plot_keyword_vs_prob(hist_probs, hist_kw, mod_probs, mod_kw):
    """4. Probability distribution: chunks with keywords vs without."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, probs, kw, label in [
        (axes[0], hist_probs, hist_kw, "Historical"),
        (axes[1], mod_probs, mod_kw, "Modern"),
    ]:
        with_kw = probs[kw]
        no_kw = probs[~kw]
        bins = np.linspace(0, 1, 40)
        ax.hist(no_kw, bins=bins, alpha=0.7, color="#9E9E9E", label=f"No Keywords ({len(no_kw)})", edgecolor="white")
        ax.hist(with_kw, bins=bins, alpha=0.8, color="#4CAF50", label=f"Has Keywords ({len(with_kw)})", edgecolor="white")
        ax.set_title(f"{label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("FastText Climate Probability")
        ax.set_ylabel("Chunk Count")
        ax.legend()

        if len(with_kw) > 0:
            stats = f"With KW: mean={np.mean(with_kw):.3f}, median={np.median(with_kw):.3f}\nNo KW: mean={np.mean(no_kw):.3f}, median={np.median(no_kw):.3f}"
            ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=8,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    fig.suptitle("Keyword Presence vs Climate Probability", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "keyword_vs_prob.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [4/6] keyword_vs_prob.png")


def plot_prob_heatmap(hist_probs, mod_probs):
    """5. Probability buckets heatmap - shows where chunks cluster."""
    fig, ax = plt.subplots(figsize=(12, 4))

    bucket_edges = np.arange(0, 1.05, 0.05)
    bucket_labels = [f"{e:.2f}" for e in bucket_edges[:-1]]

    hist_counts, _ = np.histogram(hist_probs, bins=bucket_edges)
    mod_counts, _ = np.histogram(mod_probs, bins=bucket_edges)

    # Normalize to percentages
    hist_pct = hist_counts / len(hist_probs) * 100
    mod_pct = mod_counts / len(mod_probs) * 100

    data = np.array([hist_pct, mod_pct])
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Historical", "Modern"])
    ax.set_xticks(range(len(bucket_labels)))
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Probability Bucket")
    ax.set_title("Chunk Distribution Across Probability Buckets (%)", fontsize=13, fontweight="bold")

    # Annotate cells
    for i in range(2):
        for j in range(len(bucket_labels)):
            val = data[i, j]
            if val >= 1:
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if val > 20 else "black")

    fig.colorbar(im, ax=ax, label="% of chunks", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "prob_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [5/6] prob_heatmap.png")


def plot_combined_dashboard(hist_probs, mod_probs, hist_kw, mod_kw):
    """6. Combined dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # --- Top left: overlaid histogram ---
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 40)
    ax1.hist(hist_probs, bins=bins, alpha=0.6, color="#2196F3", label="Historical", edgecolor="white")
    ax1.hist(mod_probs, bins=bins, alpha=0.6, color="#FF9800", label="Modern", edgecolor="white")
    ax1.set_xlabel("Probability")
    ax1.set_ylabel("Count")
    ax1.set_title("Probability Distribution (Overlaid)")
    ax1.legend(fontsize=8)

    # --- Top middle: pass rate line chart ---
    ax2 = fig.add_subplot(gs[0, 1])
    hist_pcts = [np.sum(hist_probs >= t) / len(hist_probs) * 100 for t in THRESHOLDS]
    mod_pcts = [np.sum(mod_probs >= t) / len(mod_probs) * 100 for t in THRESHOLDS]
    ax2.plot(THRESHOLDS, hist_pcts, "o-", color="#2196F3", linewidth=2, label="Historical")
    ax2.plot(THRESHOLDS, mod_pcts, "s-", color="#FF9800", linewidth=2, label="Modern")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Pass Rate (%)")
    ax2.set_title("Chunk Pass Rate vs Threshold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(THRESHOLDS)

    # --- Top right: box plot ---
    ax3 = fig.add_subplot(gs[0, 2])
    bp = ax3.boxplot(
        [hist_probs, mod_probs,
         hist_probs[hist_kw], mod_probs[mod_kw],
         hist_probs[~hist_kw], mod_probs[~mod_kw]],
        labels=["Hist\nAll", "Mod\nAll", "Hist\nw/KW", "Mod\nw/KW", "Hist\nno KW", "Mod\nno KW"],
        patch_artist=True
    )
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#8BC34A", "#9E9E9E", "#BDBDBD"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel("Probability")
    ax3.set_title("Probability Box Plots")

    # --- Bottom left: low prob zoom ---
    ax4 = fig.add_subplot(gs[1, 0])
    low_bins = np.linspace(0, 0.1, 30)
    ax4.hist(hist_probs[hist_probs < 0.1], bins=low_bins, alpha=0.7, color="#2196F3", label="Historical", edgecolor="white")
    ax4.hist(mod_probs[mod_probs < 0.1], bins=low_bins, alpha=0.7, color="#FF9800", label="Modern", edgecolor="white")
    ax4.set_xlabel("Probability")
    ax4.set_ylabel("Count")
    ax4.set_title("Zoom: prob < 0.1")
    ax4.legend(fontsize=8)

    # --- Bottom middle: high prob zoom ---
    ax5 = fig.add_subplot(gs[1, 1])
    high_bins = np.linspace(0.5, 1.0, 30)
    ax5.hist(hist_probs[hist_probs >= 0.5], bins=high_bins, alpha=0.7, color="#2196F3", label="Historical", edgecolor="white")
    ax5.hist(mod_probs[mod_probs >= 0.5], bins=high_bins, alpha=0.7, color="#FF9800", label="Modern", edgecolor="white")
    ax5.set_xlabel("Probability")
    ax5.set_ylabel("Count")
    ax5.set_title("Zoom: prob >= 0.5")
    ax5.legend(fontsize=8)

    # --- Bottom right: summary stats table ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = []
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        h_pass = int(np.sum(hist_probs >= t))
        m_pass = int(np.sum(mod_probs >= t))
        h_pct = h_pass / len(hist_probs) * 100
        m_pct = m_pass / len(mod_probs) * 100
        table_data.append([f"{t}", f"{h_pass}/{len(hist_probs)}", f"{h_pct:.1f}%",
                           f"{m_pass}/{len(mod_probs)}", f"{m_pct:.1f}%"])

    table = ax6.table(
        cellText=table_data,
        colLabels=["Thresh", "Hist Pass", "Hist %", "Mod Pass", "Mod %"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax6.set_title("Pass Rates Summary", fontweight="bold", pad=20)

    fig.suptitle("FastText Climate Classifier - Chunk Analysis Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(OUT_DIR / "dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [6/6] dashboard.png")


def load_validation_metrics():
    """Load validation metrics from threshold_comparison.json."""
    data = json.load(open(THRESHOLD_DATA))
    return data["results"]


def plot_validation_curves(metrics):
    """7. Precision / Recall / F1 / Accuracy curves across thresholds."""
    thresholds = [m["threshold"] for m in metrics]
    precision = [m["precision"] for m in metrics]
    recall = [m["recall"] for m in metrics]
    f1 = [m["f1"] for m in metrics]
    accuracy = [m["accuracy"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: P/R/F1
    ax1.plot(thresholds, precision, "o-", color="#E91E63", linewidth=2, markersize=8, label="Precision")
    ax1.plot(thresholds, recall, "s-", color="#2196F3", linewidth=2, markersize=8, label="Recall")
    ax1.plot(thresholds, f1, "D-", color="#4CAF50", linewidth=2, markersize=8, label="F1")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("Precision / Recall / F1 vs Threshold", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(thresholds)
    ax1.set_ylim(0.7, 0.95)

    # Annotate the crossover / best F1
    best_f1_idx = np.argmax(f1)
    ax1.annotate(f"Best F1={f1[best_f1_idx]:.3f}\n@ {thresholds[best_f1_idx]}",
                 xy=(thresholds[best_f1_idx], f1[best_f1_idx]),
                 xytext=(thresholds[best_f1_idx] + 0.1, f1[best_f1_idx] - 0.05),
                 arrowprops=dict(arrowstyle="->", color="#4CAF50"),
                 fontsize=9, color="#4CAF50", fontweight="bold")

    # Right: Accuracy
    ax2.plot(thresholds, accuracy, "o-", color="#9C27B0", linewidth=2, markersize=8)
    ax2.fill_between(thresholds, accuracy, alpha=0.15, color="#9C27B0")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy vs Threshold", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(thresholds)
    ax2.set_ylim(0.91, 0.95)
    for i, t in enumerate(thresholds):
        ax2.annotate(f"{accuracy[i]:.3f}", (t, accuracy[i]),
                     textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [7/9] validation_curves.png")


def plot_confusion_matrix_grid(metrics):
    """8. Confusion matrix at each threshold."""
    fig, axes = plt.subplots(1, 9, figsize=(22, 3))

    for ax, m in zip(axes, metrics):
        t = m["threshold"]
        cm = np.array([[m["tp"], m["fn"]], [m["fp"], m["tn"]]])
        total = cm.sum()

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"t={t}", fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Clim", "Other"], fontsize=7)
        ax.set_yticklabels(["Clim", "Other"], fontsize=7)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / total * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=7, color="white" if val > total * 0.3 else "black")

    fig.suptitle("Confusion Matrices Across Thresholds (rows=actual, cols=predicted)",
                 fontsize=14, fontweight="bold", y=1.08)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [8/9] confusion_matrices.png")


def plot_validation_dashboard(metrics):
    """9. Combined validation dashboard with tradeoff analysis."""
    thresholds = [m["threshold"] for m in metrics]
    precision = [m["precision"] for m in metrics]
    recall = [m["recall"] for m in metrics]
    f1 = [m["f1"] for m in metrics]
    tp = [m["tp"] for m in metrics]
    fn = [m["fn"] for m in metrics]
    fp = [m["fp"] for m in metrics]
    tn = [m["tn"] for m in metrics]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # --- Top left: Precision-Recall tradeoff ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(recall, precision, "o-", color="#E91E63", linewidth=2, markersize=8)
    for i, t in enumerate(thresholds):
        ax1.annotate(f"{t}", (recall[i], precision[i]), textcoords="offset points",
                     xytext=(8, 0), fontsize=8, color="gray")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Tradeoff")
    ax1.grid(True, alpha=0.3)

    # --- Top middle: TP/FP/FN stacked ---
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(thresholds))
    ax2.bar(x, tp, color="#4CAF50", label="TP (correct climate)")
    ax2.bar(x, fp, bottom=tp, color="#FF5722", label="FP (false alarm)")
    ax2.bar(x, fn, bottom=[t + f for t, f in zip(tp, fp)], color="#FFC107", label="FN (missed climate)")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Count")
    ax2.set_title("TP / FP / FN Breakdown")
    ax2.set_xticks(x)
    ax2.set_xticklabels(thresholds)
    ax2.legend(fontsize=8)

    # --- Top right: FP and FN rates ---
    ax3 = fig.add_subplot(gs[0, 2])
    total_pos = [t + f for t, f in zip(tp, fn)]  # actual positives
    total_neg = [f + t for f, t in zip(fp, tn)]  # actual negatives
    fpr = [f / n * 100 for f, n in zip(fp, total_neg)]
    fnr = [f / p * 100 for f, p in zip(fn, total_pos)]
    ax3.plot(thresholds, fpr, "o-", color="#FF5722", linewidth=2, label="False Positive Rate")
    ax3.plot(thresholds, fnr, "s-", color="#FFC107", linewidth=2, label="False Negative Rate")
    ax3.set_xlabel("Threshold")
    ax3.set_ylabel("Rate (%)")
    ax3.set_title("Error Rates vs Threshold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(thresholds)

    # --- Bottom left: F1 with best highlighted ---
    ax4 = fig.add_subplot(gs[1, 0])
    colors = ["#4CAF50" if f == max(f1) else "#90CAF9" for f in f1]
    bars = ax4.bar(x, f1, color=colors, edgecolor="white")
    ax4.set_xlabel("Threshold")
    ax4.set_ylabel("F1 Score")
    ax4.set_title("F1 Score (best highlighted)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(thresholds)
    ax4.set_ylim(0.80, 0.85)
    for bar, val in zip(bars, f1):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.3f}", ha="center", fontsize=8)

    # --- Bottom middle: cost analysis (what you gain vs lose) ---
    ax5 = fig.add_subplot(gs[1, 1])
    # Climate caught vs total climate
    climate_caught_pct = [t / p * 100 for t, p in zip(tp, total_pos)]
    false_alarm_pct = [f / (t + f) * 100 if (t + f) > 0 else 0 for t, f in zip(tp, fp)]
    ax5.plot(thresholds, climate_caught_pct, "o-", color="#2196F3", linewidth=2, label="Climate caught (%)")
    ax5.plot(thresholds, [100 - p for p in false_alarm_pct], "s-", color="#4CAF50", linewidth=2, label="Purity (1 - false alarm %)")
    ax5.set_xlabel("Threshold")
    ax5.set_ylabel("%")
    ax5.set_title("Coverage vs Purity")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(thresholds)

    # --- Bottom right: summary table ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    table_data = []
    for m in metrics:
        table_data.append([
            f"{m['threshold']}",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['tp']}", f"{m['fp']}", f"{m['fn']}"
        ])
    table = ax6.table(
        cellText=table_data,
        colLabels=["Thresh", "Prec", "Recall", "F1", "TP", "FP", "FN"],
        loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)
    # Highlight best F1 row
    best_idx = np.argmax(f1)
    for col in range(7):
        table[best_idx + 1, col].set_facecolor("#C8E6C9")
    ax6.set_title("Validation Metrics", fontweight="bold", pad=20)

    fig.suptitle("FastText Validation Performance Across Thresholds",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(OUT_DIR / "validation_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [9/9] validation_dashboard.png")


def main():
    data = json.load(open(DATA))
    hist_probs, hist_kw, mod_probs, mod_kw = load_probs(data)

    print(f"Historical: {len(hist_probs)} chunks | Modern: {len(mod_probs)} chunks")
    print(f"Generating visualizations...")

    plot_prob_distribution(hist_probs, mod_probs)
    plot_pass_rate_vs_threshold(hist_probs, mod_probs)
    plot_cdf(hist_probs, mod_probs)
    plot_keyword_vs_prob(hist_probs, hist_kw, mod_probs, mod_kw)
    plot_prob_heatmap(hist_probs, mod_probs)
    plot_combined_dashboard(hist_probs, mod_probs, hist_kw, mod_kw)

    # Validation metrics
    val_metrics = load_validation_metrics()
    plot_validation_curves(val_metrics)
    plot_confusion_matrix_grid(val_metrics)
    plot_validation_dashboard(val_metrics)

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
