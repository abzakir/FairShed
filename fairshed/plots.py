"""Matplotlib figures. Colours follow the entity, never the rank:
constraint-preserving QAOA = blue, penalty QAOA = orange, exact = aqua, greedy = yellow."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from .instance import Instance
from .qubo import index_to_bits, to_matrix

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
OFF_FILL, ON_FILL = "#184f95", "#f0efec"
COLORS = {"constrained": BLUE, "penalty": ORANGE, "exact": AQUA, "greedy": YELLOW}
LABELS = {"constrained": "Constraint-preserving QAOA", "penalty": "Penalty QAOA (textbook)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})


def _finish(fig, path):
    if path is not None:
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
    return fig


def draw_schedule(ax, inst: Instance, index: int, title: str, subtitle: str = "") -> None:
    X = to_matrix(inst, index_to_bits(index, inst.n_vars))
    F, T = X.shape
    ax.set_xlim(0, T)
    ax.set_ylim(F, 0)
    ax.grid(False)
    for f in range(F):
        for t in range(T):
            off = bool(X[f, t])
            ax.add_patch(FancyBboxPatch(
                (t + 0.05, f + 0.08), 0.90, 0.84, boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=OFF_FILL if off else ON_FILL, edgecolor="none"))
            ax.text(t + 0.5, f + 0.5, "OFF" if off else "on", ha="center", va="center",
                    color="#ffffff" if off else MUTED, fontsize=9,
                    fontweight="bold" if off else "normal")
    ax.set_xticks(np.arange(T) + 0.5, inst.slot_labels)
    ax.set_yticks(np.arange(F) + 0.5, [fd.name for fd in inst.feeders])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, -0.12, subtitle, fontsize=8.5, color=INK2, va="bottom")


def plot_schedules(inst, panels, path=None):
    """panels: list of (title, subtitle, index)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 3.1), sharey=True)
    for ax, (title, subtitle, index) in zip(np.atleast_1d(axes), panels):
        draw_schedule(ax, inst, index, title, subtitle)
    fig.tight_layout(w_pad=2.5)
    return _finish(fig, path)


def plot_distribution(probs, ref, path=None, top=14, title=""):
    """Most likely measured schedules; the bar of the true optimum is highlighted."""
    order = np.argsort(probs)[::-1][:top]
    optimal = ref.optimal_mask
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    colors = [AQUA if optimal[k] else BLUE for k in order]
    bars = ax.bar(range(len(order)), probs[order] * 100, color=colors, width=0.72)
    n_valid = int(ref.exact_cover.sum())
    line = ax.axhline(100 / n_valid, color=INK2, linewidth=1.3, linestyle=(0, (4, 3)))
    ax.legend(
        [Patch(color=AQUA), Patch(color=BLUE), line],
        ["exact optimum", "other valid schedule",
         f"picking a valid schedule at random ({100 / n_valid:.1f}%)"],
        loc="upper right", fontsize=8.5)
    for rect, k in zip(bars, order):
        if optimal[k]:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    f"{probs[k] * 100:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(order)), [f"{ref.energies[k]:.1f}" for k in order])
    ax.set_xlabel("objective value of the measured schedule (lower is better)")
    ax.set_ylabel("measurement probability (%)")
    ax.set_ylim(0, probs[order].max() * 100 * 1.45)
    ax.grid(axis="x", visible=False)
    ax.set_title(title or "Most likely measurement outcomes")
    return _finish(fig, path)


def plot_depth_sweep(rows, path=None):
    """rows: dicts with mode, p, p_optimal, p_exact_cover, noisy_p_optimal, noisy_p_exact_cover."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for mode in ("constrained", "penalty"):
        sel = [r for r in rows if r["mode"] == mode]
        ps = [r["p"] for r in sel]
        c = COLORS[mode]
        axes[0].plot(ps, [100 * r["p_optimal"] for r in sel], "-o", color=c,
                     label=f"{LABELS[mode]}, ideal", markeredgecolor=SURFACE, markeredgewidth=1.5)
        axes[0].plot(ps, [100 * r["noisy_p_optimal"] for r in sel], "--s", color=c,
                     label=f"{LABELS[mode]}, noisy", markeredgecolor=SURFACE, markeredgewidth=1.5)
        axes[1].plot(ps, [100 * r["p_exact_cover"] for r in sel], "-o", color=c,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
        axes[1].plot(ps, [100 * r["noisy_p_exact_cover"] for r in sel], "--s", color=c,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_title("Chance that one shot is the exact optimum")
    axes[0].set_ylabel("probability (%), log scale")
    axes[1].set_title("Chance that one shot covers the shortfall exactly")
    axes[1].set_ylabel("probability (%)")
    axes[1].set_ylim(0, 105)
    for ax in axes:
        ax.set_xlabel("QAOA layers p")
        ax.set_xticks(sorted({r["p"] for r in rows}))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.14), fontsize=9)
    fig.tight_layout()
    return _finish(fig, path)


def plot_tradeoff(rows, path=None):
    """rows: dicts with fairness_scale, exact_social, exact_worst, qaoa_social, qaoa_worst."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    x = np.arange(len(rows))
    labels = [f"{r['fairness_scale']:g}×" for r in rows]
    for ax, key, title in (
        (axes[0], "social", "Social cost of the schedule"),
        (axes[1], "worst", "Outage slots on the worst-hit feeder"),
    ):
        ax.plot(x, [r[f"exact_{key}"] for r in rows], "-", color=AQUA, label="Exact optimum")
        ax.plot(x, [r[f"qaoa_{key}"] for r in rows], "o", color=BLUE, markersize=9,
                markeredgecolor=SURFACE, markeredgewidth=2, label="QAOA best sample")
        ax.set_xticks(x, labels)
        ax.set_xlabel("fairness weight (multiple of default)")
        ax.set_title(title)
        lo = min(min(r[f"exact_{key}"], r[f"qaoa_{key}"]) for r in rows)
        hi = max(max(r[f"exact_{key}"], r[f"qaoa_{key}"]) for r in rows)
        ax.set_ylim(lo - 0.15 * (hi - lo + 1), hi + 0.15 * (hi - lo + 1))
    axes[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return _finish(fig, path)


def plot_landscape(gammas, betas, values, best=None, path=None):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    mesh = ax.pcolormesh(betas, gammas, values, cmap="Blues_r", shading="auto")
    ax.grid(False)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("expected objective ⟨H⟩ (lower is better)", color=INK2)
    cb.outline.set_visible(False)
    if best is not None:
        ax.plot([best[1]], [best[0]], "o", color=ORANGE, markersize=10,
                markeredgecolor="#ffffff", markeredgewidth=2)
        ax.annotate("optimiser result", (best[1], best[0]), xytext=(8, 8),
                    textcoords="offset points", color=INK, fontsize=9,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.85))
    ax.set_xlabel("mixer angle β")
    ax.set_ylabel("cost angle γ")
    ax.set_title("QAOA energy landscape at p = 1")
    return _finish(fig, path)


def plot_convergence(results, exact_min, path=None):
    """Optimiser trace for one depth per panel (small multiples, shared y-axis)."""
    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.2), sharey=True)
    for ax, res in zip(np.atleast_1d(axes), results):
        ax.plot(np.minimum.accumulate(res.history), color=BLUE)
        ax.axhline(exact_min, color=AQUA, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.set_title(f"p = {res.p}")
        ax.set_xlabel("circuit evaluations")
    np.atleast_1d(axes)[0].set_ylabel("best ⟨H⟩ so far")
    np.atleast_1d(axes)[-1].text(0.98, exact_min, "exact optimum ", ha="right", va="bottom",
                                 transform=np.atleast_1d(axes)[-1].get_yaxis_transform(),
                                 fontsize=8.5, color=INK2)
    fig.tight_layout()
    return _finish(fig, path)


def plot_architecture(path=None):
    """Architecture diagram of the hybrid quantum-classical pipeline."""
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 115)
    ax.set_ylim(-6, 54)
    ax.axis("off")
    W = 26

    def box(x, y, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), W, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor="#ffffff", edgecolor=color, linewidth=1.8))
        ax.add_patch(FancyBboxPatch((x, y + h - 1.6), W, 1.6,
                                    boxstyle="round,pad=0,rounding_size=0.6",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + 1.5, y + h - 4.4, title, fontsize=10, fontweight="bold", color=INK, va="center")
        for i, line in enumerate(lines):
            ax.text(x + 1.5, y + h - 8.0 - 3.0 * i, line, fontsize=8.3, color=INK2, va="center")

    def arrow(p0, p1, label="", side="top"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=1.5,
                                     color=INK2, shrinkA=0, shrinkB=0))
        if not label:
            return
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        if side == "top":
            ax.text(mx, my + 0.8, label, fontsize=8, color=INK2, ha="center", va="bottom")
        elif side == "left":
            ax.text(mx - 0.8, my, label, fontsize=8, color=INK2, ha="right", va="center")
        else:
            ax.text(mx + 0.8, my, label, fontsize=8, color=INK2, ha="left", va="center")

    x1, x2, x3, x4 = 1, 30.3, 59.6, 88.9
    top, bot, ht, hb = 34, 4, 17, 21
    box(x1, top, ht, "1  Grid inputs", ["Feeder blocks + social cost", "Shortfall per hour (MW)",
                                        "Fairness weights B, C"], MUTED)
    box(x2, top, ht, "2  QUBO model", ["x[f,t] = feeder f off in slot t", "Cost + fairness (quadratic)",
                                       "Rule: D_t feeders off per slot"], MUTED)
    box(x3, top, ht, "3  Ising mapping", ["x = (1 − Z) / 2", "H = Σ h·Z + Σ J·ZZ",
                                          "Qiskit SparsePauliOp"], BLUE)
    box(x4, top, ht, "4  QAOA circuit (Qiskit)", ["Dicke state per slot",
                                                  "p × [RZ, RZZ cost layer",
                                                  "      + XY mixer per slot]"], BLUE)
    box(x4, bot, hb, "5  Classical optimiser", ["Estimator primitive → ⟨H⟩", "COBYLA tunes γ, β",
                                                "multi-start + warm start", "keeps lowest ⟨H⟩"], MUTED)
    box(x3, bot, hb, "6  Decode + verify", ["Aer sampler → bitstrings", "bitstring → schedule",
                                            "pick lowest-cost sample", "check every constraint"], AQUA)
    box(x2, bot, hb, "Exact baseline", ["Enumerate all 2ⁿ schedules", "+ greedy priority list",
                                        "→ approximation ratio,", "    P(optimum)"], AQUA)
    box(x1, bot, hb, "7  Operator view", ["Hourly schedule heatmap", "Cost vs fairness trade-off",
                                          "Quantum vs exact vs greedy"], MUTED)

    y_mid = top + ht / 2
    arrow((x1 + W, y_mid), (x2, y_mid))
    arrow((x2 + W, y_mid), (x3, y_mid))
    arrow((x3 + W, y_mid), (x4, y_mid))
    arrow((x4 + 8, top), (x4 + 8, bot + hb), "⟨H⟩", "left")
    arrow((x4 + 18, bot + hb), (x4 + 18, top), "new γ, β", "right")
    arrow((x4, bot + 9), (x3 + W, bot + 9))
    arrow((x2 + W / 2, top), (x2 + W / 2, bot + hb), "same QUBO", "right")
    arrow((x2 + W, bot + 9), (x3, bot + 9))
    # verified schedule: from box 6, under the baseline box, up into box 7
    xs, xe, yl = x3 + W / 2, x1 + W / 2, -2.5
    ax.plot([xs, xs, xe], [bot, yl, yl], color=INK2, linewidth=1.5, solid_capstyle="round")
    arrow((xe, yl), (xe, bot))
    ax.text((xs + xe) / 2, yl + 0.7, "verified schedule", fontsize=8, color=INK2,
            ha="center", va="bottom")
    ax.text(1, 53, "Blue = quantum (Qiskit)   ·   Grey = classical   ·   Green = validation",
            fontsize=8.5, color=INK2, va="center")
    return _finish(fig, path)
