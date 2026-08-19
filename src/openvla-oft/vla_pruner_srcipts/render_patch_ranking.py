"""Renders patch-importance ranking visualizations from ranking dumps.

Input: a dump directory produced by `run_libero_eval.py --save_ranking_viz True`
(one subdir per episode, one npz per model query step). For every step and camera
view it computes the *survival order* of the 256 visual patches — the order in
which patches would drop out if we pruned them one by one — and renders:

  - step{t}_rank.png   survival-order heatmap for both views (bright = survives
                       longest; #1..#10 numbered, #1 outlined)
  - strip.png          temporal strip: survival heatmaps at sampled steps
  - filmstrip.png      one step shown at several retention levels (kept patches
                       visible, pruned patches blacked out)

Survival order under `semantic_action`: the selector keeps the union of the
top-k patches by semantic (prefill) attention and top-k by (temporally smoothed)
action attention, then trims redundant candidates. A patch therefore stays alive
down to keep-count k iff min(rank_semantic, rank_action) < k, so the survival
level is min of the two ranks; ties are broken by the larger normalized score.
(The redundancy trim can locally reorder equal-level patches; this is exact up
to that trim.)

Usage:
    python vla_pruner_srcipts/render_patch_ranking.py --dump_dir experiments/logs/ranking_viz/<run_id>
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Rectangle

VIEW_NAMES = ["third-person", "wrist"]
RETENTIONS = [1.0, 0.5, 0.25, 0.125, 0.0725]


def _rank_desc(scores):
    """rank 0 = highest score."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def _normalize(scores):
    scores = scores.astype(np.float64)
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-12)


def survival_rank(prefill, action):
    """Returns per-patch survival rank: 0 = the patch that survives longest."""
    rank_p = _rank_desc(prefill)
    if action is None:
        level = rank_p
        tie = _normalize(prefill)
    else:
        level = np.minimum(rank_p, _rank_desc(action))
        tie = np.maximum(_normalize(prefill), _normalize(action))
    order = np.lexsort((-tie, level))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def load_step(path):
    data = np.load(path)
    per_img = int(data["patches_per_image"])
    images = data["images"]
    num_views = images.shape[0]
    prefill = data["prefill_scores"]
    action = data["action_scores"] if "action_scores" in data else None
    views = []
    for v in range(num_views):
        sl = slice(v * per_img, (v + 1) * per_img)
        p = prefill[sl]
        a = action[sl] if action is not None and action.shape[0] >= (v + 1) * per_img else None
        kept = data[f"kept_view{v}"] if f"kept_view{v}" in data else None
        views.append(
            {
                "image": images[v],
                "ranks": survival_rank(p, a),
                "prefill": p,
                "action": a,
                "kept": kept,
            }
        )
    return {"env_step": int(data["env_step"]), "views": views}


def rank_overlay(image, ranks, alpha=0.55, grid=16, cmap_name="inferno"):
    """Blends a per-patch survival-order colormap over the image."""
    n = ranks.size
    colors = matplotlib.colormaps[cmap_name](1.0 - ranks / max(n - 1, 1))[:, :3]
    patch_px = image.shape[0] // grid
    color_img = np.kron(
        colors.reshape(grid, grid, 3), np.ones((patch_px, patch_px, 1))
    )
    base = image.astype(np.float64) / 255.0
    return (1 - alpha) * base + alpha * color_img


def draw_rank_panel(ax, view, title, top_n=10, grid=16, show_kept=False):
    img = view["image"]
    patch_px = img.shape[0] // grid
    ax.imshow(rank_overlay(img, view["ranks"]))
    if show_kept and view["kept"] is not None:
        # Cyan dots = patches the pruner actually kept this step (the diversity
        # trim can deviate from pure score order among near-tied candidates).
        rows, cols = np.divmod(view["kept"], grid)
        ax.scatter(cols * patch_px + patch_px / 2, rows * patch_px + patch_px / 2,
                   s=6, c="cyan", marker=".", linewidths=0)
    order = np.argsort(view["ranks"])
    for i in range(min(top_n, order.size)):
        pid = order[i]
        row, col = divmod(int(pid), grid)
        x, y = col * patch_px, row * patch_px
        if i == 0:
            ax.add_patch(Rectangle((x, y), patch_px, patch_px, fill=False, edgecolor="white", linewidth=2.0))
        ax.text(
            x + patch_px / 2,
            y + patch_px / 2,
            str(i + 1),
            color="white",
            fontsize=7 if i else 8,
            fontweight="bold",
            ha="center",
            va="center",
            path_effects=None,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def retention_panel(view, retention, grid=16):
    """Image with only the top `retention` fraction of patches visible."""
    img = view["image"].astype(np.float64) / 255.0
    n = view["ranks"].size
    num_keep = int(round(n * retention))
    keep = (view["ranks"] < num_keep).reshape(grid, grid)
    patch_px = img.shape[0] // grid
    mask = np.kron(keep, np.ones((patch_px, patch_px), dtype=bool))
    out = img.copy()
    out[~mask] = 0.0
    return out


def kept_overlap(view):
    """Fraction of the actually-kept patches that the ranking's top-|kept| predicts."""
    kept = view["kept"]
    if kept is None or kept.size == 0:
        return None
    predicted = set(np.argsort(view["ranks"])[: kept.size].tolist())
    return len(predicted & set(kept.tolist())) / kept.size


def render_episode(ep_dir, out_dir, meta, strip_cols, step_stride):
    steps = sorted(glob.glob(os.path.join(ep_dir, "step*.npz")))
    if not steps:
        return None
    os.makedirs(out_dir, exist_ok=True)
    loaded = [load_step(p) for p in steps]
    task = meta.get("task_description", os.path.basename(ep_dir))
    label = "success" if meta.get("success") else "failure"
    overlaps = []

    # Per-step survival heatmaps
    for idx, step in enumerate(loaded):
        for view in step["views"]:
            ov = kept_overlap(view)
            if ov is not None:
                overlaps.append(ov)
        if idx % step_stride:
            continue
        num_views = len(step["views"])
        fig, axes = plt.subplots(1, num_views, figsize=(4.4 * num_views, 4.8))
        axes = np.atleast_1d(axes)
        for v, view in enumerate(step["views"]):
            name = VIEW_NAMES[v] if v < len(VIEW_NAMES) else f"view {v}"
            draw_rank_panel(axes[v], view, f"{name} — env step {step['env_step']}", show_kept=True)
        sm = cm.ScalarMappable(cmap="inferno")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.tolist(), fraction=0.03, pad=0.02)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["pruned first", "survives last"])
        fig.suptitle(f"{task} ({label}) — patch survival order", fontsize=10)
        fig.savefig(os.path.join(out_dir, f"step{step['env_step']:03d}_rank.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Temporal strip
    num_views = len(loaded[0]["views"])
    cols = min(strip_cols, len(loaded))
    pick = np.unique(np.linspace(0, len(loaded) - 1, cols).round().astype(int))
    fig, axes = plt.subplots(num_views, len(pick), figsize=(2.3 * len(pick), 2.5 * num_views), squeeze=False)
    for c, si in enumerate(pick):
        step = loaded[si]
        for v in range(num_views):
            draw_rank_panel(axes[v][c], step["views"][v], f"t={step['env_step']}", top_n=1)
            if c == 0:
                axes[v][c].set_ylabel(VIEW_NAMES[v] if v < len(VIEW_NAMES) else f"view {v}", fontsize=9)
                axes[v][c].yaxis.set_visible(True)
                axes[v][c].set_yticks([])
    fig.suptitle(f"{task} ({label}) — survival order over time (bright = survives longest)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "strip.png"), dpi=150)
    plt.close(fig)

    # Retention filmstrip at the middle step
    mid = loaded[len(loaded) // 2]
    fig, axes = plt.subplots(num_views, len(RETENTIONS), figsize=(2.3 * len(RETENTIONS), 2.5 * num_views), squeeze=False)
    for c, ret in enumerate(RETENTIONS):
        for v in range(num_views):
            axes[v][c].imshow(retention_panel(mid["views"][v], ret))
            axes[v][c].set_xticks([])
            axes[v][c].set_yticks([])
            if v == 0:
                axes[v][c].set_title(f"keep {ret * 100:.1f}%", fontsize=9)
    fig.suptitle(f"{task} ({label}) — progressive pruning at env step {mid['env_step']}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, "filmstrip.png"), dpi=150)
    plt.close(fig)

    return {
        "episode": os.path.basename(ep_dir),
        "task": task,
        "success": bool(meta.get("success")),
        "num_query_steps": len(loaded),
        "mean_kept_overlap": float(np.mean(overlaps)) if overlaps else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True, help="Run dir containing ep*/step*.npz dumps")
    parser.add_argument("--out_dir", default=None, help="Output dir (default: <dump_dir>/render)")
    parser.add_argument("--strip_cols", type=int, default=8, help="Max columns in the temporal strip")
    parser.add_argument("--step_stride", type=int, default=1, help="Render every Nth per-step heatmap")
    args = parser.parse_args()

    out_root = args.out_dir or os.path.join(args.dump_dir, "render")
    summaries = []
    for ep_dir in sorted(glob.glob(os.path.join(args.dump_dir, "ep*"))):
        if not os.path.isdir(ep_dir):
            continue
        meta_path = os.path.join(ep_dir, "meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        summary = render_episode(ep_dir, os.path.join(out_root, os.path.basename(ep_dir)), meta, args.strip_cols, args.step_stride)
        if summary:
            summaries.append(summary)
            overlap = summary["mean_kept_overlap"]
            overlap_str = f"{overlap * 100:.1f}%" if overlap is not None else "n/a"
            print(f"{summary['episode']}: {summary['num_query_steps']} steps rendered, "
                  f"ranking-vs-kept overlap {overlap_str}")

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote renders for {len(summaries)} episodes to {out_root}")


if __name__ == "__main__":
    main()
