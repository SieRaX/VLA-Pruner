"""Offline analysis for run_conditional_patch_utility.py outputs.

All quantities are action-preservation errors (MSE to the full-input action chunk, normalized
action space) and differences of such errors. They are an empirical proxy for a patch's
incremental contribution; they are NOT mutual-information estimates.

Patch index p in [0, 512): view = p // 256 (0 = agentview, 1 = wrist), q = p % 256,
grid (row, col) = (q // 16, q % 16), row-major over the 224x224 model-input image.

Usage: python analyze_conditional_patch_utility.py --data <out_dir of the run> [--figures 12]
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

PER_VIEW, GRID, PS = 256, 16, 14
VIEW_NAMES = ["agentview", "wrist"]
LABEL_MIN_FRAC = 0.2  # a patch takes the dominant non-background category covering >= 20% of its pixels
# validated reference palette (dataviz skill): categorical slots 1-5, diverging blue <-> red, gray midpoint
C_XO, C_COND, C_GRIP, C_PRUNER, C_ORACLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
C_ORACLE_CROSS = "#4a3aa7"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def coord(p):
    q = int(p) % PER_VIEW
    return int(p) // PER_VIEW, q // GRID, q % GRID


def coord_str(p):
    v, r, c = coord(p)
    return f"{VIEW_NAMES[v]}(r{r},c{c})"


def load(path):
    d = np.load(path, allow_pickle=True)
    cats = [str(c) for c in d["categories"]]
    frac = np.concatenate([d["cat_frac_view0"], d["cat_frac_view1"]], axis=0)  # (512, C)
    fg = frac.copy()
    fg[:, cats.index("background")] = 0.0
    label_idx = np.where(fg.max(axis=1) >= LABEL_MIN_FRAC, fg.argmax(axis=1), cats.index("background"))
    labels = np.array([cats[i] for i in label_idx])
    n = int(d["num_patches"])
    E = np.full((n, n), np.nan, dtype=np.float64)
    idx = d["pair_idx"].astype(np.int64)
    E[idx[:, 0], idx[:, 1]] = d["mse_pair"]
    E[idx[:, 1], idx[:, 0]] = d["mse_pair"]
    E[np.arange(n), np.arange(n)] = d["mse_single"]
    return d, cats, frac, labels, E


def ranks_desc(values, exclude):
    """1 = largest value; excluded indices get NaN."""
    order = np.argsort(-np.where(np.isnan(values), -np.inf, values), kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # re-rank without the excluded entries
    keep = np.ones(len(values), dtype=bool)
    keep[list(exclude)] = False
    sub = np.argsort(-values[keep], kind="stable")
    r = np.empty(keep.sum())
    r[sub] = np.arange(1, keep.sum() + 1)
    ranks[:] = np.nan
    ranks[keep] = r
    return ranks


def analyze_frame(path, xo_view=None):
    """xo_view: None = best standalone target-object patch in any view; 0/1 = restrict X_o to that view."""
    d, cats, frac, labels, E = load(path)
    n = E.shape[0]
    e_none = float(d["mse_none"])
    e_single = np.diag(E).copy()
    u_stand = e_none - e_single

    target = np.where(labels == "target_object")[0]
    if xo_view is not None:
        target = target[target // PER_VIEW == xo_view]
    if target.size == 0:
        return None
    xo = int(target[np.argmin(e_single[target])])
    e_o = e_single[xo]
    e_cond = E[xo].copy()
    e_cond[xo] = np.nan
    u_cond = e_o - e_cond
    gain = u_cond - u_stand
    gain[xo] = np.nan
    # Raw G is biased negative because the two utilities are measured against different residuals
    # (MSE(none) >> MSE(X_o)). The relative versions are the fraction of the remaining error removed.
    u_stand_rel = u_stand / e_none
    u_cond_rel = u_cond / e_o
    gain_rel = u_cond_rel - u_stand_rel
    rank_s = ranks_desc(u_stand, [xo])
    rank_c = ranks_desc(np.where(np.isnan(u_cond), -np.inf, u_cond), [xo])

    def best_of(mask, score):
        ids = np.where(mask)[0]
        ids = ids[ids != xo]
        return int(ids[np.nanargmax(score[ids])]) if ids.size else None

    xg = best_of(labels == "gripper", u_cond)
    xo2 = best_of(labels == "target_object", u_cond)
    xbest = int(np.nanargmax(u_cond))
    iu = np.triu_indices(n, 1)
    flat = int(np.argmin(E[iu]))
    oracle = (int(iu[0][flat]), int(iu[1][flat]))
    cross = E[:PER_VIEW, PER_VIEW:]
    ci = np.unravel_index(int(np.argmin(cross)), cross.shape)
    oracle_cross = (int(ci[0]), int(ci[1]) + PER_VIEW)

    pruner = {}
    for b in (1, 2):
        key = f"pruner_k{b}_all__patches"
        if key in d:
            pruner[f"k{b}_all"] = {"patches": d[key].tolist(), "mse": float(d[f"pruner_k{b}_all__mse"])}
            for v in range(2):
                kv = f"pruner_k{b}_view{v}__patches"
                if kv in d:
                    pruner[f"k{b}_view{v}"] = {"patches": d[kv].tolist(), "mse": float(d[f"pruner_k{b}_view{v}__mse"])}

    frame = {
        "path": path, "task_id": int(d["task_id"]), "trial": int(d["trial"]), "query_idx": int(d["query_idx"]),
        "env_step": int(d["env_step"]), "task": str(d["task_description"]),
        "labels": labels, "e_none": e_none, "e_single": e_single, "u_stand": u_stand, "e_cond": e_cond,
        "u_cond": u_cond, "gain": gain, "u_stand_rel": u_stand_rel, "u_cond_rel": u_cond_rel, "gain_rel": gain_rel, "rank_s": rank_s, "rank_c": rank_c, "E": E,
        "xo": xo, "e_o": float(e_o), "xg": xg, "xo2": xo2, "xbest": xbest, "oracle": oracle,
        "oracle_cross": oracle_cross, "pruner": pruner,
        "mse": {
            "none": e_none,
            "target_only": float(e_o),
            "best_single": float(e_single.min()),
            "target+gripper": float(E[xo, xg]) if xg is not None else np.nan,
            "target+target": float(E[xo, xo2]) if xo2 is not None else np.nan,
            "target+best_conditional": float(E[xo, xbest]),
            "oracle_pair": float(E[oracle]),
            "oracle_pair_one_per_view": float(E[oracle_cross]),
            "pruner_1_per_view(2 patches)": pruner.get("k1_all", {}).get("mse", np.nan),
            "pruner_2_per_view(4 patches)": pruner.get("k2_all", {}).get("mse", np.nan),
            "pruner_2_agentview_only": pruner.get("k2_view0", {}).get("mse", np.nan),
            "pruner_2_wrist_only": pruner.get("k2_view1", {}).get("mse", np.nan),
        },
        "image0": d["image_view0"], "image1": d["image_view1"],
    }
    return frame


def write_frame_tables(frame, out_dir):
    tag = f"task{frame['task_id']:02d}_trial{frame['trial']:02d}_query{frame['query_idx']:03d}"
    labels = frame["labels"]
    rows = []
    for p in range(len(labels)):
        v, r, c = coord(p)
        rows.append({
            "patch": p, "view": VIEW_NAMES[v], "row": r, "col": c, "category": labels[p],
            "mse_standalone": frame["e_single"][p], "u_standalone": frame["u_stand"][p], "rank_standalone": frame["rank_s"][p],
            "mse_conditional": frame["e_cond"][p], "u_conditional": frame["u_cond"][p], "rank_conditional": frame["rank_c"][p],
            "interaction_gain": frame["gain"][p], "u_standalone_rel": frame["u_stand_rel"][p],
            "u_conditional_rel": frame["u_cond_rel"][p], "interaction_gain_rel": frame["gain_rel"][p], "is_xo": int(p == frame["xo"]),
        })
    with open(os.path.join(out_dir, "per_frame", f"{tag}_patches.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [f"## {tag} — {frame['task']} (env step {frame['env_step']})", "",
             f"X_o = patch {frame['xo']} {coord_str(frame['xo'])}; MSE none {frame['e_none']:.5f}, X_o alone {frame['e_o']:.5f}", ""]
    lines += ["| configuration | patches | categories | MSE(A_full, A_pair) |", "|---|---|---|---|"]

    def pair_row(name, patches, mse):
        if patches is None or any(p is None for p in patches):
            return f"| {name} | — | — | — |"
        return (f"| {name} | {', '.join(f'{p} {coord_str(p)}' for p in patches)} | "
                f"{' + '.join(labels[p] for p in patches)} | {mse:.5f} |")

    m = frame["mse"]
    lines.append(pair_row("target + best gripper", (frame["xo"], frame["xg"]), m["target+gripper"]))
    lines.append(pair_row("target + other target", (frame["xo"], frame["xo2"]), m["target+target"]))
    lines.append(pair_row("target + best conditional", (frame["xo"], frame["xbest"]), m["target+best_conditional"]))
    lines.append(pair_row("global oracle pair", frame["oracle"], m["oracle_pair"]))
    lines.append(pair_row("oracle pair, one per view", frame["oracle_cross"], m["oracle_pair_one_per_view"]))
    for key, name in (("k1_all", "VLA-Pruner 1/view (2 patches)"), ("k2_view0", "VLA-Pruner 2/view, agentview pair only"),
                      ("k2_view1", "VLA-Pruner 2/view, wrist pair only"), ("k2_all", "VLA-Pruner 2/view (4 patches)")):
        if key in frame["pruner"]:
            lines.append(pair_row(name, frame["pruner"][key]["patches"], frame["pruner"][key]["mse"]))
    for title, score in (("standalone utility", frame["u_stand"]), ("conditional utility given X_o", frame["u_cond"]),
                         ("interaction gain G", frame["gain"])):
        lines += ["", f"Top 10 by {title}:", "",
                  "| patch | coord | category | U_standalone | rank_s | U_conditional | rank_c | G |", "|---|---|---|---|---|---|---|---|"]
        s = np.where(np.isnan(score), -np.inf, score)
        s[frame["xo"]] = -np.inf
        for p in np.argsort(-s)[:10]:
            lines.append(f"| {p} | {coord_str(p)} | {labels[p]} | {frame['u_stand'][p]:.5f} | {frame['rank_s'][p]:.0f} | "
                         f"{frame['u_cond'][p]:.5f} | {frame['rank_c'][p]:.0f} | {frame['gain'][p]:.5f} |")
    with open(os.path.join(out_dir, "per_frame", f"{tag}_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def aggregate(frames, out_dir, name):
    cats = ["target_object", "gripper", "robot_arm", "goal_object", "other_object", "background"]
    per_cat = defaultdict(lambda: defaultdict(list))
    oracle_combo, pruner_combo = Counter(), Counter()
    stats = defaultdict(list)
    for fr in frames:
        labels = fr["labels"]
        for c in cats:
            ids = np.where(labels == c)[0]
            ids = ids[ids != fr["xo"]]
            if ids.size == 0:
                continue
            pc = per_cat[c]
            pc["u_stand"].append(np.nanmean(fr["u_stand"][ids]))
            pc["u_cond"].append(np.nanmean(fr["u_cond"][ids]))
            pc["gain"].append(np.nanmean(fr["gain"][ids]))
            pc["u_stand_rel"].append(np.nanmean(fr["u_stand_rel"][ids]))
            pc["u_cond_rel"].append(np.nanmean(fr["u_cond_rel"][ids]))
            pc["gain_rel"].append(np.nanmean(fr["gain_rel"][ids]))
            pc["rank_s"].append(np.nanmean(fr["rank_s"][ids]))
            pc["rank_c"].append(np.nanmean(fr["rank_c"][ids]))
            pc["best_rank_s"].append(np.nanmin(fr["rank_s"][ids]))
            pc["best_rank_c"].append(np.nanmin(fr["rank_c"][ids]))
            pc["in_oracle"].append(float(any(labels[p] == c for p in fr["oracle"])))
        oracle_combo[" + ".join(sorted(labels[p] for p in fr["oracle"]))] += 1
        for key in ("k1_all", "k2_view0", "k2_view1"):
            if key in fr["pruner"]:
                pruner_combo[(key, " + ".join(sorted(labels[p] for p in fr["pruner"][key]["patches"])))] += 1
        for k, v in fr["mse"].items():
            stats[k].append(v)
        if fr["xg"] is not None:
            stats["_grip_top1"].append(float(labels[fr["xbest"]] == "gripper"))
            stats["_grip_top5"].append(float(np.nanmin(fr["rank_c"][labels == "gripper"]) <= 5))
            stats["_grip_top10"].append(float(np.nanmin(fr["rank_c"][labels == "gripper"]) <= 10))
            if fr["xo2"] is not None:
                stats["_tg_beats_tt"].append(float(fr["mse"]["target+gripper"] < fr["mse"]["target+target"]))
                stats["_tg_minus_tt"].append(fr["mse"]["target+gripper"] - fr["mse"]["target+target"])

    n = len(frames)
    lines = [f"# Aggregate — {name} ({n} observations, {len({(f['task_id'], f['trial']) for f in frames})} episodes)", "",
             "MSE / utilities are action-preservation proxies in normalized action space, not mutual-information estimates.", "",
             "## Mean MSE(A_full, A_set)", "", "| configuration | mean | median | n |", "|---|---|---|---|"]
    for k, v in stats.items():
        if not k.startswith("_"):
            a = np.asarray(v, dtype=np.float64)
            a = a[~np.isnan(a)]
            if a.size:
                lines.append(f"| {k} | {a.mean():.5f} | {np.median(a):.5f} | {a.size} |")
    lines += ["", "## Key comparison", ""]
    for k, title in (("_grip_top1", "best conditional patch is a gripper patch"), ("_grip_top5", "a gripper patch in conditional top-5"),
                     ("_grip_top10", "a gripper patch in conditional top-10"),
                     ("_tg_beats_tt", "target+gripper has lower MSE than target+target")):
        if stats[k]:
            lines.append(f"- {title}: {100 * np.mean(stats[k]):.1f} % of {len(stats[k])} observations")
    if stats["_tg_minus_tt"]:
        diff = np.asarray(stats["_tg_minus_tt"])
        rng = np.random.default_rng(0)
        boot = [rng.choice(diff, diff.size).mean() for _ in range(5000)]
        lines.append(f"- mean MSE(target+gripper) − MSE(target+target): {diff.mean():+.5f} "
                     f"(bootstrap 95 % CI {np.percentile(boot, 2.5):+.5f} … {np.percentile(boot, 97.5):+.5f}; "
                     "observations within an episode are correlated, so treat the CI as optimistic)")
        try:
            from scipy.stats import wilcoxon
            lines.append(f"- Wilcoxon signed-rank p = {wilcoxon(diff).pvalue:.2e}")
        except Exception:
            pass
    lines += ["", "## Semantic regions (mean over observations of the per-observation category mean; X_o excluded)", "",
              "| category | U_standalone | U_conditional | G | U_stand rel | U_cond rel | G rel | rank_s | rank_c | rank change (s−c) | best rank_s | best rank_c | in oracle pair | n obs |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cats:
        pc = per_cat[c]
        if pc["u_stand"]:
            lines.append(f"| {c} | {np.mean(pc['u_stand']):.5f} | {np.mean(pc['u_cond']):.5f} | {np.mean(pc['gain']):+.5f} | "
                         f"{np.mean(pc['u_stand_rel']):.3f} | {np.mean(pc['u_cond_rel']):.3f} | {np.mean(pc['gain_rel']):+.3f} | "
                         f"{np.mean(pc['rank_s']):.1f} | {np.mean(pc['rank_c']):.1f} | {np.mean(pc['rank_s']) - np.mean(pc['rank_c']):+.1f} | "
                         f"{np.mean(pc['best_rank_s']):.1f} | {np.mean(pc['best_rank_c']):.1f} | {100 * np.mean(pc['in_oracle']):.1f} % | {len(pc['u_stand'])} |")
    lines += ["", "## Category make-up of the global oracle pair", "", "| categories | count | share |", "|---|---|---|"]
    for combo, cnt in oracle_combo.most_common():
        lines.append(f"| {combo} | {cnt} | {100 * cnt / n:.1f} % |")
    lines += ["", "## Category make-up of VLA-Pruner's retained patches", "", "| budget | categories | count |", "|---|---|---|"]
    for (key, combo), cnt in sorted(pruner_combo.items(), key=lambda kv: (kv[0][0], -kv[1])):
        lines.append(f"| {key} | {combo} | {cnt} |")
    with open(os.path.join(out_dir, f"aggregate_{name}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, f"aggregate_{name}.json"), "w") as f:
        json.dump({"n": n, "mse": {k: [float(x) for x in v] for k, v in stats.items()},
                   "oracle_combo": dict(oracle_combo)}, f)
    return "\n".join(lines)


def figure(frame, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    div = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])  # red <- 0 (gray) -> blue
    tag = f"task{frame['task_id']:02d}_trial{frame['trial']:02d}_query{frame['query_idx']:03d}"
    fig, axes = plt.subplots(2, 6, figsize=(26, 9.4), facecolor=SURFACE)
    titles = ["model input + patch grid", "standalone utility  U = MSE(none) − MSE(i)",
              "conditional utility  U = MSE(X_o) − MSE(X_o, i)", "interaction gain  G = U_cond − U_stand",
              "relative gain  U_cond/MSE(X_o) − U_stand/MSE(none)", "retained patches"]
    maps = [None, frame["u_stand"], frame["u_cond"], frame["gain"], frame["gain_rel"], None]

    def box(ax, p, color, label, inset=0.0, ls="-"):
        _, r, c = coord(p)
        ax.add_patch(plt.Rectangle((c * PS + inset, r * PS + inset), PS - 2 * inset, PS - 2 * inset, fill=False,
                                   edgecolor=color, linewidth=2.2, linestyle=ls))
        if False:
            ax.text(c * PS + PS / 2, r * PS - 2, label, color=INK, fontsize=7, ha="center", va="bottom",
                    bbox=dict(facecolor=SURFACE, edgecolor=color, linewidth=1, pad=1))

    for v in range(2):
        img = frame[f"image{v}"]
        for j in range(6):
            ax = axes[v, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d9d8d3")
            if v == 0:
                ax.set_title(titles[j], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(VIEW_NAMES[v], fontsize=10, color=INK2)
            if j in (0, 5):
                ax.imshow(img)
                if j == 0:
                    for g in range(1, GRID):
                        ax.axhline(g * PS - 0.5, color="white", lw=0.4, alpha=0.6)
                        ax.axvline(g * PS - 0.5, color="white", lw=0.4, alpha=0.6)
                    for g in range(0, GRID, 4):
                        ax.text(g * PS + 1, 6, f"c{g}", color="white", fontsize=6)
                        ax.text(1, g * PS + 12, f"r{g}", color="white", fontsize=6)
            else:
                m = maps[j][v * PER_VIEW:(v + 1) * PER_VIEW].reshape(GRID, GRID)
                lim = np.nanmax(np.abs(maps[j])) or 1.0
                ax.imshow(img, extent=(0, GRID, GRID, 0), alpha=0.25)
                im = ax.imshow(np.ma.masked_invalid(m), cmap=div, norm=TwoSlopeNorm(0.0, -lim, lim), alpha=0.85,
                               extent=(0, GRID, GRID, 0))
                if frame["xo"] // PER_VIEW == v:
                    _, r, c = coord(frame["xo"])
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor=INK, linewidth=2))
                    ax.text(c + 0.5, r - 0.15, "X_o", fontsize=7, ha="center", va="bottom", color=INK)
                if v == 1:
                    cb = fig.colorbar(im, ax=axes[:, j], orientation="horizontal", fraction=0.04, pad=0.03)
                    cb.ax.tick_params(labelsize=7, colors=INK2)
                    cb.outline.set_visible(False)
        ax = axes[v, 5]
        marks = [(frame["xo"], C_XO, "X_o", 0.0, "-"), (frame["xbest"], C_COND, "best cond", 1.5, "-"),
                 (frame["xg"], C_GRIP, "best gripper", 3.0, "-")]
        for p in frame["pruner"].get("k1_all", {}).get("patches", []):
            marks.append((p, C_PRUNER, "VLA-Pruner", 4.5, "--"))
        for p in frame["oracle"]:
            marks.append((p, C_ORACLE, "oracle", 0.0, ":"))
        for p in frame["oracle_cross"]:
            marks.append((p, C_ORACLE_CROSS, "oracle, one per view", -2.5, "-."))
        for p, color, label, inset, ls in marks:
            if p is not None and p // PER_VIEW == v:
                box(ax, p, color, label, inset, ls)
    handles = [plt.Line2D([], [], color=c, lw=2.2, ls=ls, label=l) for c, l, ls in
               ((C_XO, "fixed target patch X_o", "-"), (C_COND, "best conditional patch", "-"), (C_GRIP, "best gripper patch", "-"),
                (C_PRUNER, "VLA-Pruner (1 patch / view)", "--"), (C_ORACLE, "global oracle pair", ":"),
                (C_ORACLE_CROSS, "oracle pair, one per view", "-."))]
    fig.legend(handles=handles, loc="lower right", fontsize=9, frameon=False, ncol=6)
    m = frame["mse"]
    fig.suptitle(f"{tag} — {frame['task']}\nMSE to A_full:  none {m['none']:.4f} · X_o {m['target_only']:.4f} · target+gripper {m['target+gripper']:.4f} · "
                 f"target+target {m['target+target']:.4f} · oracle pair {m['oracle_pair']:.4f} · oracle one/view {m['oracle_pair_one_per_view']:.4f} · VLA-Pruner(1/view) {m['pruner_1_per_view(2 patches)']:.4f}",
                 fontsize=11, color=INK)
    fig.savefig(os.path.join(out_dir, "figures", f"{tag}.png"), dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", type=int, default=12, help="number of representative frames to render")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.data, "analysis")
    for sub in ("per_frame", "figures"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.data, "task*_trial*", "query*.npz")))
    print(f"{len(paths)} analysed observations")
    for variant, xo_view in (("xo_any_view", None), ("xo_agentview", 0), ("xo_wrist", 1)):
        frames = []
        for p in paths:
            fr = analyze_frame(p, xo_view)
            if fr is None:
                continue
            if variant == "xo_any_view":
                write_frame_tables(fr, out_dir)
            fr_light = {k: v for k, v in fr.items() if k != "E"}
            frames.append(fr_light)
        if not frames:
            continue
        print(aggregate(frames, out_dir, variant))
        if variant == "xo_agentview" and args.figures > 0:
            step = max(1, len(frames) // args.figures)
            for fr in frames[::step][: args.figures]:
                figure(fr, out_dir)


if __name__ == "__main__":
    main()
