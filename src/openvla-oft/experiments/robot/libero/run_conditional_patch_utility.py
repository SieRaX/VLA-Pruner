"""Conditional patch utility for action-dependent interaction (OpenVLA-OFT, LIBERO).

Rolls out the unpruned policy (A_full is always executed, so visited states stay on the
successful-grasp distribution) and, on selected queries after the usual 3-query warm-up,
brute-forces the action-preservation tables through the unchanged oracle evaluator
(layers 0-3 dense, pruning after layer 3, non-visual tokens always kept):

  * no visual patch (the established keep-0 condition), every single patch (512),
    every unordered patch pair over both views (130,816);
  * the patches VLA-Pruner itself retains on the same observation at 1 and 2 patches/view
    (its temporal history is maintained separately per budget), evaluated through the
    same evaluator.

Per-patch semantic labels come from a MuJoCo segmentation render of the same sim state,
pushed through the same geometric preprocessing as the RGB input (rot180 -> 224 -> center crop).
Everything downstream (choice of X_o, conditional / interaction-gain maps, tables, figures)
is derived offline from the saved tables by analyze_conditional_patch_utility.py.

Usage (from src/openvla-oft):
  python experiments/robot/libero/run_conditional_patch_utility.py \
    --pretrained_checkpoint <ckpt> --out_dir <dir> --num_trials_per_task 2
"""

import copy
import dataclasses
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import draccus
import numpy as np
from libero.libero import benchmark
from PIL import Image

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.libero.run_libero_eval import (
    TASK_MAX_STEPS,
    GenerateConfig,
    initialize_model,
    prepare_observation,
    process_action,
)
from experiments.robot.openvla_utils import (
    OPENVLA_IMAGE_SIZE,
    _configure_attention_pruning,
    prepare_images_for_vla,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere

CATEGORIES = ["background", "target_object", "gripper", "robot_arm", "goal_object", "other_object"]
GRID = 16


@dataclass
class CPUConfig(GenerateConfig):
    out_dir: str = "./conditional_patch_utility"    # Output root (npz per analysed query)
    analyze_start_query: int = 4                     # First analysed query (1-based); queries 1-3 are warm-up
    analyze_stride: int = 2                          # Analyse every n-th query from analyze_start_query on
    max_analyzed_per_episode: int = 0                # 0 = no cap
    pair_batch_size: int = 512                       # Keep-sets per batched tail forward
    skip_pairs: bool = False                         # Smoke tests: singles only
    task_ids: str = ""                               # Comma-separated subset of task ids ("" = all)
    pruner_budgets: str = "1,2"                      # VLA-Pruner patches/view budgets to record


def geom_categories(env):
    """geom id -> category index, from MuJoCo geom/body names and the task's objects of interest."""
    inner = env.env
    model = inner.sim.model
    of_interest = list(getattr(inner, "obj_of_interest", []))
    target = of_interest[0] if of_interest else None
    goals = of_interest[1:]
    object_names = list(getattr(inner, "objects_dict", {}).keys()) + list(getattr(inner, "fixtures_dict", {}).keys())
    table = np.zeros(model.ngeom + 1, dtype=np.int64)  # last slot: id -1 (nothing hit)
    for gid in range(model.ngeom):
        gname = model.geom_id2name(gid) or ""
        bname = model.body_id2name(model.geom_bodyid[gid]) or ""
        name = f"{gname}|{bname}"
        if "gripper0" in name:
            cat = "gripper"
        elif "robot0" in name or "mount0" in name:
            cat = "robot_arm"
        elif target is not None and target in name:
            cat = "target_object"
        elif any(g in name for g in goals):
            cat = "goal_object"
        elif any(o in name for o in object_names if "table" not in o and "floor" not in o):
            cat = "other_object"
        else:
            cat = "background"
        table[gid] = CATEGORIES.index(cat)
    return table, {"target": target, "goals": goals, "objects": object_names}


def patch_category_fractions(env, geom_table, camera, cfg):
    """(256, n_categories) per-patch pixel fractions for one camera, aligned with the model input."""
    seg = env.env.sim.render(camera_name=camera, height=cfg.env_img_res, width=cfg.env_img_res, segmentation=True)
    geom_ids = seg[..., 1].astype(np.int64)
    obj_types = seg[..., 0].astype(np.int64)
    geom_ids[obj_types != 5] = -1  # mjOBJ_GEOM == 5
    cats = geom_table[geom_ids]  # -1 indexes the trailing "background" slot
    # sim.render has the same orientation as the env observation; the policy input is that
    # observation rotated by 180 deg (get_libero_image), so apply the same rotation here.
    cats = cats[::-1, ::-1]
    img = Image.fromarray(cats.astype(np.uint8)).resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.NEAREST)
    if cfg.center_crop:
        side = int(round(OPENVLA_IMAGE_SIZE * np.sqrt(0.9)))
        off = (OPENVLA_IMAGE_SIZE - side) // 2
        img = img.crop((off, off, off + side, off + side)).resize(
            (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.NEAREST
        )
    cats = np.asarray(img)
    ps = OPENVLA_IMAGE_SIZE // GRID
    blocks = cats.reshape(GRID, ps, GRID, ps).transpose(0, 2, 1, 3).reshape(GRID * GRID, ps * ps)
    return np.stack([(blocks == c).mean(axis=1) for c in range(len(CATEGORIES))], axis=1).astype(np.float32), cats


def set_mode(model, cfg, **overrides):
    """Re-point the model's pruning flags without touching its temporal history."""
    hist = getattr(model, "av_hist", None)
    _configure_attention_pruning(model, dataclasses.replace(cfg, **overrides))
    if hist is not None:
        model.av_hist = hist


def query(cfg_mode, model, observation, task_description, comps):
    processor, action_head, proprio_projector, noisy_action_projector = comps
    return get_action(
        cfg_mode,
        model,
        copy.deepcopy(observation),  # get_vla_action normalises obs["state"] in place
        task_description,
        processor=processor,
        action_head=action_head,
        proprio_projector=proprio_projector,
        noisy_action_projector=noisy_action_projector,
        use_film=cfg_mode.use_film,
        last_caches=None,
    )


@draccus.wrap()
def main(cfg: CPUConfig) -> None:
    assert not cfg.use_vla_cache, "run with --use_vla_cache False (same as the previous pruning experiments)"
    set_seed_everywhere(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "run.log")

    def log(msg):
        line = f"[{time.strftime('%F %T')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    base = dataclasses.replace(cfg, use_vla_cache=False, use_fastv=False, use_vla_pruner=False, use_oracle_pruner=False)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(base)
    comps = (processor, action_head, proprio_projector, noisy_action_projector)
    resize_size = get_image_resize_size(cfg)
    budgets = [int(b) for b in cfg.pruner_budgets.split(",") if b]
    per_view = model.vision_backbone.get_num_patches()
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({k: str(v) for k, v in dataclasses.asdict(cfg).items()}, f, indent=2)

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task_ids = [int(t) for t in cfg.task_ids.split(",") if t] or list(range(task_suite.n_tasks))
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    if cfg.max_steps_cap is not None:
        max_steps = min(max_steps, cfg.max_steps_cap)

    # Trial-major order: one episode of every task first, so partial runs already cover all tasks.
    for trial in range(cfg.num_trials_per_task):
        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            geom_table, names = geom_categories(env)
            ep_dir = os.path.join(cfg.out_dir, f"task{task_id:02d}_trial{trial:02d}")
            os.makedirs(ep_dir, exist_ok=True)
            log(f"task {task_id} trial {trial}: {task_description} | target={names['target']} goals={names['goals']}")

            env.reset()
            obs = env.set_init_state(initial_states[trial])
            hists = {b: deque(maxlen=max(1, int(cfg.vla_pruner_av_hist_w))) for b in budgets}
            action_queue = deque(maxlen=cfg.num_open_loop_steps)
            t, query_idx, analyzed, success = 0, 0, 0, False
            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue
                if len(action_queue) == 0:
                    query_idx += 1
                    observation, _ = prepare_observation(obs, resize_size)
                    observation["prev_images"] = [observation["full_image"], observation["wrist_image"]]

                    # 1) VLA-Pruner's own picks on this observation, one temporal history per budget.
                    pruner = {}
                    for b in budgets:
                        mode = dict(use_vla_pruner=True, fastv_r=1.0 - b / per_view, save_ranking_viz=True)
                        set_mode(model, base, **mode)
                        model.av_hist = hists[b]
                        _, caches, _, metrics = query(dataclasses.replace(base, **mode), model, observation, task_description, comps)
                        hists[b] = model.av_hist
                        info = caches.get("pruning_info") if isinstance(caches, dict) else None
                        md = caches.get("token_metadata") if isinstance(caches, dict) else None
                        kept = None
                        if isinstance(info, dict) and info.get("kept_indices") is not None and md is not None:
                            k = np.asarray(info["kept_indices"].detach().cpu()).ravel()
                            k = k[(k >= md["visual_token_start"]) & (k < md["visual_token_end"])] - md["visual_token_start"]
                            if k.size < per_view * 2:  # warm-up queries keep everything
                                kept = k
                        pruner[b] = {"kept": kept, "ranking": metrics.get("ranking_dump")}

                    # 2) Dense reference query (+ brute-force tables on analysed queries); A_full is executed.
                    start = cfg.analyze_start_query + trial % cfg.analyze_stride  # phase-shift later trials
                    do_analyze = (
                        query_idx >= start
                        and (query_idx - start) % cfg.analyze_stride == 0
                        and (cfg.max_analyzed_per_episode <= 0 or analyzed < cfg.max_analyzed_per_episode)
                        and all(pruner[b]["kept"] is not None for b in budgets)
                    )
                    mode = dict(use_oracle_pruner=True, fastv_r=1.0 - 2 / per_view, save_oracle_trace=False)
                    set_mode(model, base, **mode)
                    model._cpu_mode = True
                    model._cpu_request = None
                    if do_analyze:
                        extra = {}
                        for b in budgets:
                            kept = pruner[b]["kept"]
                            extra[f"pruner_k{b}_all"] = kept
                            for v in range(2):
                                kv = kept[kept // per_view == v]
                                if kv.size:
                                    extra[f"pruner_k{b}_view{v}"] = kv
                        model._cpu_request = {
                            "batch_size": cfg.pair_batch_size,
                            "pairs": not cfg.skip_pairs,
                            "extra_sets": extra,
                        }
                    t0 = time.time()
                    actions, caches, _, _ = query(dataclasses.replace(base, **mode), model, observation, task_description, comps)
                    model._cpu_request = None
                    model._cpu_mode = False
                    action_queue.extend(actions)

                    if do_analyze:
                        tables = caches["cpu_analysis"]
                        frac0, segmap0 = patch_category_fractions(env, geom_table, "agentview", cfg)
                        frac1, segmap1 = patch_category_fractions(env, geom_table, "robot0_eye_in_hand", cfg)
                        images = prepare_images_for_vla([observation["full_image"], observation["wrist_image"]], cfg)
                        arrays = {
                            "task_id": task_id, "trial": trial, "query_idx": query_idx, "env_step": t,
                            "task_description": task_description,
                            "categories": np.array(CATEGORIES),
                            "cat_frac_view0": frac0, "cat_frac_view1": frac1,
                            "segmap_view0": segmap0.astype(np.uint8), "segmap_view1": segmap1.astype(np.uint8),
                            "image_view0": np.asarray(images[0]), "image_view1": np.asarray(images[1]),
                            "proprio": np.asarray(observation["state"], dtype=np.float32),
                            "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                            "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                        }
                        for key, value in tables.items():
                            if key != "extra":
                                arrays[key] = value
                        for name, entry in tables["extra"].items():
                            for field, value in entry.items():
                                arrays[f"{name}__{field}"] = value
                        for b in budgets:
                            ranking = pruner[b]["ranking"]
                            if isinstance(ranking, dict):
                                for field, value in ranking.items():
                                    if value is not None and "image" not in field:
                                        arrays[f"pruner_k{b}_ranking__{field}"] = np.asarray(value)
                        np.savez_compressed(os.path.join(ep_dir, f"query{query_idx:03d}.npz"), **arrays)
                        analyzed += 1
                        best = int(np.argmin(tables["mse_pair"])) if "mse_pair" in tables else -1
                        log(
                            f"  query {query_idx} (env step {t}): {time.time() - t0:.0f}s | mse none {tables['mse_none']:.5f} "
                            f"best single {tables['mse_single'].min():.5f}"
                            + (f" best pair {tables['mse_pair'][best]:.5f} {tables['pair_idx'][best].tolist()}" if best >= 0 else "")
                            + "".join(
                                f" | pruner k{b} {pruner[b]['kept'].tolist()} mse {tables['extra'][f'pruner_k{b}_all']['mse']:.5f}"
                                for b in budgets
                            )
                        )

                action = process_action(action_queue.popleft(), cfg.model_family, env=env)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1

            with open(os.path.join(ep_dir, "meta.json"), "w") as f:
                json.dump(
                    {"task_id": task_id, "trial": trial, "task_description": task_description, "success": success,
                     "num_queries": query_idx, "num_analyzed": analyzed, **names},
                    f, indent=2,
                )
            log(f"  episode done: success={success} queries={query_idx} analysed={analyzed}")
            env.close()
    log("ALL DONE")


if __name__ == "__main__":
    main()
