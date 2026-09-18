"""Sanity tests for the oracle patch-selection path (run before any oracle eval).

Checks, on one real LIBERO-Spatial observation:
  1. Forward equivalence: FastV's pruned action == oracle tail-runner action when the
     oracle is forced to evaluate FastV's exact keep-set (catches boundary-layer,
     mask, and RoPE-position bugs).
  2. Reverse equivalence: the oracle-selected keep-set replayed through fastv_forward
     (forced_visual_indices) reproduces the oracle's executed action.
  3. k=256 oracle exit == dense baseline action; all-512 forced tail == dense.
  4. Greedy structure: per-view quota respected; per-query latency printed.

Usage (from src/openvla-oft):
  PYTHONPATH=/PublicSSD/cspark/LIBERO MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 \
    python experiments/robot/libero/test_oracle_equivalence.py
"""

import copy
import time

import numpy as np

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
from experiments.robot.openvla_utils import _configure_attention_pruning
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
from libero.libero import benchmark

K = 2
FASTV_R = 1.0 - K / 256.0


def make_cfg(**overrides):
    cfg = GenerateConfig(
        pretrained_checkpoint="checkpoints/openvla-7b-oft-finetuned-libero-spatial",
        task_suite_name="libero_spatial",
        use_vla_cache=False,
        num_trials_per_task=1,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def main():
    set_seed_everywhere(7)
    base_cfg = make_cfg()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(base_cfg)
    resize_size = get_image_resize_size(base_cfg)

    bench = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = bench.get_task(0)
    init_states = bench.get_task_init_states(0)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    observation, img = prepare_observation(obs, resize_size)
    observation["prev_images"] = [img, observation["wrist_image"]]

    def query(cfg, tag):
        cfg.unnorm_key = base_cfg.unnorm_key  # resolved by initialize_model/check_unnorm_key
        _configure_attention_pruning(model, cfg)
        model.reset_av_history()
        start = time.time()
        # get_vla_action normalizes obs["state"] IN PLACE; deep-copy so every query sees
        # the identical observation (the real eval loop builds a fresh obs dict per step).
        actions, last_caches, _, _ = get_action(
            cfg,
            model,
            copy.deepcopy(observation),
            task_description,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
            use_film=cfg.use_film,
            last_caches=None,
        )
        elapsed = time.time() - start
        print(f"[{tag}] query took {elapsed:.2f}s")
        return np.stack([np.asarray(a) for a in actions]), last_caches, elapsed

    # --- Dense baseline ------------------------------------------------------
    a_dense, _, _ = query(make_cfg(), "dense")

    # --- 1. Forward equivalence: fastv keep-set through the oracle tail ------
    a_fastv, lc_fastv, _ = query(make_cfg(use_fastv=True, fastv_r=FASTV_R), "fastv")
    pi = lc_fastv["pruning_info"]
    tm = lc_fastv["token_metadata"]
    vs, ve = int(tm["visual_token_start"]), int(tm["visual_token_end"])
    kept = pi["kept_indices"].cpu().numpy()
    fastv_visual = kept[(kept >= vs) & (kept < ve)]
    print(f"fastv kept {len(fastv_visual)} visual tokens: {fastv_visual.tolist()}")
    assert len(fastv_visual) == K * int(tm["num_images"]), "fastv keep count mismatch"

    model._oracle_forced_keepset = fastv_visual.tolist()
    a_forced, _, _ = query(make_cfg(use_oracle_pruner=True, fastv_r=FASTV_R), "oracle-forced")
    model._oracle_forced_keepset = None
    diff_fwd = float(np.abs(a_forced - a_fastv).max())
    print(f"[1. forward equivalence] max|oracle_forced - fastv| = {diff_fwd:.6f}")
    assert diff_fwd < 1e-2, "forward equivalence FAILED (boundary/mask/position bug)"

    # --- 3a. all-512 forced tail vs dense ------------------------------------
    model._oracle_forced_keepset = list(range(vs, ve))
    a_all, _, _ = query(make_cfg(use_oracle_pruner=True, fastv_r=FASTV_R), "oracle-all512")
    model._oracle_forced_keepset = None
    diff_all = float(np.abs(a_all - a_dense).max())
    print(f"[3a. all-512 tail vs dense] max|Δ| = {diff_all:.6f}")
    if diff_all >= 1e-2:
        # Isolate: is it MY tail, or does the repo's own fastv continuation deviate from
        # dense even with zero pruning? Push all 512 through fastv_forward itself.
        model.fastv_forced_visual_indices = list(range(vs, ve))
        model.config.fastv_forced_visual_indices = model.fastv_forced_visual_indices
        a_fastv_all, _, _ = query(make_cfg(use_fastv=True, fastv_r=FASTV_R), "fastv-forced-all512")
        model.fastv_forced_visual_indices = None
        model.config.fastv_forced_visual_indices = None
        d_tail_vs_fastvall = float(np.abs(a_all - a_fastv_all).max())
        d_fastvall_vs_dense = float(np.abs(a_fastv_all - a_dense).max())
        print(f"    [probe] oracle_tail_all512 vs fastv_forward_all512: max|Δ| = {d_tail_vs_fastvall:.6f}")
        print(f"    [probe] fastv_forward_all512 vs dense:              max|Δ| = {d_fastvall_vs_dense:.6f}")
        print(f"    [probe] per-step max|oracle_all - dense| by action dim:")
        print(np.abs(a_all - a_dense).max(axis=0))
    assert diff_all < 1e-2, "all-512 tail does not reproduce dense action"

    # --- 3b. k=256 oracle exit == dense --------------------------------------
    a_k256, lc_k256, _ = query(make_cfg(use_oracle_pruner=True, fastv_r=0.0), "oracle-k256")
    assert lc_k256["pruning_info"] is None, "k=256 should skip pruning entirely"
    diff_k256 = float(np.abs(a_k256 - a_dense).max())
    print(f"[3b. k=256 exit vs dense] max|Δ| = {diff_k256:.6f}")
    assert diff_k256 < 1e-2, "k=256 exit does not reproduce dense action"

    # --- 2 + 4. Pure greedy run, quota check, reverse equivalence ------------
    a_oracle, lc_oracle, greedy_time = query(
        make_cfg(use_oracle_pruner=True, fastv_r=FASTV_R, oracle_warmup_queries=0), "oracle-greedy"
    )
    pi_o = lc_oracle["pruning_info"]
    chosen = pi_o["oracle_chosen"]
    views = pi_o["oracle_chosen_view"]
    print(f"greedy chose (order): {chosen.tolist()} views {views.tolist()}")
    print(f"round scores: {pi_o['oracle_round_scores'].tolist()}")
    print(f"L1 gap vs full action: {pi_o['oracle_l1_gap']:.5f}")
    counts = np.bincount(views, minlength=int(tm["num_images"]))
    assert (counts == K).all(), f"per-view quota violated: {counts.tolist()}"

    model.fastv_forced_visual_indices = sorted(int(g) for g in chosen)
    model.config.fastv_forced_visual_indices = model.fastv_forced_visual_indices
    a_rev, _, _ = query(make_cfg(use_fastv=True, fastv_r=FASTV_R), "fastv-forced-reverse")
    model.fastv_forced_visual_indices = None
    model.config.fastv_forced_visual_indices = None
    diff_rev = float(np.abs(a_rev - a_oracle).max())
    print(f"[2. reverse equivalence] max|fastv_forced - oracle| = {diff_rev:.6f}")
    assert diff_rev < 1e-2, "reverse equivalence FAILED"

    n_queries = 28
    print(
        f"\nALL CHECKS PASSED. Pure-greedy k={K} query latency: {greedy_time:.1f}s "
        f"-> ~{greedy_time * n_queries / 60:.1f} min per full-length episode, "
        f"~{greedy_time * n_queries * 100 / 3600:.1f} h per 100-episode run (upper bound; "
        "successful episodes are shorter)."
    )


if __name__ == "__main__":
    main()
