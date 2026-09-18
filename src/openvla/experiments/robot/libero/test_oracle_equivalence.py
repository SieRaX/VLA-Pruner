"""Sanity tests for the OpenVLA oracle patch-selection path (run before any oracle eval).

On one real LIBERO-Spatial observation:
  1. Forward equivalence: FastV's pruned action == oracle action when the oracle is forced
     to evaluate FastV's exact keep-set (discrete action tokens -> exact equality expected).
  2. Reverse equivalence: oracle-selected keep-set replayed through fastv_forward.
  3. All-256 forced oracle == dense (use_fastv=False) action.
  4. Pure-greedy k=2 run: quota check + measured latency.

Usage (from src/openvla):  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python experiments/robot/libero/test_oracle_equivalence.py
"""
import copy
import time

import numpy as np
import torch

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image, quat2axisangle
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import get_action, get_image_resize_size, get_model, set_seed_everywhere
from libero.libero import benchmark

K = 2
FASTV_R = 1.0 - K / 256.0


def configure(model, **attrs):
    for obj in (model, model.config):
        for key, value in attrs.items():
            setattr(obj, key, value)


def main():
    set_seed_everywhere(7)
    cfg = GenerateConfig(use_fastv=False, num_trials_per_task=1)
    cfg.unnorm_key = cfg.task_suite_name
    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    processor = get_processor(cfg)
    resize_size = get_image_resize_size(cfg)
    configure(model, fastv_k=cfg.fastv_k, fastv_image_token_start_index=cfg.fastv_image_token_start_index,
              fastv_image_token_length=cfg.fastv_image_token_length, use_text_vision_selection=False,
              use_prefil_attention=False, oracle_batch_size=64, oracle_warmup_queries=0)
    model.fastv_forced_visual_indices = None
    model._oracle_forced_keepset = None

    bench = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = bench.get_task(0)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    obs = env.set_init_state(bench.get_task_init_states(0)[0])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    img = get_libero_image(obs, resize_size)
    observation = {
        "full_image": img, "prev_image": img,
        "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
    }

    def query(tag, **attrs):
        configure(model, use_fastv=False, use_oracle_pruner=False, use_temporal=False, fastv_r=FASTV_R)
        configure(model, **attrs)
        model.reset_av_history()
        model.language_model.pruning_info = None
        start = time.time()
        action, _, _ = get_action(cfg, model, copy.deepcopy(observation), task_description, processor=processor, last_caches=None)
        dt = time.time() - start
        print(f"[{tag}] {dt:.2f}s action={np.round(np.asarray(action), 4).tolist()}")
        return np.asarray(action, dtype=np.float64), dict(model.language_model.pruning_info or {}), dt

    vs, ve = cfg.fastv_image_token_start_index, cfg.fastv_image_token_start_index + cfg.fastv_image_token_length

    a_dense, _, _ = query("dense")

    a_fastv, pi, _ = query("fastv", use_fastv=True)
    kept = pi["kept_indices"].cpu().numpy()
    fastv_visual = kept[(kept >= vs) & (kept < ve)]
    print(f"fastv kept {len(fastv_visual)} visual tokens: {fastv_visual.tolist()}")
    assert len(fastv_visual) == K

    model._oracle_forced_keepset = fastv_visual.tolist()
    a_forced, _, _ = query("oracle-forced", use_oracle_pruner=True)
    model._oracle_forced_keepset = None
    d1 = float(np.abs(a_forced - a_fastv).max())
    print(f"[1. forward equivalence] max|oracle_forced - fastv| = {d1:.6f}")
    assert d1 < 1e-6, "forward equivalence FAILED"

    model._oracle_forced_keepset = list(range(vs, ve))
    a_all, _, _ = query("oracle-all256", use_oracle_pruner=True)
    model._oracle_forced_keepset = None
    d3 = float(np.abs(a_all - a_dense).max())
    print(f"[3. all-256 forced vs dense] max|Δ| = {d3:.6f}")
    assert d3 < 1e-6, "all-256 oracle does not reproduce dense action"

    a_oracle, pi_o, greedy_time = query("oracle-greedy", use_oracle_pruner=True)
    chosen = pi_o["oracle_chosen"]
    print(f"greedy chose (order): {chosen.tolist()}  round scores: {pi_o['oracle_round_scores'].tolist()}  L1 gap {pi_o['oracle_l1_gap']:.4f}")
    assert len(chosen) == K and all(vs <= g < ve for g in chosen)

    model.fastv_forced_visual_indices = sorted(int(g) for g in chosen)
    model.config.fastv_forced_visual_indices = model.fastv_forced_visual_indices
    a_rev, _, _ = query("fastv-forced-reverse", use_fastv=True)
    model.fastv_forced_visual_indices = None
    model.config.fastv_forced_visual_indices = None
    d2 = float(np.abs(a_rev - a_oracle).max())
    print(f"[2. reverse equivalence] max|fastv_forced - oracle| = {d2:.6f}")
    assert d2 < 1e-6, "reverse equivalence FAILED"

    # --- 5. Teacher-forced surrogate consistency -----------------------------------
    # With all 256 patches kept, the argmax of each teacher-forced next-token distribution
    # must reproduce the dense action tokens exactly, and the expected action must be close.
    import torch as _t
    with _t.no_grad():
        inputs = processor(f"In: What action should the robot take to {task_description.lower()}?\nOut:",
                           __import__("PIL.Image", fromlist=["Image"]).fromarray(observation["full_image"])).to(
                               "cuda:0", dtype=_t.bfloat16)
        ids = inputs["input_ids"]
        if not _t.all(ids[:, -1] == 29871):
            ids = _t.cat((ids, _t.tensor([[29871]], device=ids.device)), dim=1)
        embeds, mm_mask = model._oracle_multimodal_inputs(ids, inputs["pixel_values"], inputs["attention_mask"])
        L0 = embeds.shape[1]
        boundary, prefix_cache = model.language_model.oracle_prefix_forward(embeds, mm_mask, cfg.fastv_k)
        n_tok = model.get_action_dim(cfg.unnorm_key)
        all_rows = _t.arange(L0, device=embeds.device).unsqueeze(0)
        full_tokens = model._oracle_generate_batch(boundary, prefix_cache, all_rows, cfg.fastv_k, n_tok)
        app_hidden = model.language_model.oracle_append_prefix(prefix_cache, full_tokens[:, : n_tok - 1], cfg.fastv_k)
        logits = model.language_model.oracle_tail_teacher_forced(boundary, app_hidden, all_rows, cfg.fastv_k, L0)
        tf_argmax = logits.argmax(dim=-1)[0]
        print(f"[5. teacher-forced @ keep-256] argmax tokens {tf_argmax.tolist()} vs dense {full_tokens[0].tolist()}")
        assert _t.equal(tf_argmax, full_tokens[0]), "teacher-forced argmax != dense action tokens"
        lut = model._oracle_center_lut(embeds.device)
        v_hi = int(model.vocab_size)
        expected = (_t.softmax(logits[:, :, v_hi - 256 : v_hi], dim=-1) @ lut)[0].cpu().numpy()
        full_norm = model._oracle_tokens_to_normalized(full_tokens)[0]
        print(f"    expected-action L1 to dense bin centers = {np.abs(expected - full_norm).mean():.4f}")
        # Same check on fastv's keep-set: teacher-forced argmax vs exact AR decode of that set
        rows = _t.cat((_t.cat((_t.arange(0, vs, device=embeds.device), _t.arange(ve, L0, device=embeds.device))),
                       _t.tensor(fastv_visual.tolist(), device=embeds.device))).sort().values.unsqueeze(0)
        exact = model._oracle_generate_batch(boundary, prefix_cache, rows, cfg.fastv_k, n_tok)[0]
        tf2 = model.language_model.oracle_tail_teacher_forced(boundary, app_hidden, rows, cfg.fastv_k, L0).argmax(-1)[0]
        agree = int((tf2 == exact).sum())
        print(f"    fastv keep-set: teacher-forced argmax agrees with exact AR decode on {agree}/{n_tok} tokens (first token must match: {bool(tf2[0]==exact[0])})")
        assert bool(tf2[0] == exact[0]), "first-token prediction must be identical under teacher forcing"

    # --- 6. Latency at k=5 (surrogate greedy) ----------------------------------------
    _, pi5, t5 = query("oracle-greedy-k5", use_oracle_pruner=True, fastv_r=1.0 - 5 / 256.0)
    print(f"k=5 chose {pi5['oracle_chosen'].tolist()} surrogate gap {pi5['oracle_surrogate_gap']:.4f} exact gap {pi5['oracle_l1_gap']:.4f}")
    assert len(pi5['oracle_chosen']) == 5

    print(f"\nALL CHECKS PASSED. Surrogate-greedy latency: k={K} {greedy_time:.2f}s/query, k=5 {t5:.2f}s/query "
          f"-> per ~120-query episode {greedy_time*120/60:.1f} / {t5*120/60:.1f} min; per 100 episodes "
          f"{greedy_time*120*100/3600:.1f} / {t5*120*100/3600:.1f} h.")
    print(f"(legacy line) Pure-greedy k={K} query latency: {greedy_time:.2f}s "
          f"(~{greedy_time * 120 / 60:.1f} min per ~120-query episode; "
          f"~{greedy_time * 120 * 100 / 3600:.1f} h per 100-episode run).")


if __name__ == "__main__":
    main()
