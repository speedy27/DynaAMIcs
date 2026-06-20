"""
Generic eval driver for the PURE-JEPA (metric_coeff=0) variant sweeps:
  EXP2 (m3-idm-selfsup): does a stronger IDM weight induce a plannable latent METRIC without the
        isometry loss (fully self-supervised)?
  EXP3 (m3-bottleneck):  does shrinking the latent dim toward the true-state dim help PURE planning?

Per checkpoint it measures (all in-process; no file clobbers — eval scripts' side-files go to /tmp):
  * planning (plan_glv_learned): mppi_latent / mppi_decoded / mppi_learned success+final, 3 seeds x 12 ep
  * latent METRIC + rollout (m3_metric_gate.gate_one): latent-vs-true-dist Spearman to target (the
    Euclidean state metric the tol is defined on), free-running 6-step rollout error, feat_std
  * recognition (m3_recognition): dominant_guild + basin linear-probe accuracy
Reference bars (read from committed JSONs): pure-JEPA idm=1.0 (the lowreg model that FAILS) and the
HYBRID mc=0.3 (the upper bar that closes the loop).

INTEGRITY: pure JEPA = metric_coeff=0; the IDM uses only (z_t,z_{t+1})->action from the model's own
latents (no true-state distance). IDM induces a CONTROL metric, not the Euclidean state metric tol uses,
so partial planning improvement without full tol-crossing is the expected, meaningful outcome.

Run:  .venv-cpu/bin/python -m examples.microbiome_jepa.eval_pure_variants --which idm
      .venv-cpu/bin/python -m examples.microbiome_jepa.eval_pure_variants --which dim
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from examples.microbiome_jepa import plan_glv_learned, m3_recognition
from examples.microbiome_jepa.m3_metric_gate import gate_one

WEAKREG = {"data.n_candidate": 24, "model.regularizer.sim_coeff_t": 4,
           "model.regularizer.cov_coeff": 1, "model.regularizer.std_coeff": 0.25}
FNAME = "examples/microbiome_jepa/cfgs/layerB_worldmodel.yaml"
LOWREG = "checkpoints/plan_model_k24_lowreg/latest.pth.tar"  # pure-JEPA idm=1.0 ref (d128)
R = Path("examples/microbiome_jepa/results")


def _ref_rows():
    """Reference bars from committed result JSONs (MEASURED): pure-JEPA idm=1.0 + hybrid mc=0.3."""
    def L(n):
        return json.load(open(R / n))
    rec = L("m3_recognition.json")["tasks"]
    pure_plan = L("planning_learned_lowreg.json")["summary"]["mppi_latent"]
    gate = L("m3_metric_gate_mc03.json")
    hyb_plan = L("planning_learned_metric_mc03.json")["summary"]["mppi_latent"]
    return [
        {"tag": "pure-JEPA idm=1.0 (ref, fails)", "d_model": 128,
         "mppi_latent_success": pure_plan["success_rate_mean"], "mppi_latent_final": pure_plan["mean_final_dist"],
         "freerun_6step_err": gate["pure_jepa_ref"]["rollout"]["freerun_6step_err"],
         "latent_spearman": gate["pure_jepa_ref"]["corr_to_target"]["spearman"],
         "recog_guild": rec["dominant_guild"]["pure_jepa_ref"]["acc_full"],
         "recog_basin": rec["basin"]["pure_jepa_ref"]["acc_full"]},
        {"tag": "HYBRID mc=0.3 (upper bar)", "d_model": 128,
         "mppi_latent_success": hyb_plan["success_rate_mean"], "mppi_latent_final": hyb_plan["mean_final_dist"],
         "freerun_6step_err": gate["metric_hybrid"]["rollout"]["freerun_6step_err"],
         "latent_spearman": gate["metric_hybrid"]["corr_to_target"]["spearman"],
         "recog_guild": rec["dominant_guild"]["metric_hybrid"]["acc_full"],
         "recog_basin": rec["basin"]["metric_hybrid"]["acc_full"]},
    ]


def eval_one(tag, checkpoint, d_model, device="cpu"):
    dev = torch.device(device)
    ov = dict(WEAKREG); ov["model.d_model"] = d_model
    # planning (writes side json to /tmp with unique tag)
    plan = plan_glv_learned.run(fname=FNAME, checkpoint=checkpoint, device=device, seeds="0,1,2",
                                n_episodes=12, mpc_steps=20, horizon=6, n_samples=128, n_iters=3,
                                out="/tmp/eval_pure", tag=tag, overrides=ov)
    s = plan["summary"]
    # latent metric + rollout
    g = gate_one(tag, FNAME, checkpoint, dev, ov, k=6)
    # recognition (capture return; side json -> /tmp)
    rec = m3_recognition.run(fname=FNAME, checkpoint=checkpoint, ref_checkpoint=None, device=device,
                             out="/tmp/eval_pure", overrides=ov)
    rt = rec["tasks"]
    row = {"tag": tag, "d_model": d_model, "checkpoint": checkpoint,
           "mppi_latent_success": s["mppi_latent"]["success_rate_mean"],
           "mppi_latent_final": s["mppi_latent"]["mean_final_dist"],
           "mppi_decoded_success": s["mppi_decoded"]["success_rate_mean"],
           "mppi_learned_success": s["mppi_learned"]["success_rate_mean"],
           "freerun_6step_err": g["rollout"]["freerun_6step_err"],
           "one_step_err": g["rollout"]["one_step_err"],
           "latent_spearman": g["corr_to_target"]["spearman"],
           "decode_r2_mlp": g["decode_r2"]["mlp"], "feat_std": g["feat_std"],
           "recog_guild": rt["dominant_guild"]["metric_hybrid"]["acc_full"],
           "recog_basin": rt["basin"]["metric_hybrid"]["acc_full"],
           "head_spearman": plan["head"]["head_spearman_heldout"]}
    print(f"  [{tag}] mppi_latent {row['mppi_latent_success']*100:.0f}%/{row['mppi_latent_final']:.2f} "
          f"latent-Spearman {row['latent_spearman']:+.2f} rollout6 {row['freerun_6step_err']:.3f} "
          f"recog g/b {row['recog_guild']:.3f}/{row['recog_basin']:.3f}")
    return row


IDM_MODELS = [("idm=2", "checkpoints/plan_model_idm_2/latest.pth.tar", 128),
              ("idm=5", "checkpoints/plan_model_idm_5/latest.pth.tar", 128),
              ("idm=10", "checkpoints/plan_model_idm_10/latest.pth.tar", 128)]
DIM_MODELS = [("d=16", "checkpoints/plan_model_dim_16/latest.pth.tar", 16),
              ("d=24", "checkpoints/plan_model_dim_24/latest.pth.tar", 24),
              ("d=32", "checkpoints/plan_model_dim_32/latest.pth.tar", 32)]


def run(which: str = "idm", device: str = "cpu"):
    models = IDM_MODELS if which == "idm" else DIM_MODELS
    title = ("EXP2 IDM-reweight self-supervised closure" if which == "idm"
             else "EXP3 bottleneck shrink (latent dim toward true-state dim=24)")
    rows = [eval_one(t, c, d, device) for (t, c, d) in models]
    out = {"experiment": title, "metric_coeff": 0, "note":
           ("IDM uses only (z_t,z_{t+1})->action — self-supervised, no true-state distance. IDM induces "
            "a CONTROL metric, not the Euclidean state metric tol is defined on." if which == "idm"
            else "pure JEPA; latent dim swept toward the true gLV state dim S=24."),
           "reference_bars": _ref_rows(), "variants": rows}
    fn = R / f"exp{'2' if which=='idm' else '3'}_{which}.json"
    fn.write_text(json.dumps(out, indent=2, default=float))

    md = [f"# {title} (MEASURED)\n", f"metric_coeff=0 (pure JEPA). {out['note']}\n",
          "| variant | d | mppi_latent succ/final | latent↔true Spearman | free-run 6-step | recog guild | recog basin |",
          "|---|---|---|---|---|---|---|"]
    for r in out["reference_bars"] + rows:
        md.append(f"| {r['tag']} | {r['d_model']} | "
                  f"{r['mppi_latent_success']*100:.0f}% / {r['mppi_latent_final']:.2f} | "
                  f"{r['latent_spearman']:+.3f} | {r['freerun_6step_err']:.3f} | "
                  f"{r['recog_guild']:.3f} | {r['recog_basin']:.3f} |")
    (R / f"exp{'2' if which=='idm' else '3'}_{which}.md").write_text("\n".join(md))
    print("\n".join(md[2:]))
    print(f"\nsaved -> {fn}")
    return out


if __name__ == "__main__":
    import fire
    fire.Fire(run)
