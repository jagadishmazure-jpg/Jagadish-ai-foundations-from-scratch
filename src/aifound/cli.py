"""Run every experiment, write results/results.json and draw the evidence board.

aifound demo            # full run (about 1 minute on a laptop CPU)
aifound demo --quick    # fewer training steps, for smoke tests
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from . import attention, classic, diffusion, embeddings, lm, multimodal, neural, trip

ROOT = Path(__file__).resolve().parents[2]


def run_all(quick: bool = False) -> tuple[dict, dict]:
    t0 = time.time()
    raw: dict = {}
    res: dict = {"python": platform.python_version(), "quick": quick}
    res["rules_vs_learned"] = classic.rules_vs_learned()
    res["gradient_descent"] = classic.gd_sweep()
    res["fit_sweep"] = classic.fit_sweep()
    res["kmeans"] = classic.kmeans_demo()
    res["learning_curve"] = classic.learning_curve()
    d = neural.depth_study(epochs=10 if quick else 60)
    raw["mlp"] = d.pop("_net")
    res["mlp"] = d
    e = embeddings.demo()
    raw["emb"] = (e.pop("_idx"), e.pop("_vec"))
    res["embeddings"] = e
    res["attention_toy"] = attention.pronoun_demo()
    m = lm.run(pre_steps=80 if quick else 600, sft_steps=40 if quick else 300)
    raw["lm"] = (m.pop("_model"), m.pop("_codec"))
    res["tiny_lm"] = m
    df = diffusion.run(steps=500 if quick else 8000)
    raw["frames"] = df.pop("_frames")
    res["diffusion"] = df
    res["multimodal"] = multimodal.run(steps=150 if quick else 1000)
    tp = trip.demo()
    raw["trace"] = tp.pop("_trace")
    res["trip"] = tp
    res["runtime_seconds"] = round(time.time() - t0, 1)
    return res, raw


def board(res: dict, raw: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 4, figsize=(20, 13))
    fig.suptitle("AI foundations from scratch: evidence board (real CPU runs, synthetic data)", fontsize=16, weight="bold")

    a = ax[0, 0]
    r = res["rules_vs_learned"]
    a.bar(["hand rules", "learned"], [r["rules_accuracy"], r["learned_accuracy"]], color=["#999", "#2a7"])
    a.set_ylim(0, 1)
    a.set_title(f"1. Rules vs learned spam filter ({r['test_messages']} test msgs)")
    for i, v in enumerate([r["rules_accuracy"], r["learned_accuracy"]]):
        a.text(i, v + 0.02, f"{v:.1%}", ha="center")

    a = ax[0, 1]
    x, y = classic.make_houses(400)
    for lr in (0.001, 0.01, 0.05, 0.12):
        _, h = classic.gradient_descent(x, y, lr, 200)
        a.plot(h, label=f"lr={lr}")
    a.set_yscale("log")
    a.set_ylim(10, 1e7)
    a.set_title("2. Gradient descent: loss per step (lr 0.12 blows up)")
    a.set_xlabel("step")
    a.legend()

    a = ax[0, 2]
    rows = res["fit_sweep"]["rows"]
    a.plot([q["degree"] for q in rows], [q["train_rmse"] for q in rows], "o-", label="train")
    a.plot([q["degree"] for q in rows], [q["val_rmse"] for q in rows], "o-", label="held-out")
    a.set_yscale("log")
    a.set_title("3. Underfit -> fit -> overfit (polynomial degree)")
    a.set_xlabel("degree")
    a.legend()

    a = ax[0, 3]
    lc = res["learning_curve"]
    a.semilogx([q["train_examples"] for q in lc], [q["test_accuracy"] for q in lc], "o-")
    a.set_title("4. More labelled data -> better model")
    a.set_xlabel("training examples")
    a.set_ylabel("test accuracy")

    a = ax[1, 0]
    w = raw["mlp"].W[0]
    tiles = np.zeros((4 * 9, 8 * 9))
    for i in range(32):
        r0, c0 = divmod(i, 8)
        tiles[r0 * 9 : r0 * 9 + 8, c0 * 9 : c0 * 9 + 8] = w[:, i].reshape(8, 8)
    a.imshow(tiles, cmap="RdBu")
    a.axis("off")
    a.set_title("5. Raw first-layer weights, 32 neurons (noisy, see docs)")

    a = ax[1, 1]
    rows = res["mlp"]["rows"]
    a.bar([q["model"].split(" (")[0] for q in rows], [q["test_accuracy"] for q in rows], color="#47a")
    a.set_ylim(0.5, 1.02)
    a.set_title("6. Depth helps: 8x8 shape test accuracy")
    for i, q in enumerate(rows):
        a.text(i, q["test_accuracy"] + 0.01, f"{q['test_accuracy']:.1%}", ha="center")

    a = ax[1, 2]
    idx, vec = raw["emb"]
    words = embeddings.FRUIT + embeddings.ANIMALS + ["man", "woman", "king", "queen"]
    sub = vec[[idx[wd] for wd in words]]
    sub = sub - sub.mean(0)
    p2 = sub @ np.linalg.svd(sub, full_matrices=False)[2][:2].T
    for wd, (px, py) in zip(words, p2, strict=True):
        a.scatter(px, py, color="#c52" if wd in embeddings.FRUIT else "#25c" if wd in embeddings.ANIMALS else "#292")
        pass
    for grp, col in (("fruit", "#c52"), ("animals", "#25c"), ("people", "#292")):
        a.scatter([], [], color=col, label=grp)
    a.legend()
    a.set_title(f"7. Embeddings: fruit, animals, people form 3 tight clusters; analogies {res['embeddings']['analogy_hits']}")

    a = ax[1, 3]
    t = res["tiny_lm"]["temperature"]
    a.bar([str(q["temperature"]) for q in t], [q["distinct_of_20"] for q in t], color="#a5c")
    a.set_title("8. Temperature: distinct outputs out of 20 samples")
    a.set_xlabel("temperature")

    a = ax[2, 0]
    for i, fr in zip(res["diffusion"]["ring_error_by_step"], raw["frames"], strict=True):
        if i in ("0", "40", "50"):
            a.scatter(fr[:400, 0], fr[:400, 1], s=3, label=f"denoise step {i}")
    a.set_aspect("equal")
    a.set_xlim(-3, 3)
    a.set_ylim(-3, 3)
    a.legend(markerscale=4)
    a.set_title("9. Diffusion: noise -> ring in 50 denoising steps")

    a = ax[2, 1]
    tl = res["tiny_lm"]
    labels = ["format OK", "correct"]
    pre = [int(tl["qa_after_pretraining"][k].split("/")[0]) for k in ("format_ok", "correct")]
    sft = [int(tl["qa_after_sft"][k].split("/")[0]) for k in ("format_ok", "correct")]
    xs = np.arange(2)
    a.bar(xs - 0.2, pre, 0.4, label="after pretraining")
    a.bar(xs + 0.2, sft, 0.4, label="after SFT")
    a.set_xticks(xs, labels)
    a.set_ylim(0, 6.5)
    a.set_title("10. Pretraining then SFT: 6 held-out questions")
    a.legend()

    a = ax[2, 2]
    mm = res["multimodal"]
    vals = [mm["image_to_caption_top1"], mm["caption_to_image_top1"], mm["chance"]]
    a.bar(["image->caption", "caption->image", "chance"], vals, color=["#2a7", "#2a7", "#999"])
    a.set_ylim(0, 1.05)
    a.set_title("11. Multimodal: shared image/text space")

    a = ax[2, 3]
    a.axis("off")
    tp = res["trip"]["under_500"]
    lines = ["12. 'Plan my Goa trip under $500' (end to end)", ""] + [f"{s['agent']:>10}: {s['action'][:40]}" for s in raw["trace"]]
    lines += [
        "",
        f"total ${tp['total']}  tool calls {tp['tool_calls']}",
        f"dropped ungrounded claim: {len(tp['dropped_ungrounded'])}",
        f"under $150 feasible: {res['trip']['under_150']['feasible']}",
    ]
    a.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aifound")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--quick", action="store_true")
    d.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    res, raw = run_all(a.quick)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    board(res, raw, out / "evidence_board.png")
    print(json.dumps({k: res[k] for k in ("rules_vs_learned", "mlp", "runtime_seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
