"""Write the measured numbers into README.md and docs/results.md from results/results.json.

    python scripts/render_docs.py          # rewrite the generated blocks
    python scripts/render_docs.py --check  # CI: fail if the docs drifted from results.json or the test count

Generated blocks sit between <!-- BEGIN:name --> and <!-- END:name --> markers.
results/results.json itself comes from `aifound demo` (a real CPU run).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import count_tests  # noqa: E402


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def blocks(r: dict, tests: int) -> dict[str, str]:
    rv, gd, fs, km, lc = r["rules_vs_learned"], r["gradient_descent"], r["fit_sweep"], r["kmeans"], r["learning_curve"]
    mlp, em, tl, df, mm, tp = r["mlp"], r["embeddings"], r["tiny_lm"], r["diffusion"], r["multimodal"], r["trip"]
    d2 = {q["layers"]: q for q in tl["depth"]}
    t = {q["temperature"]: q for q in tl["temperature"]}
    head = "\n".join(
        [
            "| What | Measured (synthetic data, CPU) |",
            "|---|---|",
            f"| Hand rules vs learned spam filter | {pct(rv['rules_accuracy'])} vs {pct(rv['learned_accuracy'])} on {rv['test_messages']} test messages |",
            f"| Gradient descent, house prices | learned ${gd['best_fit']['per_100m2']}k per 100 m2 and ${gd['best_fit']['per_room']}k per room "
            f"(true rule 120 / 15); step 0.12 diverges |",
            f"| Underfit vs overfit | held-out RMSE {fs['rows'][0]['val_rmse']} (degree 0), {fs['rows'][2]['val_rmse']} (degree 2), "
            f"{fs['rows'][-1]['val_rmse']} (degree 12, train RMSE {fs['rows'][-1]['train_rmse']}) |",
            f"| k-means, no labels | purity {km['purity_vs_hidden_kind']} on {km['items']} laundry items |",
            f"| More data | spam accuracy {pct(lc[0]['test_accuracy'])} with {lc[0]['train_examples']} examples -> "
            f"{pct(lc[-1]['test_accuracy'])} with {lc[-1]['train_examples']} |",
            f"| Neural net depth | {', '.join(f'{q["model"]}: {pct(q["test_accuracy"])}' for q in mlp['rows'])} |",
            f"| Embeddings | same-group cosine {em['mean_cos_same_group']} vs fruit-animal {em['mean_cos_fruit_vs_animal']}; analogies "
            f"{em['analogy_hits']} at {em['dims']} dims, {em['analogy_hits_at_8_dims']} at 8 |",
            f"| Tiny transformer | val loss {d2[1]['val_loss_before']} -> {d2[1]['val_loss_after']} (1 layer), {d2[2]['val_loss_after']} (2 layers) |",
            f"| Temperature | distinct outputs of 20: {', '.join(f'T={k}: {v["distinct_of_20"]}' for k, v in t.items())} |",
            f"| Pretraining -> SFT, held-out places | format {tl['qa_after_pretraining']['format_ok']} -> {tl['qa_after_sft']['format_ok']}, "
            f"correct {tl['qa_after_pretraining']['correct']} -> {tl['qa_after_sft']['correct']} |",
            f"| Diffusion | ring error {df['ring_error_by_step']['0']} (noise) -> {df['ring_error_by_step']['50']} after {df['denoise_steps']} steps "
            f"(real data {df['ring_error_real_data']}) |",
            f"| Multimodal | image->caption {pct(mm['image_to_caption_top1'])}, caption->image {pct(mm['caption_to_image_top1'])} (chance {pct(mm['chance'])}) |",
            f"| Trip planner, $500 | total ${tp['under_500']['total']}, {tp['under_500']['tool_calls']} tool calls, "
            f"{len(tp['under_500']['dropped_ungrounded'])} planted ungrounded claim dropped; $150 -> says no |",
            f"| Tests | {tests} |",
        ]
    )
    return {"headline": head, "results": "```json\n" + json.dumps(r, indent=2) + "\n```", "tests": f"**{tests} tests**"}


def apply(text: str, bl: dict[str, str]) -> str:
    for k, v in bl.items():
        text = re.sub(rf"(<!-- BEGIN:{k} -->\n).*?(<!-- END:{k} -->)", lambda m, v=v: m.group(1) + v + "\n" + m.group(2), text, flags=re.S)
    return text


def main() -> int:
    r = json.loads((ROOT / "results/results.json").read_text())
    bl = blocks(r, count_tests.count())
    bad = []
    for f in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        old = f.read_text()
        new = apply(old, bl)
        if new != old:
            if "--check" in sys.argv:
                bad.append(f.name)
            else:
                f.write_text(new)
    if bad:
        print("out of date:", ", ".join(bad), "- run: python scripts/render_docs.py")
        return 1
    print("docs match results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
