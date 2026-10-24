import numpy as np
import pytest
import torch

from aifound import attention, diffusion, lm, multimodal
from aifound.pii import SafeError


def test_torch_head_matches_numpy_reference():
    torch.manual_seed(0)
    h = lm.Head(8, 16)
    x = torch.randn(1, 6, 8)
    out, w = h(x, return_weights=True)
    q, k, v = (m(x)[0].detach().numpy() for m in (h.q, h.k, h.v))
    ref_out, ref_w = attention.attention(q, k, v)
    assert np.allclose(w[0].detach().numpy(), ref_w, atol=1e-5)
    assert np.allclose(out[0].detach().numpy(), ref_out, atol=1e-5)


def test_codec_roundtrip_and_unknown():
    c = lm.CharCodec("abc ")
    assert c.decode(c.encode("cab a")) == "cab a"
    with pytest.raises(SafeError):
        c.encode("z")


def test_more_layers_more_params():
    a, b = lm.TinyLM(10, layers=1), lm.TinyLM(10, layers=3)
    assert sum(p.numel() for p in b.parameters()) > sum(p.numel() for p in a.parameters())


def test_tinylm_rejects_zero_layers():
    with pytest.raises(SafeError):
        lm.TinyLM(10, layers=0)


def test_temperature_zero_is_one_hot_and_negative_rejected():
    m = lm.TinyLM(5, d=8, layers=1, block=8)
    p = lm.next_token_probs(m, [1, 2], 0.0)
    assert float(p.max()) == 1.0 and float(p.sum()) == 1.0
    with pytest.raises(SafeError):
        lm.next_token_probs(m, [1], -0.5)


def test_higher_temperature_flattens():
    torch.manual_seed(0)
    m = lm.TinyLM(6, d=8, layers=1, block=8)
    with torch.no_grad():
        m.out.weight.mul_(20)
    e = [lm.entropy_bits(lm.next_token_probs(m, [1, 2], t)) for t in (0.3, 1.0, 3.0)]
    assert e[0] < e[1] < e[2]


def test_corpora_are_generated_and_heldout_places_absent_from_sft():
    sft = lm.sft_text()
    assert all(p not in sft for p in lm.HELDOUT_PLACES)
    assert all(p in lm.pretrain_text() for p in lm.HELDOUT_PLACES)


@pytest.fixture(scope="module")
def quick_lm():
    return lm.run(pre_steps=150, sft_steps=80)


def test_pretraining_lowers_val_loss(quick_lm):
    for row in quick_lm["depth"]:
        assert row["val_loss_after"] < row["val_loss_before"] * 0.7


def test_greedy_is_repeatable_and_hot_is_varied(quick_lm):
    t = {r["temperature"]: r for r in quick_lm["temperature"]}
    assert t[0.0]["distinct_of_20"] == 1 and t[1.5]["distinct_of_20"] > 5


def test_generate_stops_at_stop_char(quick_lm):
    m, c = quick_lm["_model"], quick_lm["_codec"]
    out = lm.generate(m, c, "Q: weather in pune? A:", 40, stop="\n")
    assert len(out) <= 40


def test_train_rejects_bad_steps():
    with pytest.raises(SafeError):
        lm.train(lm.TinyLM(5, d=8, layers=1, block=8), torch.zeros(50, dtype=torch.long), 0)


def test_diffusion_schedule_shrinks_signal():
    _, _, abar = diffusion.schedule()
    assert 0.99 < float(abar[0]) <= 1 and float(abar[-1]) < 0.05


def test_ring_error_zero_on_ring():
    assert diffusion.ring_error(torch.tensor([[1.0, 0.0], [0.0, -1.0]])) < 1e-6


def test_diffusion_denoising_moves_toward_ring():
    r = diffusion.run(steps=800)
    e = r["ring_error_by_step"]
    assert e["50"] < e["0"] * 0.6


def test_diffusion_rejects_zero_steps():
    with pytest.raises(SafeError):
        diffusion.train(0)


def test_multimodal_pairs_shape():
    x, c, lab = multimodal.make_pairs(10)
    assert x.shape == (10, 64) and c.shape == (10, 7) and lab.max() < 8


def test_multimodal_beats_chance():
    r = multimodal.run(steps=200)
    assert r["image_to_caption_top1"] > 3 * r["chance"]


def test_multimodal_rejects_zero_steps():
    with pytest.raises(SafeError):
        multimodal.train(0)
