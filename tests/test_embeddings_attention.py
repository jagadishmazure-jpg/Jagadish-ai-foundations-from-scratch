import numpy as np
import pytest

from aifound import attention as a
from aifound import embeddings as e
from aifound.pii import SafeError


@pytest.fixture(scope="module")
def emb():
    return e.demo()


def test_same_group_closer_than_across(emb):
    assert emb["mean_cos_same_group"] > emb["mean_cos_fruit_vs_animal"] + 0.5


def test_king_minus_man_plus_woman_is_queen(emb):
    assert emb["analogies"][0]["got"] == "queen"


def test_all_analogies_at_16_dims(emb):
    assert emb["analogy_hits"] == "5/5"


def test_known_miss_at_8_dims(emb):
    # Kept visible on purpose: with only 8 dimensions one relation collapses.
    assert emb["analogy_hits_at_8_dims"] == "4/5"


def test_nearest_only_returns_nouns(emb):
    idx, vec = emb["_idx"], emb["_vec"]
    assert e.nearest(idx, vec, vec[idx["mango"]], {"mango"})[0] in e.FRUIT


def test_cos_bounds():
    assert np.isclose(e.cos(np.array([1.0, 0]), np.array([2.0, 0])), 1.0)


@pytest.mark.parametrize("kw", [{"dim": 0}, {"window": 0}])
def test_train_rejects_bad_args(kw):
    with pytest.raises(SafeError):
        e.train([["a", "b"]], **kw)


def test_attention_rows_sum_to_one():
    rng = np.random.default_rng(0)
    q, k, v = (rng.normal(size=(5, 4)) for _ in range(3))
    _, w = a.attention(q, k, v)
    assert np.allclose(w.sum(1), 1)


def test_causal_mask_hides_future():
    rng = np.random.default_rng(1)
    q, k, v = (rng.normal(size=(5, 4)) for _ in range(3))
    _, w = a.attention(q, k, v, causal=True)
    assert np.allclose(np.triu(w, 1), 0)


def test_first_token_attends_only_to_itself():
    rng = np.random.default_rng(2)
    q, k, v = (rng.normal(size=(3, 4)) for _ in range(3))
    out, _ = a.attention(q, k, v)
    assert np.allclose(out[0], v[0])


def test_non_causal_sees_everything():
    rng = np.random.default_rng(3)
    q, k, v = (rng.normal(size=(4, 4)) for _ in range(3))
    _, w = a.attention(q, k, v, causal=False)
    assert np.all(w > 0)


def test_pronoun_attends_to_cat():
    w = a.pronoun_demo()["weights_from_it"]
    assert max(w, key=w.get) == "cat" and w["cat"] > 0.5
