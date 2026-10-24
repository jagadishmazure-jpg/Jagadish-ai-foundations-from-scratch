import numpy as np
import pytest

from aifound import classic as c
from aifound.pii import SafeError


def test_learned_beats_rules():
    r = c.rules_vs_learned()
    assert r["learned_accuracy"] > r["rules_accuracy"] + 0.1


def test_learned_weights_pick_spam_words():
    assert set(c.rules_vs_learned()["top_spam_weights"]) <= set(c.SPAM_WORDS)


def test_rule_classifier_triggers():
    assert c.rule_classifier(["free", "lunch"]) == 1
    assert c.rule_classifier(["cash", "bonus"]) == 0  # the miss that rules make


def test_featurize_counts_words():
    x = c.featurize([["free", "free", "team"]])
    assert x[0, c.VOCAB.index("free")] == 2 and x.sum() == 3


def test_sigmoid_is_stable():
    assert np.all(np.isfinite(c.sigmoid(np.array([-1e6, 0, 1e6]))))


@pytest.mark.parametrize("kw", [{"lr": 0}, {"lr": -1}, {"epochs": 0}])
def test_logistic_rejects_bad_args(kw):
    with pytest.raises(SafeError):
        c.train_logistic(np.zeros((2, 2)), np.zeros(2), **kw)


def test_gd_recovers_true_rule():
    b = c.gd_sweep()["best_fit"]
    assert abs(b["per_100m2"] - 120) < 5 and abs(b["per_room"] - 15) < 3


def test_gd_large_step_diverges_small_step_is_slow():
    s = c.gd_sweep()
    assert s["0.12"]["diverged"] and not s["0.05"]["diverged"]
    assert s["0.001"]["final_mse"] > s["0.05"]["final_mse"] * 5


def test_gd_loss_never_increases_for_safe_step():
    x, y = c.make_houses(200)
    _, h = c.gradient_descent(x, y, 0.01, 300)
    assert all(b <= a + 1e-9 for a, b in zip(h, h[1:], strict=False))


@pytest.mark.parametrize("lr", [0, -0.1])
def test_gd_rejects_bad_lr(lr):
    with pytest.raises(SafeError):
        c.gradient_descent(np.zeros((2, 1)), np.zeros(2), lr)


def test_fit_sweep_shape():
    s = c.fit_sweep()
    rows = {r["degree"]: r for r in s["rows"]}
    assert rows[0]["val_rmse"] > rows[2]["val_rmse"]  # underfit
    assert rows[12]["val_rmse"] > rows[2]["val_rmse"] * 3  # overfit
    assert rows[12]["train_rmse"] < rows[2]["train_rmse"]  # but it memorised the training points
    assert s["best_degree"] in (2, 4)


def test_kmeans_finds_groups_without_labels():
    r = c.kmeans_demo()
    assert r["labels_used_for_training"] == 0 and r["purity_vs_hidden_kind"] > 0.95


@pytest.mark.parametrize("k", [0, 1000])
def test_kmeans_rejects_bad_k(k):
    with pytest.raises(SafeError):
        c.kmeans(np.zeros((5, 2)) + np.arange(5)[:, None], k)


def test_purity_perfect():
    assert c.purity(np.array([1, 1, 0]), np.array([0, 0, 1])) == 1.0


def test_learning_curve_rises():
    lc = c.learning_curve()
    assert lc[-1]["test_accuracy"] > lc[0]["test_accuracy"] + 0.2


def test_data_is_deterministic():
    assert c.make_messages(5, 3)[0] == c.make_messages(5, 3)[0]
