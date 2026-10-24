import numpy as np
import pytest

from aifound import neural as n
from aifound.pii import SafeError


def test_softmax_sums_to_one_and_is_stable():
    p = n.softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isclose(p.sum(), 1) and np.all(np.isfinite(p))


def test_relu():
    assert list(n.relu(np.array([-1.0, 0.0, 2.0]))) == [0, 0, 2]


def test_neuron_fires_only_when_positive():
    assert n.neuron(np.array([1.0, 1.0]), np.array([1.0, 1.0]), -1) == 1.0
    assert n.neuron(np.array([1.0, 1.0]), np.array([-1.0, 0.0]), 0) == 0.0


@pytest.mark.parametrize("k", range(4))
def test_draw_each_class_has_ink(k):
    img = n.draw(k, np.random.default_rng(k))
    assert img.shape == (8, 8) and (img > 0.5).sum() >= 3


def test_backprop_matches_numeric_gradient():
    rng = np.random.default_rng(0)
    net = n.MLP([4, 5, 3], 0)
    x, y = rng.normal(size=(6, 4)), np.array([0, 1, 2, 0, 1, 2])

    def loss():
        p = n.softmax(net.forward(x)[0])
        return -np.mean(np.log(p[np.arange(6), y]))

    w0 = net.W[0].copy()
    eps = 1e-5
    net.W[0][1, 2] += eps
    up = loss()
    net.W[0][1, 2] -= 2 * eps
    down = loss()
    net.W[0] = w0.copy()
    numeric = (up - down) / (2 * eps)
    net.step(x, y, lr=1.0)
    analytic = w0[1, 2] - net.W[0][1, 2]
    assert abs(numeric - analytic) < 1e-6


def test_training_lowers_loss():
    x, y = n.make_shapes(400)
    net = n.MLP([64, 16, 4])
    h = net.fit(x, y, epochs=25)
    assert h[-1] < h[0] * 0.8


def test_hidden_layer_beats_linear():
    d = n.depth_study(epochs=15)
    acc = [r["test_accuracy"] for r in d["rows"]]
    assert acc[1] > acc[0] + 0.05


def test_example_probabilities_sum_to_one():
    d = n.depth_study(epochs=5)
    assert abs(sum(d["example"]["probabilities"].values()) - 1) < 0.01


@pytest.mark.parametrize("sizes", [[4], [4, 0]])
def test_mlp_rejects_bad_sizes(sizes):
    with pytest.raises(SafeError):
        n.MLP(sizes)


def test_fit_rejects_bad_lr():
    with pytest.raises(SafeError):
        n.MLP([2, 2]).fit(np.zeros((2, 2)), np.zeros(2, int), lr=0)
