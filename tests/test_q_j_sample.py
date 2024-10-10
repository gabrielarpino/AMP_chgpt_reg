import numpy as np
import pytest
from amp.marginal_separable_jax import q, q_j_sample


@pytest.mark.parametrize("n", [1, 5, 100])
@pytest.mark.parametrize("L", [1, 2, 3])
def test_noiseless(n, L):
    σ = 0 # so that output of q is simply the specified entry in each row of θ
    C_true = np.random.choice(L, size=n) # for test purpose lets skip sorting this
    θ = np.random.randn(n, L)
    expected_Y = θ[np.arange(n), C_true]
    Y = q(θ, C_true, σ).reshape(n,)
    assert expected_Y.shape == (n,)
    assert Y.shape == (n,)
    assert np.allclose(Y, expected_Y)
