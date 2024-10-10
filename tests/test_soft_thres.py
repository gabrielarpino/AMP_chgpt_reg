from amp.marginal_separable_jax import soft_thres
import jax.numpy as jnp
import pytest
import numpy as np

@pytest.mark.parametrize("θ", [0, 0.1, 5])
@pytest.mark.parametrize("n", [1, 10, 100])
def test(θ, n):
    y = np.random.randn(n)
    ŷ = soft_thres(y, θ)

    ŷ_expected = y.copy()
    ŷ_expected[np.abs(y) <= θ] = 0
    ŷ_expected[y > θ] -= θ
    ŷ_expected[y < -θ] += θ
    assert np.allclose(ŷ, ŷ_expected)