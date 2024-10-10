import pytest
import jax.numpy as jnp
import numpy as np
import sys
sys.path.append('/Users/xiaoqiliu/Desktop/3_Changepoints_detection/AMP/')
from amp.fully_separable import define_g, psd_mat
import jax
from jax import config, jacfwd
from jax import random
config.update("jax_enable_x64", True)
from jax.test_util import check_grads
from amp.separable_jax import g_j


@pytest.mark.parametrize("L", [2, 3, 4])
def test_g_j_AD(L):
    # Check if autodiff matches finite difference for separable g_j
    key = random.PRNGKey(0)
    V = random.normal(key, (1, L))
    u = random.normal(key, (1, ))
    ϕ = random.uniform(key, (L, ))
    ϕ = ϕ / jnp.sum(ϕ)
    σ = 1e-3
    ρ = jnp.eye(L) * 1.0
    ν = random.normal(key, (L, L))
    ν = ν @ ν.T # ensure ν is PSD
    κ_T = random.normal(key, (L, L))
    κ_T = κ_T @ κ_T.T # ensure κ_T is PSD
    EPS = 1e-5
    for i in range(L):
        # Differentiate each output entry of g_j wrt the L entries in V
        # 1) Use jax.test_util.check_grads to check if AD matches finite difference:
        print(f'== entry_idx {i} ==')
        def g_j_ith_output_entry(V):
            return g_j(V, u, ϕ, σ, ρ, ν, κ_T)[1][i]
        check_grads(f=g_j_ith_output_entry, args=(V), \
            order=1, modes='fwd', eps=1e-3, atol=1e-2)
        
        # 2) Manually compare AD and finite difference:
        dgdV_AD = jacfwd(g_j_ith_output_entry, argnums=0)(V.squeeze()) # shape must be (L, )
        print(f'AD: {dgdV_AD}')

        dgdV_finite_diff = np.zeros((L, L))
        for j in range(L):
            V_perturbed = V.copy()
            V_perturbed = V_perturbed.at[0, j].set(V[0, j] + EPS)
            # Fill in jth column which correspond to derivative wrt jth entry in input V:
            dgdV_finite_diff[:, j] = (g_j_ith_output_entry(V_perturbed.squeeze()) - \
                g_j_ith_output_entry(V.squeeze())) / EPS
        print(f'finite diff: {dgdV_finite_diff}')
        assert np.allclose(dgdV_AD, dgdV_finite_diff, rtol=1e-2)


if __name__ == '__main__':
    test_g_j_AD(4)