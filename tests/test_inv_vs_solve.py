import pytest
import jax
import jax.numpy as jnp
import numpy as np

@pytest.mark.parametrize('L', [2, 3, 4, 10])
def test_basics(L):
    for _ in range(10):
        A = np.random.rand(L, L)
        B = np.random.rand(L, L)
        invA_res0 = jnp.linalg.pinv(A) @ B
        invA_res1 = jax.scipy.linalg.solve(A, B)
        assert jnp.allclose(invA_res0, invA_res1, rtol=1e-2)

        invB_res0 = A @ jnp.linalg.pinv(B)
        invB_res1 = jax.scipy.linalg.solve(B.T, A.T).T
        assert jnp.allclose(invB_res0, invB_res1, rtol=1e-2)

        # Test A symmetric:
        A_symm = (A + A.T)/2
        invA_symm_res0 = jnp.linalg.pinv(A_symm) @ B
        invA_symm_res1 = jax.scipy.linalg.solve(A_symm, B, assume_a='sym')
        invA_symm_res2 = jax.scipy.linalg.solve(A_symm, B)
        assert jnp.allclose(invA_symm_res0, invA_symm_res1, rtol=1e-2)
        assert jnp.allclose(invA_symm_res0, invA_symm_res2, rtol=1e-2)

        # Test B symmetric:
        B_symm = (B + B.T)/2
        invB_symm_res0 = A @ jnp.linalg.pinv(B_symm)
        invB_symm_res1 = jax.scipy.linalg.solve(B_symm.T, A.T, assume_a='sym').T
        invB_symm_res2 = jax.scipy.linalg.solve(B_symm.T, A.T).T
        assert jnp.allclose(invB_symm_res0, invB_symm_res1, rtol=1e-2)
        assert jnp.allclose(invB_symm_res0, invB_symm_res2, rtol=1e-2)

if __name__ == '__main__':
    test_basics(3)