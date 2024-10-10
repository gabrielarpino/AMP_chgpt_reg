import pytest
import numpy as np
import sys
sys.path.append('/Users/xiaoqiliu/Desktop/3_Changepoints_detection/AMP/')
from amp.fully_separable import define_f, psd_mat

if __name__ == '__main__':
    δ = 0.1
    L = 3
    ρ = psd_mat(L)
    ν̂_B = np.random.rand(L,L)
    # κ_B is the covariance matrix of G_B so PSD and symmetric:
    κ_B = psd_mat(L)

    define_f(δ, L, ν̂_B, κ_B, ρ)


@pytest.mark.parametrize('δ', [0.1, 0.5, 2.1])
@pytest.mark.parametrize('L', [2, 3, 4, 10])
def test_f_functionality(δ, L):
    """Ensure define_f and f run without errors."""
    ρ = psd_mat(L)
    ν̂_B = np.random.rand(L,L)
    # κ_B is the covariance matrix of G_B so PSD and symmetric:
    κ_B = psd_mat(L)
    f = define_f(δ, L, ν̂_B, κ_B, ρ)
    s = np.random.rand(1, L)
    res = f(s)