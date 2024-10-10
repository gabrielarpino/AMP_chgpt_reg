import numpy as np
import pytest
from amp.performance_measures import hausdorff

def test_empty_set():
    """Empty estimated set should have Hausdorff distance inf from non-empty ground truth set."""
    est = []
    num_eles = 5
    gt = np.random.randn(num_eles)
    assert hausdorff(est, gt) == np.inf

def test_1_n():
    """Test that Hausdorff stays unchanged whether [1,n] are included in the set or not."""
    num_eles = 5
    est = np.random.randn(num_eles)
    gt = np.random.randn(num_eles)
    common_eles = np.random.randn(2)
    assert hausdorff(est, gt) == hausdorff(np.append(est, common_eles), np.append(gt, common_eles))