"""Top-level package for AMP."""

__version__ = '0.1.0'

# noinspection PyUnresolvedReferences

# from plum import PromisedType, Dispatcher

# PromisedFDD = PromisedType()
# PromisedGP = PromisedType()
# PromisedMeasure = PromisedType()

# _dispatch = Dispatcher()

# from .rv import *
# from .operators import *

ϵ_0 = 1e-6 # Numerical error tolerance (for float64, but for float32 I think we have to set ϵ_0 = 1e-7, otherwise it makes no difference. )
# PAL = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3'] # Color Palette
PAL = ['#1b9e77','#d95f02','#7570b3','#e7298a']
# PAL = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

from .changepoint_jax import *
from .covariances import *
# from .posterior import *
from .performance_measures import *
# from .comparison import *
from .signal_configuration import *

__all__ = ["PAL"]