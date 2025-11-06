from typing import Literal, TypeVar

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from astropy.units import Quantity

FloatOrArray = TypeVar('FloatOrArray', float, NDArray[np.float64])
QuantityOrArray = Quantity | NDArray[np.float64] | pd.Series
ParticleType = Literal['dm', 'baryon']
