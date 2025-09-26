import numpy as np
from typing import TypeVar, Literal
from numpy.typing import NDArray

FloatOrArray = TypeVar('FloatOrArray', float, NDArray[np.float64])
ParticleType = Literal['dm', 'baryon']
