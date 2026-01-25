from typing import Literal, cast

import numpy as np

from src.types import QuantityLike

# def TDMA_solver(aa, ba, ca, da_in, nf):
#     """
#     Tridiagonal matrix algorithm.
#     """
#     dc = -da_in  # d needs to be on other side of hydrostatic equation for TDMA
#     for it in range(1, nf):
#         wa = aa[it - 1] / ba[it - 1]
#         ba[it] = ba[it] - wa * ca[it - 1]
#         dc[it] = dc[it] - wa * dc[it - 1]
#     xa = ba
#     xa[-1] = dc[-1] / ba[-1]
#     for il in range(nf - 2, -1, -1):
#         xa[il] = (dc[il] - ca[il] * xa[il + 1]) / ba[il]
#     return xa
