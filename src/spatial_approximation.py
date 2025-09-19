import numpy as np
from . import utils
from numba import njit,prange
from .types import FloatOrArray
from numpy.typing import NDArray
from .density.density import Density
from typing import Any,cast,Self

class Lattice:
    def __init__(self,n_posts:int|float,start:float,end:float,log:bool=True) -> None:
        self.start = float(start)
        self.end = float(end)
        if log:
            self.start_lattice:float = np.log10(self.start)
            self.end_lattice:float = np.log10(self.end)
        else:
            self.start_lattice = self.start
            self.end_lattice = self.end
        self.log = log
        self.n_posts = int(n_posts)
        self.lattice_spacing:float = np.abs(self.end_lattice-self.start_lattice)/self.n_posts

    @classmethod
    def from_density(cls,density:Density,start:float=1e-4,overide_start:bool=True,n_posts:int|float=int(1e4),**kwargs:Any) -> Self:
        return cls(start=density.Rmin.value if overide_start else start,end=density.Rmax.value,n_posts=n_posts,**kwargs)

    def __len__(self):
        return self.n_posts

    def __call__(self,x:FloatOrArray) -> FloatOrArray:
        return self.to_lattice_coordinates(x)

    @property
    def posts(self):
        if self.log:
            return np.geomspace(self.start,self.end,self.n_posts)
        return np.linspace(self.start,self.end,self.n_posts)

    def update(self,r:NDArray[np.float64],n_posts:int|float|None=None):
        self.end = np.max([r.max(),self.end])
        if self.log:
            self.start_lattice = np.log10(self.start)
            self.end_lattice = np.log10(self.end)
        else:
            self.start_lattice = self.start
            self.end_lattice = self.end
        if n_posts:
            self.n_posts = int(n_posts)
        self.lattice_spacing = np.abs(self.end_lattice-self.start_lattice)/self.n_posts

    @staticmethod
    @njit(parallel=True)
    def fast_augment_to_lattice(x:NDArray[np.float64],start_lattice:float,log:bool,clip:bool=False,min_lattice:float=0,max_lattice:int=100000) -> NDArray[np.float64]:
        output = np.empty_like(x)
        for i in prange(len(x)):
            if log:
                new_x = np.log10(x[i]) - start_lattice
            else:
                new_x = x[i] - start_lattice
            if clip:
                if new_x < min_lattice:
                    new_x = min_lattice
                elif new_x > max_lattice:
                    new_x = max_lattice
            output[i] = new_x
        return output

    @staticmethod
    @njit(parallel=True)
    def fast_augment_from_lattice(x:NDArray[np.float64],start_lattice:float,log:bool) -> NDArray[np.float64]:
        output = np.empty_like(x)
        for i in prange(len(x)):
            if log:
                output[i] = 10**(x[i] + start_lattice)
            else:
                output[i] = x[i] + start_lattice
        return output

    def augment_to_lattice(self,x:FloatOrArray,clip:bool=True) -> FloatOrArray:
        output = self.fast_augment_to_lattice(np.atleast_1d(x),self.start_lattice,self.log,clip,0,len(self))
        if np.isscalar(x):
            output = output[0]
        return cast(FloatOrArray,output)

    def augment_from_lattice(self,x:FloatOrArray) -> FloatOrArray:
        output = self.fast_augment_from_lattice(np.atleast_1d(x),self.start_lattice,self.log)
        if np.isscalar(x):
            output = output[0]
        return cast(FloatOrArray,output)

    def in_lattice(self,x:FloatOrArray) -> FloatOrArray:
        return cast(FloatOrArray,(x >= self.start) * (x <= self.end))

    def to_lattice_coordinates(self,x:FloatOrArray,clip:bool=True) -> FloatOrArray:
        x_lattice = self.augment_to_lattice(x,clip)/self.lattice_spacing
        if np.isscalar(x):
            if x_lattice == 0:
                return cast(FloatOrArray,0)
            else:
                return cast(FloatOrArray,int(x_lattice))
        assert isinstance(x_lattice, np.ndarray)
        x_lattice[x == 0] = 0
        x_lattice = (x_lattice).astype(int)
        return cast(FloatOrArray,x_lattice)

    def to_space_coordinates(self,x:FloatOrArray) -> FloatOrArray:
        return self.augment_from_lattice(x*self.lattice_spacing)

    def lattice_to_density(self,x:NDArray[np.float64]) -> NDArray[np.int64]:
        return np.bincount(x.clip(min=0),minlength=len(self))

    def lattice_to_density_cumsum(self,x:NDArray[np.float64]) -> NDArray[np.int64]:
        density = self.lattice_to_density(x)
        return np.cumsum(density)

    def assign_from_density(self,x:FloatOrArray) -> FloatOrArray:
        x_lattice = self(x)
        density_cumsum = self.lattice_to_density_cumsum(np.atleast_1d(x_lattice)).astype(np.float64)
        assigned = utils.fast_assign(np.atleast_1d(x_lattice),density_cumsum)
        if np.isscalar(x):
            return assigned[0]
        return cast(FloatOrArray,assigned)
