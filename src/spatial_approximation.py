import numpy as np
from .constants import kpc
from numba import njit,prange

epsilon = 1e-4 * kpc

class Lattice:
    def __init__(self,n_posts,start,end,log=True):
        self.start = start
        self.end = end
        if log:
            self.start_lattice = np.log10(self.start)
            self.end_lattice = np.log10(self.end)
        else:
            self.start_lattice = self.start
            self.end_lattice = self.end
        self.log = log
        self.n_posts = int(n_posts)
        self.lattice_spacing = np.abs(self.end_lattice-self.start_lattice)/self.n_posts

    @classmethod
    def from_density(cls,density,start=epsilon,overide_start=True,n_posts=int(1e4),**kwargs):
        if not overide_start:
            start = density.Rmin
        return cls(start=start,end=density.Rmax,n_posts=n_posts,**kwargs)

    def __len__(self):
        return self.n_posts

    def __call__(self,x):
        return self.to_lattice_coordinates(x)

    @property
    def posts(self):
        if self.log:
            return np.geomspace(self.start,self.end,self.n_posts)
        return np.linspace(self.start,self.end,self.n_posts)

    def update(self,r,n_posts=None):
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
        self._lattice = None

    @staticmethod
    @njit(parallel=True)
    def fast_augment_to_lattice(x,start_lattice,log,clip=False,min_lattice=0,max_lattice=100000):
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
    def fast_augment_from_lattice(x,start_lattice,log):
        output = np.empty_like(x)
        for i in prange(len(x)):
            if log:
                output[i] = 10**(x[i] + start_lattice)
            else:
                output[i] = x[i] + start_lattice
        return output

    def augment_to_lattice(self,x,clip=True):
        return self.fast_augment_to_lattice(x,self.start_lattice,self.log,clip,0,len(self))

    def augment_from_lattice(self,x):
        return self.fast_augment_from_lattice(x,self.start_lattice,self.log)

    def in_lattice(self,x):
        return (x >= self.start) * (x <= self.end)

    def to_lattice_coordinates(self,x,clip=True):
        scalar_input = np.isscalar(x)
        if scalar_input:
            x = np.array([x])
        x_lattice = self.augment_to_lattice(x,clip)/self.lattice_spacing
        x_lattice[x == 0] = 0
        x_lattice = (x_lattice).astype(int)
        if scalar_input:
            x_lattice = x_lattice[0]
        return x_lattice

    def to_space_coordinates(self,x):
        return self.augment_from_lattice(x*self.lattice_spacing)

    def lattice_to_density(self,x):
        return np.bincount(x.clip(min=0),minlength=len(self))

    def lattice_to_density_cumsum(self,x):
        density = self.lattice_to_density(x)
        return np.cumsum(density)

    def assign_from_density(self,x):
        x_lattice = self(x)
        density_cumsum = self.lattice_to_density_cumsum(x_lattice)
        return density_cumsum[x_lattice]
