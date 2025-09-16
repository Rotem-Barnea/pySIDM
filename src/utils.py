import numpy as np
import pandas as pd
from numba import njit,prange
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from .constants import kpc,km,second,default_units,Unit

def random_angle(like,acos):
    rolls = np.random.rand(len(like)) if len(like.shape) == 1 else np.random.rand(*like.shape)
    if acos:
        return np.acos(rolls*2-1)
    return rolls*2*np.pi

def from_radial(r,theta,quick_sin=True):
    cos = np.cos(theta)
    if quick_sin:
        sin = np.sqrt(1-cos**2)*np.sign(np.pi-theta)
    else:
        sin = np.sin(theta)
    return r*cos,r*sin

def split_2d(r,acos):
    return from_radial(r,theta=random_angle(r,acos))

def split_3d(r):
    radial,perp = from_radial(r,theta=random_angle(r,acos=True))
    x,y = from_radial(perp,theta=random_angle(perp,acos=False))
    return x,y,radial

def joint_clean(arrays,keys,clean_by):
    data = pd.DataFrame(dict(zip(keys,arrays)))
    data = data.drop_duplicates(clean_by).sort_values(clean_by)
    return data.to_numpy().T

def clean_pairs(pairs,blacklist = []):
    cleaned_pairs = pd.DataFrame(pairs).drop_duplicates(0).drop_duplicates(1).to_numpy() #Ensures there are no particles considered multiple times
    if len(blacklist) > 0:
        cleaned_pairs = np.array([pair for pair in cleaned_pairs if pair[0] not in blacklist and pair[1] not in blacklist])
    return cleaned_pairs

def split_pairs_to_batches(pairs,max_iteration=100):
    batches = []
    queue = pairs
    for _ in range(max_iteration):
        if len(queue) == 0:
            break
        encountered = []
        batch = []
        waiting = []
        for pair in queue:
            if pair[0] not in encountered and pair[1] not in encountered:
                batch += [pair]
                encountered += pair
            else:
                waiting += [pair]
        batches += [batch]
        queue = waiting
    return batches

def rank_array(r):
    return r.argsort().argsort()

def mask_to_indices(mask):
    return np.arange(len(mask))[mask]

def indices_to_mask(indices,shape):
    mask = np.full(shape,False)
    if len(indices) > 0:
        mask[indices] = True
    return mask

def derivate(x_array,y_fn,h=1e-4):
    scalar_input = np.isscalar(x_array)
    if scalar_input:
        x_array = np.array([x_array])
    if len(x_array) == 0:
        return np.array([])
    derivative = (y_fn(x_array+h)-y_fn(x_array))/h
    if scalar_input:
        return derivative[0]
    return derivative

def derivate2(x_array,y_fn,h=1e-4):
    scalar_input = np.isscalar(x_array)
    if scalar_input:
        x_array = np.array([x_array])
    if len(x_array) == 0:
        return np.array([])
    derivative = (y_fn(x_array+2*h)-2*y_fn(x_array+h)+y_fn(x_array))/h**2
    if scalar_input:
        return derivative[0]
    return derivative

@njit
def linear_interpolation(xs,ys,x):
    i = np.searchsorted(xs,x) - 1
    if i < 0:
        i = 0
    elif i >= len(xs)-1:
        i = len(xs)-2
    w = (x-xs[i])/(xs[i+1]-xs[i])
    return (1-w)*ys[i]+w*ys[i+1]

def plot_2d(grid,extent=None,x_range=None,y_range=None,x_units:Unit=default_units(''),y_units:Unit=default_units(''),cbar_units:Unit=default_units(''),
            x_nbins:int|None=6,y_nbins:int|None=6,x_tick_format:str='%.0f',y_tick_format:str='%.0f',title='',xlabel='',ylabel='',cbar_label='',
            fig=None,ax=None,**kwargs):
    if extent is None:
        if x_range is None:
            x_range = np.array([1e-2,50])*kpc/x_units['value']
        if y_range is None:
            y_range = np.array([0,100])*(km/second)/y_units['value']
        extent = (x_range.min(),x_range.max(),y_range.min(),y_range.max())

    if fig is None or ax is None:
        fig,ax = plt.subplots(figsize=(6,5))
    fig.tight_layout()
    im = ax.imshow(grid,origin='lower',aspect='auto',extent=extent,**kwargs)
    cbar = fig.colorbar(im,ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label.format(**cbar_units))

    if x_nbins is not None:
        ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=x_nbins))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter(x_tick_format))
        ax.xaxis.tick_bottom()
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
            lab.set_horizontalalignment('center')
    ax.set_xlabel(xlabel.format(**x_units))

    if y_nbins is not None:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=y_nbins))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(y_tick_format))
    ax.set_ylabel(ylabel.format(**y_units))
    if title:
        ax.set_title(title)
    return fig,ax

def plot_phase_space(grid,r_range=None,v_range=None,length_units:Unit=default_units('length'),velocity_units:Unit=default_units('velocity'),
                     x_nbins=6,y_nbins=6,x_tick_format:str='%.0f',y_tick_format:str='%.0f',fig=None,ax=None,**kwargs):
    return plot_2d(grid,x_range=r_range,y_range=v_range,x_units=length_units,y_units=velocity_units,x_nbins=x_nbins,y_nbins=y_nbins,
                   x_tick_format=x_tick_format,y_tick_format=y_tick_format,xlabel='radius [{name}]',ylabel='velocity [{name}]',fig=fig,ax=ax,**kwargs)

@njit(parallel=True)
def fast_assign(indices,array):
    output = np.empty_like(indices,dtype=np.float64)
    for i in prange(len(indices)):
        output[i] = array[indices[i]]
    return output

@njit(parallel=True)
def fast_spherical_rho_integrate(r,rho_fn,rho_s=1,Rs=1,Rvir=1,start=0,num_steps=10000):
    integral = np.empty_like(r,dtype=np.float64)
    for i in prange(len(r)):
        x = np.linspace(start,r[i],num_steps)[1:]
        J = 4*np.pi*x**2
        ys = rho_fn(x,rho_s=rho_s,Rs=Rs,Rvir=Rvir)
        integral[i] = np.trapezoid(y=ys*J,x=x)
    return integral

@njit(parallel=True)
def fast_unique_mask(x):
    """use with np.where(fast_unique_mask(x) > 1)[0] to get all unique elements"""
    output = np.zeros_like(x,dtype=np.int64)
    for i in prange(len(x)):
        output[x[i]] += 1
    return output
