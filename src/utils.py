import numpy as np
import pandas as pd
from numba import njit,prange
from numpy.typing import NDArray
from typing import Callable,Any,cast
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.ticker as mtick
from astropy import units,table
from astropy.units.typing import UnitLike
from . import run_units
from .types import FloatOrArray

def random_angle(like:NDArray[np.float64],acos:bool) -> NDArray[np.float64]:
    rolls = np.random.rand(len(like)) if len(like.shape) == 1 else np.random.rand(*like.shape)
    if acos:
        return np.acos(rolls*2-1)
    return rolls*2*np.pi

def from_radial(r:NDArray[np.float64],theta:NDArray[np.float64],quick_sin:bool=True) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    cos:NDArray[np.float64] = np.cos(theta)
    sin:NDArray[np.float64] = np.sqrt(1-cos**2)*np.sign(np.pi-theta) if quick_sin else np.sin(theta)
    return r*cos,r*sin

def split_2d(r:NDArray[np.float64],acos:bool) -> tuple[NDArray[np.float64],NDArray[np.float64]]:
    return from_radial(r,theta=random_angle(r,acos))

def split_3d(r:NDArray[np.float64]) -> tuple[NDArray[np.float64],NDArray[np.float64],NDArray[np.float64]]:
    radial,perp = from_radial(r,theta=random_angle(r,acos=True))
    x,y = from_radial(perp,theta=random_angle(perp,acos=False))
    return x,y,radial

def joint_clean(arrays:list[NDArray[Any]],keys:list[str],clean_by:str) -> NDArray[Any]:
    data = pd.DataFrame(dict(zip(keys,arrays)))
    data = data.drop_duplicates(clean_by).sort_values(clean_by)
    return data.to_numpy().T

def clean_pairs(pairs:NDArray[np.int64],blacklist:list[int]|NDArray[np.int64]=[]) -> NDArray[np.int64]:
    cleaned_pairs = pd.DataFrame(pairs).drop_duplicates(0).drop_duplicates(1).to_numpy() #Ensures there are no particles considered multiple times
    if len(blacklist) > 0:
        cleaned_pairs = np.array([pair for pair in cleaned_pairs if pair[0] not in blacklist and pair[1] not in blacklist])
    return cleaned_pairs

def drop_None(**kwargs:Any) -> dict[Any,Any]:
    return {key:value for key,value in kwargs.items() if value is not None}

def rank_array(r:NDArray[Any]) -> NDArray[np.int64]:
    return r.argsort().argsort()

def derivate(x:FloatOrArray,y_fn:Callable[[FloatOrArray],FloatOrArray],h:float=1e-4) -> FloatOrArray:
    return (y_fn(x+h)-y_fn(x))/h

def derivate2(x:FloatOrArray,y_fn:Callable[[FloatOrArray],FloatOrArray],h:float=1e-4) -> FloatOrArray:
    return (y_fn(x+2*h)-2*y_fn(x+h)+y_fn(x))/h**2

def quantity_derivate(x:units.Quantity,y_fn:Callable[[units.Quantity],units.Quantity],h:float=1e-4) -> units.Quantity:
    t = units.Quantity(h,x.unit)
    return cast(units.Quantity,(y_fn(cast(units.Quantity,x+t))-y_fn(x))/t)

def quantity_derivate2(x:units.Quantity,y_fn:Callable[[units.Quantity],units.Quantity],h:float=1e-4) -> units.Quantity:
    t = units.Quantity(h,x.unit)
    return cast(units.Quantity,(y_fn(cast(units.Quantity,x+2*t))-2*y_fn(cast(units.Quantity,x+t))+y_fn(x))/t**2)

@njit
def linear_interpolation(xs:NDArray[np.float64],ys:NDArray[np.float64],x:float) -> NDArray[np.float64]:
    i = np.searchsorted(xs,x) - 1
    if i < 0:
        i = 0
    elif i >= len(xs)-1:
        i = len(xs)-2
    w = (x-xs[i])/(xs[i+1]-xs[i])
    return (1-w)*ys[i]+w*ys[i+1]

def plot_2d(grid:NDArray[Any],extent:tuple[units.Quantity,units.Quantity,units.Quantity,units.Quantity]|None=None,x_range:units.Quantity|None=None,
            y_range:units.Quantity|None=None,x_units:UnitLike=run_units.length,y_units:UnitLike=run_units.velocity,x_nbins:int|None=6,y_nbins:int|None=6,
            x_tick_format:str='%.0f',y_tick_format:str='%.0f',title:str|None=None,xlabel:str|None=None,ylabel:str|None=None,cbar_label:str|None=None,
            fig:Figure|None=None,ax:Axes|None=None,**kwargs:Any) -> tuple[Figure,Axes]:
    if extent is None:
        assert x_range is not None and y_range is not None, "x_range and y_range must be provided if extent is None"
        extent = (x_range.to(x_units).min(),x_range.to(x_units).max(),y_range.to(y_units).min(),y_range.to(y_units).max())
    extent_value = (float(extent[0].to(x_units).value),float(extent[1].to(x_units).value),float(extent[2].to(y_units).value),float(extent[3].to(y_units).value))

    fig,ax = setup_plot(fig=fig,ax=ax,grid=False,title=title,xlabel=xlabel,ylabel=ylabel)
    im = ax.imshow(grid,origin='lower',aspect='auto',extent=extent_value,**kwargs)
    cbar = fig.colorbar(im,ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label)

    if x_nbins is not None:
        ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=x_nbins))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter(x_tick_format))
        ax.xaxis.tick_bottom()
        for lab in ax.get_xticklabels():
            lab.set_rotation(0)
            lab.set_horizontalalignment('center')

    if y_nbins is not None:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=y_nbins))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(y_tick_format))
    return fig,ax

def plot_phase_space(grid:NDArray[Any],r_range:units.Quantity['length']|None=np.array([1e-2,50])*units.kpc,
                     v_range:units.Quantity['velocity']|None=np.array([0,100])*units.Unit('km/second'),length_units:UnitLike=run_units.length,
                     velocity_units:UnitLike=run_units.velocity,**kwargs:Any) -> tuple[Figure,Axes]:
    return plot_2d(grid,xlabel=add_label_unit('Radius',length_units),ylabel=add_label_unit('Velocity',velocity_units),x_units=length_units,
                   y_units=velocity_units,**drop_None(x_range=r_range,y_range=v_range),**kwargs)

@njit(parallel=True)
def fast_assign(indices:NDArray[np.int64],array:NDArray[np.float64]) -> NDArray[np.float64]:
    output = np.empty_like(indices,dtype=np.float64)
    for i in prange(len(indices)):
        output[i] = array[indices[i]]
    return output

@njit(parallel=True)
def fast_spherical_rho_integrate(r:NDArray[np.float64],rho_fn:Callable[...,NDArray[np.float64]],rho_s:float=1,
                                 Rs:float=1,Rvir:float=1,start:float=0,num_steps:int=10000) -> NDArray[np.float64]:
    integral = np.empty_like(r,dtype=np.float64)
    for i in prange(len(r)):
        x_grid = np.linspace(start,r[i],num_steps)[1:]
        x = np.empty_like(x_grid,dtype=np.float64)
        x[:] = x_grid
        J = 4*np.pi*x**2
        ys = rho_fn(x,rho_s=rho_s,Rs=Rs,Rvir=Rvir)
        integral[i] = np.trapezoid(y=ys*J,x=x)
    return integral

@njit(parallel=True)
def fast_unique_mask(x:NDArray[np.int64]) -> NDArray[np.int64]:
    """use with np.where(fast_unique_mask(x) > 1)[0] to get all unique elements"""
    output = np.zeros_like(x,dtype=np.int64)
    for i in prange(len(x)):
        output[x[i]] += 1
    return output

def setup_plot(fig:Figure|None=None,ax:Axes|None=None,grid:bool=True,minorticks:bool=False,figsize:tuple[int,int]|None=(6,5),ax_set:dict[str,str]|None=None,
               title:str|None=None,xlabel:str|None=None,ylabel:str|None=None,**kwargs:Any) -> tuple[Figure,Axes]:
    if fig is None or ax is None:
        fig,ax = plt.subplots(figsize=figsize,**kwargs)
    assert fig is not None and ax is not None
    fig.tight_layout()
    ax.grid(grid)
    if minorticks:
        ax.minorticks_on()
    if ax_set is not None:
        ax.set(**ax_set)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return fig,ax

def aggregate_QTable(data:table.QTable,groupby:str|list[str],keys:str|list[str],agg_fn:str|Callable[[Any],Any],final_units:dict[str,UnitLike]|None=None) -> table.QTable:
    return table.QTable(table.Table.from_pandas(pd.DataFrame(data.to_pandas().groupby(groupby)[keys].agg(agg_fn)),index=True,units=final_units))

def add_label_unit(label:str|None,plot_units:UnitLike) -> str|None:
    if label is None:
        return None
    return f'{label} [{units.Unit(cast(str,plot_units)):latex}]'

@njit(parallel=True)
def fast_norm(x:NDArray[np.float64],square:bool=False) -> NDArray[np.float64]:
    output = np.empty(len(x),dtype=np.float64)
    for i in prange(len(x)):
        s = (x[i]**2).sum()
        if square:
            output[i] = s
        else:
            output[i] = np.sqrt(s)
    return output

@njit(parallel=True)
def fast_v_correction(Psi:NDArray[np.float64],Ein:NDArray[np.float64],v_norm:NDArray[np.float64]) -> NDArray[np.float64]:
    output = np.empty((len(v_norm),1),dtype=np.float64)
    for i in prange(len(v_norm)):
        output[i,0] = np.sqrt(np.abs(2*(Psi[i]-Ein[i])))/v_norm[i]
    return output
