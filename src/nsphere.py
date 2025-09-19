from pathlib import Path
from typing import NotRequired,TypedDict,Unpack,Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray
import regex
from astropy import units

class File_params(TypedDict):
    base_filename: str
    ntimesteps: int
    tfinal: int
    max_time: NotRequired[units.Quantity['time']]
    root_path: NotRequired[str|Path]

# Define the record dtype (as in your original code)
record_dtype = {
    'Rank_Mass_Rad_VRad_unsorted': np.dtype([('rank',  np.int32),
                                             ('mass',  np.float32),
                                             ('R',     np.float32),
                                             ('Vrad',  np.float32),
                                             ('PsiA',  np.float32),
                                             ('E',     np.float32),
                                             ('L',     np.float32)])
}

def gather_files(base_filename:str,ntimesteps:int,tfinal:int,max_time:units.Quantity['time']=1*units.Gyr,
                 root_path:str|Path=r'../../NSphere-SIDM/data/') -> pd.DataFrame:
    if not isinstance(root_path,Path):
        root_path = Path(root_path)
    files = pd.DataFrame({'path':list(root_path.glob(f'{base_filename}_t*_100000_{ntimesteps+1}_{tfinal}.dat'))})
    files['save_step'] = files.path.apply(get_save_step)
    files['time'] = files['save_step']/files['save_step'].max()*max_time
    files['record_dtype'] = record_dtype.get(str(base_filename),{})
    return files.sort_values('time')

def get_save_step(path:Path) -> int:
    return int(regex.findall(r'_t(\d+)_',path.stem)[0])

def load_file(path:str|Path,dtype:np.dtype[Any]) -> NDArray[Any]:
    return np.fromfile(path,dtype=dtype)

def load_all_files(files:pd.DataFrame|None=None,**kwargs:Unpack[File_params]):
    if files is None:
        files = gather_files(**kwargs)
    data: list[pd.DataFrame] = []
    for path,dtype,time,save_step in tqdm(files[['path','record_dtype','time','save_step']].to_numpy(),desc='Load files'):
        sub = pd.DataFrame(load_file(path,dtype))
        sub['time'] = time
        sub['save_step'] = save_step
        data += [sub]
    return pd.concat(data,ignore_index=True)

def to_saved_state_like(data:pd.DataFrame):
    data = data.rename(columns={'Vrad':'vr','R':'r','save_step':'step'})
    data['vp'] = data['L']/data['r']
    data['v_norm'] = np.sqrt(data['vp']**2+data['vr']**2)
    data = data.drop(columns=['rank','mass','PsiA','E','L'])
    data['live'] = True
    return data
