from os import PathLike
from pathlib import Path
from typing import NotRequired,TypedDict
import numpy as np
import pandas as pd
import regex

class File_params(TypedDict):
    base_filename: str
    ntimesteps: int
    tfinal: int
    max_time: NotRequired[int]
    root_path: NotRequired[PathLike[str]]

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

def gather_files(base_filename:str,ntimesteps:int,tfinal:int,max_time:int=1,root_path:PathLike[str]=Path('../../NSphere-SIDM/data/')) -> pd.DataFrame:
    if not isinstance(root_path,Path):
        root_path = Path(root_path)
    files = pd.DataFrame({'path':list(root_path.glob(f'{base_filename}_t*_100000_{ntimesteps+1}_{tfinal}.dat'))})
    files['save_step'] = files.path.apply(lambda x:int(regex.findall(r'_t(\d+)_',x.stem)[0]))
    files['time'] = files['save_step']/files['save_step'].max()*max_time
    files['record_dtype'] = record_dtype.get(str(base_filename),{})
    return files.sort_values('time')

def load_file(path:PathLike[str],dtype:np.dtype) -> np.ndarray:
    return np.fromfile(path,dtype=dtype)
