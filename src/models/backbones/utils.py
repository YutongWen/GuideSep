from inspect import isfunction
from typing import Callable, List, Optional, Sequence, TypeVar, Union, Dict, Tuple
from typing_extensions import TypeGuard
import collections.abc
from itertools import repeat
from omegaconf import ListConfig

T = TypeVar("T")
# helper
def group_dict_by_prefix(prefix: str, d: Dict) -> Tuple[Dict, Dict]:
    return_dicts: Tuple[Dict, Dict] = ({}, {})
    for key in d.keys():
        no_prefix = int(not key.startswith(prefix))
        return_dicts[no_prefix][key] = d[key]
    return return_dicts

def groupby(prefix: str, d: Dict, keep_prefix: bool = False) -> Tuple[Dict, Dict]:
    kwargs_with_prefix, kwargs = group_dict_by_prefix(prefix, d)
    if keep_prefix:
        return kwargs_with_prefix, kwargs
    kwargs_no_prefix = {k[len(prefix) :]: v for k, v in kwargs_with_prefix.items()}
    return kwargs_no_prefix, kwargs

def exists(val: Optional[T]) -> TypeGuard[T]:
    return val is not None

def filter_by_keys(fn, d):
    return {k: v for k, v in d.items() if fn(k)}

def map_keys(fn, d):
    return {fn(k): v for k, v in d.items()}

def iff(condition: bool, value: T) -> Optional[T]:
    return value if condition else None

def is_sequence(obj: T) -> TypeGuard[Union[list, tuple]]:
    return isinstance(obj, list) or isinstance(obj, tuple)

def default(val: Optional[T], d: Union[Callable[..., T], T]) -> T:
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, length=None):
    if isinstance(val, list) or isinstance(val, ListConfig):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))
    
    if exists(length):
        assert len(output) == length

    return output

def to_list(val: Union[T, Sequence[T]]) -> List[T]:
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return [val]  # type: ignore

def prefix_dict(prefix: str, d: Dict) -> Dict:
    return {prefix + str(k): v for k, v in d.items()}

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)
