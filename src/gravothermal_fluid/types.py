"""Helper types for the gravothermal fluid object"""

from typing import Literal, TypeVar, TypedDict, cast

from src.utils import utils


class RelaxationParams(TypedDict):
    """Parameter dictionary for the HSE relaxation.

    Attributes:
        relaxation_dt_factor: The relaxation time-step, as a multiplicant factor of the heat transfer time-step `dt`.
        max_relaxation_iterations: The maximum number of iterations attempts to make when relaxing the system.
        relaxation_threshold: The threshold for early stopping the iterations, compared with the maximum relative change in shell position before and after the current iteration.
        relaxation_core_fraction: The fraction of shells to consider for early quiting the relaxation, i.e. only consider the inner-most `relaxation_core_fraction`% of them when compared with `relaxation_threshold`.
        driving_force_limit: The maximum driving force to use in relaxing, clips the force_ratio between `1/driving_force_limit` and `driving_force_limit` before taking the log.
    """

    relaxation_dt_factor: float
    max_relaxation_iterations: int
    relaxation_threshold: float
    relaxation_core_fraction: float
    driving_force_limit: float


default_params = {
    'relaxation': {
        'relaxation_dt_factor': 1 / 100,
        'max_relaxation_iterations': 50,
        'relaxation_threshold': 1e-12,
        'relaxation_core_fraction': 0.25,
        'driving_force_limit': 10,
    }
}

ParamsType = TypeVar('ParamsType', RelaxationParams, RelaxationParams)


def normalize_params(params: ParamsType | None, params_type: Literal['relaxation'] = 'relaxation') -> ParamsType:
    """Normalize the parameters to set default values for missing attributes."""
    return cast(type(params), {**default_params[params_type], **utils.handle_default(params, {})})
