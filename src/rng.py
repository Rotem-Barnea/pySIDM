from typing import Any
from collections.abc import Mapping

import numpy as np

generator = np.random.default_rng()  # Global RNG instance


def set_seed(seed: int | None = None) -> None:
    """Set the random seed for all modules."""
    global generator
    generator = np.random.default_rng(seed)


def get_state() -> Mapping[str, Any]:
    """Get the current state of the RNG for saving.

    Returns:
        Dictionary containing the RNG state.
    """
    return generator.bit_generator.state


def set_state(state: Mapping[str, Any]) -> None:
    """Restore the RNG state from a saved state.

    Parameters:
        state: Dictionary containing the RNG state from `get_state()`.
    """
    global generator
    generator.bit_generator.state = state
