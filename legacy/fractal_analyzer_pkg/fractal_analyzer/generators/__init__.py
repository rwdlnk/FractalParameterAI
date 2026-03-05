# generators/__init__.py
from typing import Dict, Callable, Tuple, List

# Import all generator functions
from .koch import generate_koch
from .sierpinski import generate_sierpinski
from .minkowski import generate_minkowski
from .hilbert import generate_hilbert
from .dragon import generate_dragon

# Create a registry of generator functions
GENERATORS: Dict[str, Callable[[int], Tuple[List, List]]] = {
    'koch': generate_koch,
    'sierpinski': generate_sierpinski,
    'minkowski': generate_minkowski,
    'hilbert': generate_hilbert,
    'dragon': generate_dragon
}

__all__ = ['generate_koch', 'generate_sierpinski', 'generate_minkowski', 
           'generate_hilbert', 'generate_dragon', 'GENERATORS']
