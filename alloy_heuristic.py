import re
import random
import numpy as np
from pymatgen.core import Structure, Lattice, Element


def parse_alloy_formula(formula: str):
    """Parse alloy formula like Fe70Ni30 → {Fe:0.7, Ni:0.3}"""
    tokens = re.findall(r"([A-Z][a-z]*)(\d+)", formula)
    if not tokens:
        raise ValueError("Invalid alloy formula")

    total = sum(int(n) for _, n in tokens)
    return {el: int(n) / total for el, n in tokens}


def build_solid_solution(alloy: dict,
                         prototype: str = "FCC",
                         a: float = 3.6,
                         supercell: int = 2):
    """Build approximate solid-solution structure with random site occupancy"""
    if prototype == "FCC":
        lattice = Lattice.cubic(a)
        basis = [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ]
    elif prototype == "BCC":
        lattice = Lattice.cubic(a)
        basis = [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ]
    else:
        raise ValueError("Only FCC/BCC supported for now")

    elements = list(alloy.keys())
    weights = list(alloy.values())

    species = [random.choices(elements, weights)[0] for _ in basis]

    structure = Structure(lattice, species, basis)
    structure.make_supercell([supercell] * 3)

    return structure