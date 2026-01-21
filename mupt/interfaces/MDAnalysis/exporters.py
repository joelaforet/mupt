"""
MuPT to MDAnalysis Topology Exporter

This module provides functionality to convert MuPT Representation objects
(univprim) into MDAnalysis Universe objects, focusing on topology information
(atoms, residues, segments, and bonds).
"""

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds

import numpy as np
from typing import Optional

from ...mupr.primitives import Primitive
from ...mutils.allatomutils import _is_AA_export_compliant
from ...chemistry.core import BOND_ORDER

# DEV:JRL -> Debating whether to keep resname_map optional.
def _pdb_resname(label: str, resname_map: Optional[dict]) -> str:
    """
    Map a residue label to a PDB-compliant 3-character residue name.

    This helper function is used to ensure residue names are valid for 
    PDB export and downstream visualization (e.g., in PyMOL). 
    It optionally applies a user-provided mapping from original labels 
    to 3-letter codes and enforces uppercase formatting.

    Parameters
    ----------
    label : str
        Original residue label from the Primitive object.
    resname_map : dict, optional
        Optional mapping from residue labels to 3-character PDB residue names. 
        If the label is in the dictionary, the mapped value is used. 
        Otherwise, the original label is returned.

    Returns
    -------
    str
        Uppercase, 3-character PDB-compliant residue name.

    Raises
    ------
    ValueError
        If the resulting residue name is not exactly 3 characters long.

    Examples
    --------
    >>> _pdb_resname('head', {'head': 'HEA', 'tail': 'TAL'})
    'HEA'
    >>> _pdb_resname('mid', {'head': 'HEA', 'tail': 'TAL'})
    'MID'
    """
    if resname_map and label in resname_map:
        name = resname_map[label]
    else:
        name = label

    if len(name) != 3:
        raise ValueError(
            f"Residue name '{name}' (from '{label}') is not 3 characters long"
        )
    return name.upper()