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