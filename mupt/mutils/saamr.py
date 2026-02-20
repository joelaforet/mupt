"""Utilities for dealing with systems compliant with the Standard All-Atom Molecular Representation
   namely Primitives that conform to the organization of [Universe -> Molecules -> Repeat-Units -> Atoms]"""
   
__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

from ..mupr.primitives import Primitive

def _is_SAAMR_compliant(prim : Primitive) -> bool:
    """
    Check whether a Primitive hierarchy is organized
    as universe -> molecule -> repeat-unit -> atom
    """
    
    return all(
        leaf.is_atom and (leaf.depth == 3)        
            for leaf in prim.leaves
    )
