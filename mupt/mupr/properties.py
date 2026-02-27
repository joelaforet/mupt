"""Utility methods for Primitive objects that determine various properties like exportability
or adherence to a canonical organization (i.e. SAAMR-compliant primitives)"""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from .primitives import Primitive


"""Utilities for dealing with systems compliant with the Standard All-Atom Molecular Representation
   namely Primitives that conform to the organization of [Universe -> Molecules -> Repeat-Units -> Atoms]"""


def _is_SAAMR_compliant(prim: Primitive) -> bool:
    """
    Check whether a Primitive hierarchy is organized
    as universe -> molecule -> repeat-unit -> atom
    """

    return all(leaf.is_atom and (leaf.depth == 3) for leaf in prim.leaves)
