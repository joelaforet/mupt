"""
Tests to ensure export from MuPT to MDAnalysis preserves molecular identity and connectivity.
"""

# Shortcut to run tests for this file:
# python -m pytest mupt/tests/interfaces/mdanalysis/test_exporters.py -v
# With Coverage:
# python -m pytest mupt/tests/interfaces/mdanalysis/test_exporters.py --cov=mupt.interfaces.mdanalysis --cov-report=term -v

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

import pytest
from mupt.interfaces.mdanalysis.exporters import primitive_to_mdanalysis
from mupt.mupr.primitives import Primitive
from periodictable import elements


def count_bonds_in_primitive(univprim):
    """
    Count intra-residue and inter-residue bonds in a SAAMR-compliant Primitive.

    This helper function traverses the Primitive hierarchy and separately counts:
    - Intra-residue bonds (stored in residue.topology)
    - Inter-residue bonds (stored in chain.internal_connections)

    Parameters
    ----------
    univprim : Primitive
        SAAMR-compliant universe Primitive [Universe -> Molecule -> Repeat-Unit -> Atom]

    Returns
    -------
    tuple[int, int]
        (intra_residue_bonds, inter_residue_bonds)
    """
    intra_residue_bonds = 0
    inter_residue_bonds = 0

    # Count intra-residue bonds
    for chain in univprim.children:
        for residue in chain.children:
            # Each residue's topology contains edges between atom handles
            if hasattr(residue, "topology") and residue.topology is not None:
                intra_residue_bonds += len(list(residue.topology.edges()))

    # Count inter-residue bonds
    for chain in univprim.children:
        if hasattr(chain, "internal_connections") and chain.internal_connections:
            inter_residue_bonds += len(chain.internal_connections)

    return intra_residue_bonds, inter_residue_bonds


@pytest.mark.parametrize(
    "fixture_name,resname_fixture",
    [
        ("single_polyethylene_2mer", "polyethylene_resname_map"),
        ("single_polyethylene_3mer", "polyethylene_resname_map"),
        ("multi_polyethylene_system", "polyethylene_resname_map"),
        ("BPA_BPS_copolymer", "BPA_BPS_resname_map"),
        ("single_helium_atom_saamr", "helium_resname_map"),
    ],
    ids=["2mer", "3mer", "multi_chain", "BPA_BPS", "single_helium_atom"],
)
def test_atom_count_preservation(fixture_name, resname_fixture, request):
    """
    Parametrized test verifying that primitive_to_mdanalysis preserves all atoms.

    This test runs with multiple different polymer systems to ensure the conversion
    maintains atom count regardless of system size or chemistry type.

    The parametrize decorator runs this test once for each tuple in the list:
    - fixture_name: Name of the fixture providing the Primitive system
    - resname_fixture: Name of the fixture providing the residue name map
    - request: Pytest built-in fixture for dynamic fixture loading

    Usage: Add new systems by adding tuples to the parametrize list.
    """
    # Arrange: Dynamically get the fixtures by name
    univprim = request.getfixturevalue(fixture_name)
    resname_map = request.getfixturevalue(resname_fixture)

    # Act: Convert to MDAnalysis
    mda_exported_system = primitive_to_mdanalysis(univprim, resname_map=resname_map)

    # Assert: MDAnalysis atom count should match Primitive leaf count
    assert mda_exported_system.atoms.n_atoms == len(univprim.leaves), (
        f"Expected {len(univprim.leaves)} atoms, found {mda_exported_system.atoms.n_atoms}"
    )


@pytest.mark.parametrize(
    "fixture_name,resname_fixture",
    [
        ("single_polyethylene_2mer", "polyethylene_resname_map"),
        ("single_polyethylene_3mer", "polyethylene_resname_map"),
        ("multi_polyethylene_system", "polyethylene_resname_map"),
        ("BPA_BPS_copolymer", "BPA_BPS_resname_map"),
        ("single_helium_atom_saamr", "helium_resname_map"),
    ],
    ids=["2mer", "3mer", "multi_chain", "BPA_BPS", "single_helium_atom"],
)
def test_bond_connectivity_preservation(fixture_name, resname_fixture, request):
    """
    Parametrized test verifying that primitive_to_mdanalysis preserves bond connectivity.

    This test ensures that the bonding structure (topology) is correctly transferred
    from the MuPT Primitive hierarchy to the MDAnalysis Universe. It separately counts:
    - Intra-residue bonds (within repeat units)
    - Inter-residue bonds (between repeat units in a chain)

    The test verifies that the sum of intra-residue and inter-residue bonds from
    the Primitive equals the total bond count in the exported MDAnalysis Universe.
    This ensures no bonds are lost, duplicated, or misclassified during export.

    Parameters
    ----------
    fixture_name : str
        Name of the fixture providing the Primitive system
    resname_fixture : str
        Name of the fixture providing the residue name map
    request : FixtureRequest
        Pytest built-in fixture for dynamic fixture loading
    """
    # Arrange: Get the fixtures
    univprim = request.getfixturevalue(fixture_name)
    resname_map = request.getfixturevalue(resname_fixture)

    # Count bonds in original Primitive (separately by type)
    intra_residue_bonds, inter_residue_bonds = count_bonds_in_primitive(univprim)
    expected_total_bonds = intra_residue_bonds + inter_residue_bonds

    # Act: Convert to MDAnalysis
    mda_exported_system = primitive_to_mdanalysis(univprim, resname_map=resname_map)

    # Get actual bond count from MDAnalysis
    # MDAnalysis stores bonds as a TopologyAttr, accessible via universe.bonds
    if hasattr(mda_exported_system, "bonds") and mda_exported_system.bonds is not None:
        actual_bond_count = len(mda_exported_system.bonds)
    else:
        actual_bond_count = 0

    # Assert: Sum of intra + inter bonds should equal MDAnalysis total
    assert actual_bond_count == expected_total_bonds, (
        f"Bond count mismatch: Primitive has {intra_residue_bonds} intra-residue + "
        f"{inter_residue_bonds} inter-residue = {expected_total_bonds} total bonds, "
        f"but MDAnalysis Universe has {actual_bond_count} bonds"
    )


# ============================================================================
# NEGATIVE TEST CASES: Verify proper error handling for invalid inputs
# ============================================================================

# --- Helper builders for invalid Primitives ---
# Each returns a fresh Primitive per call to avoid mutation risks from shared
# mutable anytree NodeMixin state across parametrized test runs.


def _build_non_saamr_shallow() -> Primitive:
    """Universe -> Atom directly (depth=1, should be 3). Violates SAAMR."""
    universe = Primitive(label="universe")
    atom = Primitive(label="He", element=elements.He)
    universe.attach_child(atom)
    return universe


def _build_non_saamr_non_atom_leaf() -> Primitive:
    """Leaf has no element attribute. Violates SAAMR atom requirement."""
    universe = Primitive(label="universe")
    molecule = Primitive(label="mol")
    repeat_unit = Primitive(label="unit")
    non_atom = Primitive(label="not_atom")  # No element → not an atom
    universe.attach_child(molecule)
    molecule.attach_child(repeat_unit)
    repeat_unit.attach_child(non_atom)
    return universe


def _build_valid_saamr_helium() -> Primitive:
    """Minimal valid SAAMR structure for testing resname_map validation.
    Hierarchy: Universe -> Molecule -> Repeat-Unit ('unit') -> He atom."""
    universe = Primitive(label="universe")
    molecule = Primitive(label="mol")
    repeat_unit = Primitive(label="unit")
    atom = Primitive(label="He", element=elements.He)
    universe.attach_child(molecule)
    molecule.attach_child(repeat_unit)
    repeat_unit.attach_child(atom)
    return universe


@pytest.mark.parametrize(
    "build_primitive, resname_map",
    [
        (_build_non_saamr_shallow, {"He": "HEL"}),  # depth=1, should be 3
        (_build_non_saamr_non_atom_leaf, {"unit": "UNT"}),  # leaf missing element
    ],
    ids=["shallow_depth", "non_atom_leaf"],
)
def test_non_saamr_primitive_raises_value_error(build_primitive, resname_map):
    """
    Non-SAAMR-compliant Primitives must be rejected with ValueError.

    The exporter requires Primitives organized as
    Universe -> Molecules -> Repeat-Units -> Atoms (depth=3 for all leaves,
    all leaves must have an element attribute). Violations of either condition
    must raise ValueError so users get clear feedback rather than silent
    corruption of the exported topology.

    Note: the SAAMR compliance check was changed from ``assert`` to
    ``raise ValueError`` in exporters.py, so the expected exception is
    ValueError (not AssertionError).
    """
    with pytest.raises(ValueError):
        primitive_to_mdanalysis(build_primitive(), resname_map=resname_map)


@pytest.mark.parametrize(
    "resname_map",
    [
        {"unit": "HE"},  # 2 chars — too short for PDB 3-char requirement
        {"unit": "HELL"},  # 4 chars — too long for PDB 3-char requirement
        {},  # missing entry; falls back to label 'unit' (4 chars) → ValueError
    ],
    ids=["too_short", "too_long", "missing_entry"],
)
def test_invalid_resname_map_raises_value_error(resname_map):
    """
    Invalid resname_map entries must raise ValueError.

    The ``_pdb_resname`` helper enforces that residue names are exactly
    3 characters for PDB compliance. This test covers three failure modes:
    - Mapped name too short (2 chars)
    - Mapped name too long (4 chars)
    - Missing map entry where the fallback label itself is not 3 chars

    Uses pytest.raises (not pytest.mark.xfail) because these tests verify
    intentional error-handling behavior, not known failures.
    """
    univprim = _build_valid_saamr_helium()
    with pytest.raises(ValueError):
        primitive_to_mdanalysis(univprim, resname_map=resname_map)
