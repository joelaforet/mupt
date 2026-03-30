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
from mupt.interfaces.mdanalysis.strategies import AllAtomExportStrategy
from mupt.mupr.primitives import Primitive
from mupt.mupr.roles import PrimitiveRole
from mupt.mupr.properties import assign_SAAMR_roles
from mupt.chemistry import ELEMENTS


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
    "primitive_fixture,resname_fixture",
    [
        ("single_polyethylene_2mer", "polyethylene_resname_map"),
        ("single_polyethylene_3mer", "polyethylene_resname_map"),
        ("multi_polyethylene_system", "polyethylene_resname_map"),
        ("PES_copolymer", "PES_resname_map"),
        ("single_helium_atom_saamr", "helium_resname_map"),
    ],
    ids=["2mer", "3mer", "multi_chain", "PES", "single_helium_atom"],
)
def test_atom_count_preservation(primitive_fixture, resname_fixture, request):
    """
    Parametrized test verifying that primitive_to_mdanalysis preserves all atoms.

    This test runs with multiple different polymer systems to ensure the conversion
    maintains atom count regardless of system size or chemistry type.

    The parametrize decorator runs this test once for each tuple in the list:
    - primitive_fixture: Name of the fixture providing the Primitive system
    - resname_fixture: Name of the fixture providing the residue name map
    - request: Pytest built-in fixture for dynamic fixture loading

    Usage: Add new systems by adding tuples to the parametrize list.
    """
    # Arrange: Dynamically get the fixtures by name
    univprim = request.getfixturevalue(primitive_fixture)
    resname_map = request.getfixturevalue(resname_fixture)

    # Act: Convert to MDAnalysis
    mda_exported_system = primitive_to_mdanalysis(univprim, resname_map=resname_map)

    # Assert: MDAnalysis atom count should match Primitive leaf count
    assert mda_exported_system.atoms.n_atoms == len(univprim.leaves), (
        f"Expected {len(univprim.leaves)} atoms, found {mda_exported_system.atoms.n_atoms}"
    )

@pytest.mark.parametrize(
    "primitive_fixture,resname_fixture",
    [
        ("single_polyethylene_2mer", "polyethylene_resname_map"),
        ("single_polyethylene_3mer", "polyethylene_resname_map"),
        ("multi_polyethylene_system", "polyethylene_resname_map"),
        ("PES_copolymer", "PES_resname_map"),
        ("single_helium_atom_saamr", "helium_resname_map"),
    ],
    ids=["2mer", "3mer", "multi_chain", "PES", "single_helium_atom"],
)
def test_bond_connectivity_preservation(primitive_fixture, resname_fixture, request):
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
    primitive_fixture : str
        Name of the fixture providing the Primitive system
    resname_fixture : str
        Name of the fixture providing the residue name map
    request : FixtureRequest
        Pytest built-in fixture for dynamic fixture loading
    """
    # Arrange: Get the fixtures
    univprim = request.getfixturevalue(primitive_fixture)
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

@pytest.fixture(scope='function')
def non_SAAMR_hierarchy_shallow() -> Primitive:
    """Universe -> Atom directly (depth=1, should be 3). Violates SAAMR."""
    universe = Primitive(label="universe")
    atom = Primitive(label="He", element=ELEMENTS[2])
    universe.attach_child(atom)
    return universe

@pytest.fixture(scope='function')
def non_SAAMR_hierarchy_non_atom_leaf() -> Primitive:
    """Leaf has no element attribute. Violates SAAMR atom requirement."""
    universe = Primitive(label="universe")
    molecule = Primitive(label="mol")
    repeat_unit = Primitive(label="unit")
    non_atom = Primitive(label="not_atom")  # No element → not an atom
    universe.attach_child(molecule)
    molecule.attach_child(repeat_unit)
    repeat_unit.attach_child(non_atom)
    return universe

@pytest.fixture(scope='function')
def SAAMR_hierarchy_helium() -> Primitive:
    """Minimal valid SAAMR structure for testing resname_map validation.
    Hierarchy: Universe -> Molecule -> Repeat-Unit ('unit') -> He atom."""
    universe = Primitive(label="universe")
    molecule = Primitive(label="mol")
    repeat_unit = Primitive(label="unit")
    atom = Primitive(label="He", element=ELEMENTS[2])
    universe.attach_child(molecule)
    molecule.attach_child(repeat_unit)
    repeat_unit.attach_child(atom)
    return universe


@pytest.mark.parametrize(
    "primitive_fixture, resname_map",
    [
        ('non_SAAMR_hierarchy_shallow', {"He": "HEL"}),         # depth=1, should be 3
        ('non_SAAMR_hierarchy_non_atom_leaf', {"unit": "UNT"}), # leaf missing element
    ],
    ids=["shallow_depth", "non_atom_leaf"],
)
def test_mda_export_reject_non_SAAMR(primitive_fixture, resname_map, request):
    """
    Check that non-SAAMR-compliant Primitives raise ValueError when attempting export to MDAnalysis.

    The exporter requires Primitives organized as
    Universe -> Molecules -> Repeat-Units -> Atoms (depth=3 for all leaves,
    all leaves must have an element attribute). 
    Violations of either condition must raise ValueError to give user clear feedback
    """
    univprim = request.getfixturevalue(primitive_fixture)
    with pytest.raises(ValueError):
        primitive_to_mdanalysis(univprim, resname_map=resname_map)

@pytest.mark.parametrize(
    "primitive_fixture, resname_map",
    [
        ("SAAMR_hierarchy_helium", {"unit": "HE"}),     # 2 chars — too short for PDB 3-char requirement
        ("SAAMR_hierarchy_helium", {"unit": "HELL"}),   # 4 chars — too long for PDB 3-char requirement
        ("SAAMR_hierarchy_helium", {}), # missing entry; falls back to label 'unit' (4 chars) → ValueError
    ],
    ids=["too_short", "too_long", "missing_entry"],
)
def test_invalid_resname_map_raises_value_error(primitive_fixture, resname_map, request):
    """
    Check that invalid resname_map entries raise ValueError when attempting export to MDAnalysis.

    The ``_pdb_resname`` helper enforces that residue names are exactly
    3 characters for PDB compliance. This test covers three failure modes:
    - Mapped name too short (2 chars)
    - Mapped name too long (4 chars)
    - Missing map entry where the fallback label itself is not 3 chars
    """
    univprim = request.getfixturevalue(primitive_fixture)
    with pytest.raises(ValueError):
        primitive_to_mdanalysis(univprim, resname_map=resname_map)


# ============================================================================
# STRATEGY-BASED EXPORT TESTS
# ============================================================================

@pytest.mark.parametrize(
    "primitive_fixture,resname_fixture",
    [
        ("single_polyethylene_2mer", "polyethylene_resname_map"),
        ("single_polyethylene_3mer", "polyethylene_resname_map"),
        ("single_helium_atom_saamr", "helium_resname_map"),
    ],
    ids=["2mer_explicit", "3mer_explicit", "helium_explicit"],
)
def test_explicit_strategy_produces_same_result(primitive_fixture, resname_fixture, request):
    """
    Verify that passing AllAtomExportStrategy explicitly produces the
    same atom/bond counts as the default (auto-assign) path.
    """
    univprim = request.getfixturevalue(primitive_fixture)
    resname_map = request.getfixturevalue(resname_fixture)

    # Default path (auto-assigns roles internally)
    mda_default = primitive_to_mdanalysis(univprim, resname_map=resname_map)

    # Explicit strategy path (roles already assigned by fixture)
    strategy = AllAtomExportStrategy()
    mda_explicit = primitive_to_mdanalysis(univprim, resname_map=resname_map, strategy=strategy)

    assert mda_default.atoms.n_atoms == mda_explicit.atoms.n_atoms
    assert mda_default.residues.n_residues == mda_explicit.residues.n_residues
    assert mda_default.segments.n_segments == mda_explicit.segments.n_segments

    # Bond comparison — handle the no-bonds case (e.g. single helium atom)
    default_has_bonds = hasattr(mda_default, "bonds") and hasattr(mda_default.atoms, "_topology") and "bonds" in mda_default.atoms._topology.attrs
    explicit_has_bonds = hasattr(mda_explicit, "bonds") and hasattr(mda_explicit.atoms, "_topology") and "bonds" in mda_explicit.atoms._topology.attrs
    if default_has_bonds and explicit_has_bonds:
        assert len(mda_default.bonds) == len(mda_explicit.bonds)
    else:
        assert default_has_bonds == explicit_has_bonds


def test_strategy_rejects_unroled_primitives():
    """
    AllAtomExportStrategy.validate() must raise ValueError when
    Primitives lack role assignments, with a message pointing to
    assign_SAAMR_roles() or manual role assignment.
    """
    universe = Primitive(label="universe")
    molecule = Primitive(label="mol")
    repeat_unit = Primitive(label="unit")
    atom = Primitive(label="He", element=ELEMENTS[2])
    universe.attach_child(molecule)
    molecule.attach_child(repeat_unit)
    repeat_unit.attach_child(atom)

    strategy = AllAtomExportStrategy()
    with pytest.raises(ValueError, match="assign_SAAMR_roles"):
        strategy.validate(universe)


def test_strategy_rejects_missing_segment_role():
    """
    AllAtomExportStrategy.validate() must raise ValueError when
    no SEGMENT-role nodes exist, even if root has UNIVERSE role.
    """
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    molecule = Primitive(label="mol")  # No role assigned
    repeat_unit = Primitive(label="unit", role=PrimitiveRole.RESIDUE)
    atom = Primitive(label="He", element=ELEMENTS[2], role=PrimitiveRole.PARTICLE)
    universe.attach_child(molecule)
    molecule.attach_child(repeat_unit)
    repeat_unit.attach_child(atom)

    strategy = AllAtomExportStrategy()
    with pytest.raises(ValueError, match="SEGMENT"):
        strategy.validate(universe)


def test_strategy_rejects_unroled_leaves():
    """
    AllAtomExportStrategy.validate() must raise ValueError when
    leaf Primitives lack PARTICLE role, even if upper levels are set.
    """
    universe = Primitive(label="universe", role=PrimitiveRole.UNIVERSE)
    molecule = Primitive(label="mol", role=PrimitiveRole.SEGMENT)
    repeat_unit = Primitive(label="unit", role=PrimitiveRole.RESIDUE)
    atom = Primitive(label="He", element=ELEMENTS[2])  # No role
    universe.attach_child(molecule)
    molecule.attach_child(repeat_unit)
    repeat_unit.attach_child(atom)

    strategy = AllAtomExportStrategy()
    with pytest.raises(ValueError, match="PARTICLE"):
        strategy.validate(universe)


@pytest.mark.parametrize(
    "primitive_fixture,resname_fixture,expected_segments",
    [
        ("single_polyethylene_2mer", "polyethylene_resname_map", 1),
        ("multi_polyethylene_system", "polyethylene_resname_map", 10),
        ("PES_copolymer", "PES_resname_map", 5),
        ("single_helium_atom_saamr", "helium_resname_map", 1),
    ],
    ids=["2mer_1seg", "multi_10seg", "PES_5seg", "helium_1seg"],
)
def test_segment_count_preservation(
    primitive_fixture, resname_fixture, expected_segments, request
):
    """
    Verify that the number of MDAnalysis segments matches the number
    of SEGMENT-role children in the Primitive hierarchy.
    """
    univprim = request.getfixturevalue(primitive_fixture)
    resname_map = request.getfixturevalue(resname_fixture)

    mda_universe = primitive_to_mdanalysis(univprim, resname_map=resname_map)
    assert mda_universe.segments.n_segments == expected_segments
