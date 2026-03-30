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
import numpy as np
import MDAnalysis as mda
from MDAnalysis.core.topologyattrs import Bonds
from collections import Counter

from mupt.interfaces.mdanalysis.exporters import primitive_to_mdanalysis
from mupt.interfaces.mdanalysis.strategies import AllAtomExportStrategy, _pdb_resname
from mupt.mupr.primitives import Primitive
from mupt.mupr.roles import PrimitiveRole
from mupt.mupr.properties import assign_SAAMR_roles, is_SAAMR_compliant
from mupt.chemistry import ELEMENTS
from mupt.chemistry.core import BOND_ORDER


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


# ============================================================================
# DEV-ONLY SNAPSHOT COMPARISON (remove before merging to main)
# ============================================================================
# The function below reproduces the OLD hard-coded SAAMR exporter logic
# exactly as it existed prior to the strategy refactor.  It is used solely
# to verify output parity between old and new code paths during development.

def _old_exporter_build_universe(
    univprim: Primitive,
    resname_map: dict[str, str],
) -> mda.Universe:
    """
    Reproduce the pre-refactor primitive_to_mdanalysis() logic verbatim.

    **DEV-ONLY** — must be removed before merging to main.
    """
    if not is_SAAMR_compliant(univprim):
        raise ValueError("Primitive is not SAAMR-compliant.")

    atom_elements = []
    atom_names = []
    atom_positions = []
    atom_resindex = []
    atom_segindex = []
    residue_names = []
    residue_segindex = []
    residue_ids = []
    bonds = []
    bond_orders = []
    bonds_set: set[tuple[int, int]] = set()
    residue_atom_maps: list[list] = []
    residue_to_atom_global_idx: list[dict] = []

    atom_idx = 0
    res_idx = 0

    for chain_idx, chain in enumerate(univprim.children):
        resid_counter = 1

        for residue in chain.children:
            residue_names.append(_pdb_resname(residue.label, resname_map))
            residue_segindex.append(chain_idx)
            residue_ids.append(resid_counter)

            atom_handle_to_global: dict = {}
            local_atoms_in_residue: list = []

            for atom_handle, atom in residue.children_by_handle.items():
                atom_elements.append(atom.element.symbol)
                atom_names.append(atom.element.symbol)

                if hasattr(atom, "shape") and atom.shape is not None:
                    atom_positions.append(atom.shape.centroid)
                else:
                    atom_positions.append([0.0, 0.0, 0.0])

                atom_resindex.append(res_idx)
                atom_segindex.append(chain_idx)
                atom_handle_to_global[atom_handle] = atom_idx
                local_atoms_in_residue.append(atom_handle)
                atom_idx += 1

            residue_atom_maps.append(local_atoms_in_residue)
            residue_to_atom_global_idx.append(atom_handle_to_global)

            # Intra-residue bonds
            if hasattr(residue, "topology") and residue.topology is not None:
                for atom_handle_1, atom_handle_2 in residue.topology.edges():
                    if (
                        atom_handle_1 in atom_handle_to_global
                        and atom_handle_2 in atom_handle_to_global
                    ):
                        global_idx_1 = atom_handle_to_global[atom_handle_1]
                        global_idx_2 = atom_handle_to_global[atom_handle_2]
                        bond_pair = tuple(sorted([global_idx_1, global_idx_2]))
                        if bond_pair not in bonds_set:
                            bonds.append(bond_pair)
                            bonds_set.add(bond_pair)

                            bond_order = 1.0
                            if hasattr(residue, "internal_connections"):
                                for conn_ref1, conn_ref2 in residue.internal_connections:
                                    if (
                                        conn_ref1.primitive_handle == atom_handle_1
                                        and conn_ref2.primitive_handle == atom_handle_2
                                    ) or (
                                        conn_ref1.primitive_handle == atom_handle_2
                                        and conn_ref2.primitive_handle == atom_handle_1
                                    ):
                                        atom1 = residue.fetch_child(atom_handle_1)
                                        if conn_ref1.connector_handle in atom1.connectors:
                                            connector = atom1.connectors[
                                                conn_ref1.connector_handle
                                            ]
                                            if (
                                                hasattr(connector, "bondtype")
                                                and connector.bondtype in BOND_ORDER
                                            ):
                                                bond_order = BOND_ORDER[connector.bondtype]
                                        break
                            bond_orders.append(bond_order)

            resid_counter += 1
            res_idx += 1

    # Inter-residue bonds
    for chain_idx, chain in enumerate(univprim.children):
        residue_handle_to_global_res_idx: dict = {}
        global_res_offset = sum(
            len(univprim.children[c].children) for c in range(chain_idx)
        )
        for local_res_idx, (res_handle, _residue) in enumerate(
            chain.children_by_handle.items()
        ):
            residue_handle_to_global_res_idx[res_handle] = global_res_offset + local_res_idx

        if hasattr(chain, "internal_connections") and chain.internal_connections:
            for conn_ref1, conn_ref2 in chain.internal_connections:
                residue1_handle = conn_ref1.primitive_handle
                residue2_handle = conn_ref2.primitive_handle
                residue1 = chain.fetch_child(residue1_handle)
                residue2 = chain.fetch_child(residue2_handle)

                if conn_ref1.connector_handle not in residue1.external_connectors:
                    continue
                if conn_ref2.connector_handle not in residue2.external_connectors:
                    continue

                atom_ref1 = residue1.external_connectors[conn_ref1.connector_handle]
                atom_ref2 = residue2.external_connectors[conn_ref2.connector_handle]
                atom1_handle = atom_ref1.primitive_handle
                atom2_handle = atom_ref2.primitive_handle

                global_res1_idx = residue_handle_to_global_res_idx[residue1_handle]
                global_res2_idx = residue_handle_to_global_res_idx[residue2_handle]

                if atom1_handle not in residue_to_atom_global_idx[global_res1_idx]:
                    continue
                if atom2_handle not in residue_to_atom_global_idx[global_res2_idx]:
                    continue

                global_atom1_idx = residue_to_atom_global_idx[global_res1_idx][atom1_handle]
                global_atom2_idx = residue_to_atom_global_idx[global_res2_idx][atom2_handle]

                bond_pair = tuple(sorted([global_atom1_idx, global_atom2_idx]))
                if bond_pair not in bonds_set:
                    bonds.append(bond_pair)
                    bonds_set.add(bond_pair)

                    bond_order = 1.0
                    if conn_ref1.connector_handle in residue1.connectors:
                        connector = residue1.connectors[conn_ref1.connector_handle]
                        if (
                            hasattr(connector, "bondtype")
                            and connector.bondtype in BOND_ORDER
                        ):
                            bond_order = BOND_ORDER[connector.bondtype]
                    bond_orders.append(bond_order)

    # Build MDAnalysis Universe
    atom_positions_arr = np.asarray(atom_positions, dtype=float)
    atom_resindex_arr = np.asarray(atom_resindex, dtype=int)
    residue_segindex_arr = np.asarray(residue_segindex, dtype=int)

    num_atoms = len(atom_resindex)
    num_residues = len(residue_names)
    num_segments = len(univprim.children)

    universe = mda.Universe.empty(
        num_atoms,
        n_residues=num_residues,
        n_segments=num_segments,
        atom_resindex=atom_resindex_arr,
        residue_segindex=residue_segindex_arr,
        trajectory=True,
    )

    universe.add_TopologyAttr("name", atom_names)
    universe.add_TopologyAttr("type", atom_elements)
    universe.add_TopologyAttr("element", atom_elements)
    universe.add_TopologyAttr("resname", residue_names)
    universe.add_TopologyAttr("resid", residue_ids)

    segids = [str(i + 1) for i in range(num_segments)]
    universe.add_TopologyAttr("segid", segids)

    if bonds:
        if bond_orders and len(bond_orders) == len(bonds):
            bond_attr = Bonds(np.asarray(bonds, dtype=np.int32), order=bond_orders)
            universe.add_TopologyAttr(bond_attr)
        else:
            universe.add_TopologyAttr("bonds", np.asarray(bonds, dtype=np.int32))

    universe.atoms.positions = atom_positions_arr
    return universe


_SNAPSHOT_SAAMR_FIXTURES = [
    ("single_polyethylene_2mer", "polyethylene_resname_map"),
    ("single_polyethylene_3mer", "polyethylene_resname_map"),
    ("multi_polyethylene_system", "polyethylene_resname_map"),
    ("PES_copolymer", "PES_resname_map"),
    ("single_helium_atom_saamr", "helium_resname_map"),
]
_SNAPSHOT_IDS = ["2mer", "3mer", "multi_chain", "PES", "single_helium"]


def _has_bonds(u: mda.Universe) -> bool:
    """Check whether an MDAnalysis Universe has bond topology."""
    return (
        hasattr(u, "bonds")
        and hasattr(u.atoms, "_topology")
        and "bonds" in u.atoms._topology.attrs
    )


def _bond_set(u: mda.Universe) -> set[tuple[int, int]]:
    """Return the set of sorted bond-index pairs, or empty set if no bonds."""
    if not _has_bonds(u):
        return set()
    return {tuple(sorted(b.indices)) for b in u.bonds}


@pytest.mark.parametrize(
    "primitive_fixture,resname_fixture",
    _SNAPSHOT_SAAMR_FIXTURES,
    ids=_SNAPSHOT_IDS,
)
class TestSnapshotComparison:
    """
    DEV-ONLY — remove this class (and ``_old_exporter_build_universe``)
    before merging to main.

    These tests verify that the new strategy-based exporter produces
    **identical** output to the old hard-coded SAAMR exporter for every
    SAAMR-compliant test fixture.
    """

    def _build_pair(self, primitive_fixture, resname_fixture, request):
        """Helper: build old and new universes for comparison."""
        univprim = request.getfixturevalue(primitive_fixture)
        resname_map = request.getfixturevalue(resname_fixture)
        old_u = _old_exporter_build_universe(univprim, resname_map)
        new_u = primitive_to_mdanalysis(univprim, resname_map=resname_map)
        return old_u, new_u

    def test_snapshot_atom_count(self, primitive_fixture, resname_fixture, request):
        """Old and new exporters produce the same number of atoms."""
        old_u, new_u = self._build_pair(primitive_fixture, resname_fixture, request)
        assert old_u.atoms.n_atoms == new_u.atoms.n_atoms

    def test_snapshot_atom_elements(self, primitive_fixture, resname_fixture, request):
        """Element symbol at every atom index matches."""
        old_u, new_u = self._build_pair(primitive_fixture, resname_fixture, request)
        for i in range(old_u.atoms.n_atoms):
            assert old_u.atoms[i].element == new_u.atoms[i].element, (
                f"Element mismatch at atom {i}: old={old_u.atoms[i].element}, "
                f"new={new_u.atoms[i].element}"
            )

    def test_snapshot_atom_positions(self, primitive_fixture, resname_fixture, request):
        """Atom positions match within floating-point tolerance."""
        old_u, new_u = self._build_pair(primitive_fixture, resname_fixture, request)
        np.testing.assert_allclose(
            old_u.atoms.positions, new_u.atoms.positions,
            atol=1e-12,
            err_msg="Atom positions differ between old and new exporter",
        )

    def test_snapshot_residue_assignment(self, primitive_fixture, resname_fixture, request):
        """Every atom maps to the same residue index."""
        old_u, new_u = self._build_pair(primitive_fixture, resname_fixture, request)
        for i in range(old_u.atoms.n_atoms):
            assert old_u.atoms[i].resindex == new_u.atoms[i].resindex, (
                f"resindex mismatch at atom {i}: old={old_u.atoms[i].resindex}, "
                f"new={new_u.atoms[i].resindex}"
            )

    def test_snapshot_segment_assignment(self, primitive_fixture, resname_fixture, request):
        """Every atom maps to the same segment index."""
        old_u, new_u = self._build_pair(primitive_fixture, resname_fixture, request)
        for i in range(old_u.atoms.n_atoms):
            assert old_u.atoms[i].segindex == new_u.atoms[i].segindex, (
                f"segindex mismatch at atom {i}: old={old_u.atoms[i].segindex}, "
                f"new={new_u.atoms[i].segindex}"
            )

    def test_snapshot_bond_set(self, primitive_fixture, resname_fixture, request):
        """The set of bond pairs (sorted tuples) is identical."""
        old_u, new_u = self._build_pair(primitive_fixture, resname_fixture, request)
        old_bonds = _bond_set(old_u)
        new_bonds = _bond_set(new_u)
        assert old_bonds == new_bonds, (
            f"Bond set mismatch:\n"
            f"  In old but not new: {old_bonds - new_bonds}\n"
            f"  In new but not old: {new_bonds - old_bonds}"
        )

    def test_snapshot_residue_names(self, primitive_fixture, resname_fixture, request):
        """Residue names match in order."""
        old_u, new_u = self._build_pair(primitive_fixture, resname_fixture, request)
        assert old_u.residues.n_residues == new_u.residues.n_residues
        for i in range(old_u.residues.n_residues):
            assert old_u.residues[i].resname == new_u.residues[i].resname, (
                f"resname mismatch at residue {i}: old={old_u.residues[i].resname}, "
                f"new={new_u.residues[i].resname}"
            )

    def test_snapshot_segment_count(self, primitive_fixture, resname_fixture, request):
        """Segment count matches."""
        old_u, new_u = self._build_pair(primitive_fixture, resname_fixture, request)
        assert old_u.segments.n_segments == new_u.segments.n_segments


# ============================================================================
# DEPTH-4 (NON-SAAMR) EXPORT TESTS
# ============================================================================

class TestDepth4Export:
    """
    Tests for exporting a depth-4 Primitive tree (Universe→Domain→Chain→Residue→Atom)
    where an intermediate "domain" node exists between universe and segments.

    These verify that the strategy-based exporter handles arbitrary tree depth
    when roles are manually assigned, without relying on SAAMR's fixed 3-level
    hierarchy.
    """

    def test_depth4_atom_count(self, depth4_helium_system, helium_resname_map):
        """Depth-4 export preserves all 3 He atoms."""
        strategy = AllAtomExportStrategy()
        mda_u = primitive_to_mdanalysis(
            depth4_helium_system, resname_map=helium_resname_map, strategy=strategy
        )
        assert mda_u.atoms.n_atoms == 3

    def test_depth4_segment_count(self, depth4_helium_system, helium_resname_map):
        """Depth-4 export discovers 2 segments despite intermediate domain node."""
        strategy = AllAtomExportStrategy()
        mda_u = primitive_to_mdanalysis(
            depth4_helium_system, resname_map=helium_resname_map, strategy=strategy
        )
        assert mda_u.segments.n_segments == 2

    def test_depth4_residue_count(self, depth4_helium_system, helium_resname_map):
        """Depth-4 export discovers 2 residues across both chains."""
        strategy = AllAtomExportStrategy()
        mda_u = primitive_to_mdanalysis(
            depth4_helium_system, resname_map=helium_resname_map, strategy=strategy
        )
        assert mda_u.residues.n_residues == 2

    def test_depth4_resnames(self, depth4_helium_system, helium_resname_map):
        """Depth-4 export applies resname_map correctly to all residues."""
        strategy = AllAtomExportStrategy()
        mda_u = primitive_to_mdanalysis(
            depth4_helium_system, resname_map=helium_resname_map, strategy=strategy
        )
        for res in mda_u.residues:
            assert res.resname == "HEL", f"Expected 'HEL', got '{res.resname}'"

    def test_depth4_no_bonds(self, depth4_helium_system, helium_resname_map):
        """Depth-4 helium system has no bonds; export should reflect that."""
        strategy = AllAtomExportStrategy()
        mda_u = primitive_to_mdanalysis(
            depth4_helium_system, resname_map=helium_resname_map, strategy=strategy
        )
        # MDAnalysis raises NoDataError when accessing .bonds if none exist
        has_bonds = (
            hasattr(mda_u, "bonds")
            and hasattr(mda_u.atoms, "_topology")
            and "bonds" in mda_u.atoms._topology.attrs
        )
        if has_bonds:
            assert len(mda_u.bonds) == 0
        # If no bond attribute at all, that's also correct for 0 bonds

    def test_depth4_element_names(self, depth4_helium_system, helium_resname_map):
        """All atoms in depth-4 helium system should be He."""
        strategy = AllAtomExportStrategy()
        mda_u = primitive_to_mdanalysis(
            depth4_helium_system, resname_map=helium_resname_map, strategy=strategy
        )
        for atom in mda_u.atoms:
            assert atom.element == "He", f"Expected 'He', got '{atom.element}'"
