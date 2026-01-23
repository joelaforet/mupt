'''
Tests to ensure export from MuPT to MDAnalysis preserves molecular identity and connectivity.
'''

# Shortcut to run tests for this file:
# python -m pytest mupt/tests/interfaces/mdanalysis/test_exporters.py -v
# With Coverage:
# python -m pytest mupt/tests/interfaces/mdanalysis/test_exporters.py --cov=mupt.interfaces.mdanalysis --cov-report=term -v

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

import pytest
from mupt.interfaces.mdanalysis.exporters import primitive_to_mdanalysis


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
        ("single_polyethane_2mer", "polyethane_resname_map"),
        ("single_polyethane_3mer", "polyethane_resname_map"),
        ("multi_polyethane_system", "polyethane_resname_map"),
        ("BPA_BPS_copolymer", "BPA_BPS_resname_map"),
        ("single_helium_atom_saamr", "helium_resname_map"),
    ],
    ids=["2mer", "3mer", "multi_chain", "BPA_BPS", "single_helium_atom"]
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
    assert mda_exported_system.atoms.n_atoms == len(univprim.leaves), \
        f"Expected {len(univprim.leaves)} atoms, found {mda_exported_system.atoms.n_atoms}"


@pytest.mark.parametrize(
    "fixture_name,resname_fixture",
    [
        ("single_polyethane_2mer", "polyethane_resname_map"),
        ("single_polyethane_3mer", "polyethane_resname_map"),
        ("multi_polyethane_system", "polyethane_resname_map"),
        ("BPA_BPS_copolymer", "BPA_BPS_resname_map"),
        ("single_helium_atom_saamr", "helium_resname_map"),
    ],
    ids=["2mer", "3mer", "multi_chain", "BPA_BPS", "single_helium_atom"]
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
    if hasattr(mda_exported_system, 'bonds') and mda_exported_system.bonds is not None:
        actual_bond_count = len(mda_exported_system.bonds)
    else:
        actual_bond_count = 0
    
    # Assert: Sum of intra + inter bonds should equal MDAnalysis total
    assert actual_bond_count == expected_total_bonds, \
        f"Bond count mismatch: Primitive has {intra_residue_bonds} intra-residue + " \
        f"{inter_residue_bonds} inter-residue = {expected_total_bonds} total bonds, " \
        f"but MDAnalysis Universe has {actual_bond_count} bonds"
