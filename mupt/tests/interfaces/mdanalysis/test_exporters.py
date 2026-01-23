'''
Tests to ensure export from MuPT to MDAnalysis preserves molecular identity and connectivity.
'''

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

import pytest
from mupt.interfaces.mdanalysis.exporters import primitive_to_mdanalysis

@pytest.mark.parametrize(
    "fixture_name,resname_fixture",
    [
        ("single_polyethane_2mer", "polyethane_resname_map"),
        ("single_polyethane_3mer", "polyethane_resname_map"),
        ("multi_polyethane_system", "polyethane_resname_map"),
        ("BPA_BPS_copolymer", "BPA_BPS_resname_map"),
    ],
    ids=["2mer", "3mer", "multi_chain", "BPA_BPS"]
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


