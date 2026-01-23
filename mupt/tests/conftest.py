'''
File to house various fixtures that are used by multiple tests.
'''

__author__ = 'Joseph R. Laforet Jr.'
__email__ = 'jola3134@colorado.edu'

import pytest

import logging
from typing import Mapping, Iterable, Optional

import numpy as np
import networkx as nx

from ..mupr.primitives import Primitive
from ..mupr.topology import TopologicalStructure
from ..geometry.shapes import Ellipsoid
from ..geometry.coordinates.reference import origin
from ..geometry.coordinates.directions import random_unit_vector
from ..geometry.transforms.rigid import rigid_vector_coalignment
from ..interfaces.smiles import primitive_from_smiles
from ..interfaces.rdkit import suppress_rdkit_logs
from ..builders.random_walk import AngleConstrainedRandomWalk


logger = logging.getLogger(__name__)

# DEV:JRL The following functions are useful helpers to streamline the building 
# of copolymer systems from SMILES. They were taken from the ellipsoidal_chain_placement.ipynb
# tutorial notebook authored by @timbernat

def sequence_repeat_units(
    chain_len: int,
    head_name: str,
    tail_name: str,
    mid_distrib: Mapping[str, float],
) -> Iterable[str]:
    """
    Generate a sequence of repeat unit names for a polymer chain.
    
    Parameters
    ----------
    chain_len : int
        Total length of the polymer chain (number of repeat units, including end groups).
    head_name : str
        Name of the head repeat unit.
    tail_name : str
        Name of the tail repeat unit.
    mid_distrib : Mapping[str, float]
        Distribution of mid repeat units (map from names to probabilities).
        Probabilities must sum to 1.0.
        
    Yields
    ------
    str
        Names of repeat units in sequence from head to tail.
        
    Raises
    ------
    ValueError
        If chain_len < 2 (need at least head and tail).
    """
    if chain_len < 2:
        raise ValueError('Chain length must be at least 2 to accommodate head and tail units')
    
    yield head_name
    for mid_name in np.random.choice(
        list(mid_distrib.keys()),
        size=(chain_len - 2),
        p=list(mid_distrib.values()),
    ).astype(object):
        yield mid_name
    yield tail_name


def build_lexicon(
    rep_unit_smiles: dict[str, str],
    axis: int = 0,
    semiminor_fraction: float = 0.5,
) -> dict[str, Primitive]:
    """
    Build a lexicon of repeat unit Primitives from SMILES strings.
    
    Each repeat unit is oriented along the specified axis with an encompassing
    ellipsoidal shape for coarse-grained representation.
    
    Parameters
    ----------
    rep_unit_smiles : dict[str, str]
        Mapping from unit names to SMILES strings. SMILES should have
        atom map numbers [*:1] and [*:2] marking the head/tail connection sites.
    axis : int, default=0
        Axis along which to orient repeat units (0=X, 1=Y, 2=Z).
    semiminor_fraction : float, default=0.5
        Fraction of major radius for the minor axes of the ellipsoid.
        
    Returns
    -------
    dict[str, Primitive]
        Lexicon mapping unit names to oriented Primitive objects.
    """
    lexicon: dict[str, Primitive] = {}
    
    with suppress_rdkit_logs():
        for unit_name, smiles in rep_unit_smiles.items():
            unitprim = primitive_from_smiles(
                smiles,
                ensure_explicit_Hs=True,
                embed_positions=True,
                label=unit_name,
            )
            
            # Find edge atoms marked with atom map numbers
            head_atom, tail_atom = unitprim.search_hierarchy_by(
                lambda prim: 'molAtomMapNumber' in prim.metadata,
                min_count=2,
            )
            head_pos = head_atom.shape.centroid
            tail_pos = tail_atom.shape.centroid
            
            # Orient along specified axis
            major_radius = np.linalg.norm(tail_pos - head_pos) / 2.0
            axis_vec = np.zeros(3, dtype=float)
            axis_vec[axis] = major_radius
            
            axis_alignment = rigid_vector_coalignment(
                vector1_start=head_pos,
                vector1_end=tail_pos,
                vector2_start=origin(3),
                vector2_end=axis_vec,
                t1=0.5,
                t2=0.0,
            )
            unitprim.rigidly_transform(axis_alignment)
            
            # Set encompassing ellipsoidal shape
            semiminor = semiminor_fraction * major_radius
            radii = np.full(3, semiminor)
            radii[axis] = major_radius
            unitprim.shape = Ellipsoid(radii)
            
            lexicon[unit_name] = unitprim
            logger.debug(f"Built repeat unit '{unit_name}': {len(unitprim.leaves)} atoms")
    
    return lexicon


def build_polymer_system(
    rep_unit_smiles: dict[str, str],
    mid_distrib: dict[str, float],
    n_chains: int,
    chain_len_min: int,
    chain_len_max: int,
    head_name: str = 'head',
    tail_name: str = 'tail',
    bond_length: float = 1.5,
    angle_max_rad: float = np.pi / 4,
    exclusion_radius: float = 20.0,
    axis: int = 0,
    semiminor_fraction: float = 0.5,
    random_seed: Optional[int] = None,
    show_progress: bool = True,
) -> Primitive:
    """
    Build a (random co)polymer system with specified composition.
    
    Constructs a universe Primitive containing multiple polymer chains with
    random lengths and monomer sequences according to the specified distribution.
    Chains are placed using an angle-constrained random walk.
    
    The resulting Primitive hierarchy is structured as:
    universe -> chains -> residues (repeat units) -> atoms
    
    Parameters
    ----------
    rep_unit_smiles : dict[str, str]
        Mapping from unit names to SMILES strings. Must include entries for
        head_name and tail_name. SMILES should have atom map numbers [*:1] and
        [*:2] marking head/tail connection sites.
    mid_distrib : dict[str, float]
        Distribution of middle repeat units (probabilities must sum to 1.0).
    n_chains : int
        Number of polymer chains to generate.
    chain_len_min : int
        Minimum chain length (number of repeat units including end groups).
    chain_len_max : int
        Maximum chain length (inclusive).
    head_name : str, default='head'
        Name of the head repeat unit in rep_unit_smiles.
    tail_name : str, default='tail'
        Name of the tail repeat unit in rep_unit_smiles.
    bond_length : float, default=1.5
        Distance between repeat unit centers along the chain.
    angle_max_rad : float, default=π/4
        Maximum bend angle (radians) between consecutive bonds.
    exclusion_radius : float, default=20.0
        Radius from origin at which to start placing chains.
    axis : int, default=0
        Axis along which to orient repeat units (0=X, 1=Y, 2=Z).
    semiminor_fraction : float, default=0.5
        Fraction of major radius for ellipsoid minor axes.
    random_seed : int, optional
        Random seed for reproducibility.
    show_progress : bool, default=True
        Whether to display a progress bar during construction.
        
    Returns
    -------
    Primitive
        Universe Primitive containing the complete polymer system.
        
    Examples
    --------
    Build a PSU/PES copolymer system:
    
    >>> rep_unit_smiles = {
    ...     'head': '[H]-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*',
    ...     'bisphenol_S': '*-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*',
    ...     'bisphenol_A': '*-[O:1]c1ccc(cc1)C(-C)(-C)c1cc[c:2](cc1)-*',
    ...     'tail': '*-[O:1]c1ccc(cc1)S(=O)(=O)c1ccc(cc1)[O:2]-[H]',
    ... }
    >>> mid_distrib = {'bisphenol_S': 0.4, 'bisphenol_A': 0.6}
    >>> univprim = build_copolymer_system(
    ...     rep_unit_smiles, mid_distrib,
    ...     n_chains=10, chain_len_min=5, chain_len_max=10,
    ... )
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Build lexicon of repeat units
    logger.info(f"Building lexicon with {len(rep_unit_smiles)} repeat unit types")
    lexicon = build_lexicon(rep_unit_smiles, axis=axis, semiminor_fraction=semiminor_fraction)
    
    # Validate that required units are present
    for required in [head_name, tail_name] + list(mid_distrib.keys()):
        if required not in lexicon:
            raise ValueError(f"Missing repeat unit '{required}' in rep_unit_smiles")
    
    # Build universe
    univprim = Primitive(label='universe')
    chain_lengths = np.random.randint(chain_len_min, chain_len_max + 1, size=n_chains)
    
    if show_progress:
        try:
            from rich.progress import Progress, BarColumn, TimeRemainingColumn
            progress_ctx = Progress(
                "[progress.description]{task.description}",
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeRemainingColumn(),
            )
        except ImportError:
            show_progress = False
    
    def _build_chains():
        for chain_idx, chain_len in enumerate(chain_lengths):
            # Build chain hierarchy
            molprim = Primitive(label=f'{chain_len}-mer_chain')
            
            for unit_name in sequence_repeat_units(
                chain_len,
                head_name=head_name,
                tail_name=tail_name,
                mid_distrib=mid_distrib,
            ):
                rep_unit_prim = lexicon[unit_name].copy()
                molprim.attach_child(rep_unit_prim)
            
            # Set path graph topology
            molprim.set_topology(
                nx.path_graph(
                    molprim.children_by_handle.keys(),
                    create_using=TopologicalStructure,
                ),
                max_registration_iter=100,
            )
            
            # Place beads by random walk
            direction = random_unit_vector()
            # DEV-JRL: This is where we should think about inserting the DPD Builder!
            builder = AngleConstrainedRandomWalk( 
                bond_length=bond_length,
                angle_max_rad=angle_max_rad,
                initial_point=exclusion_radius * direction,
                initial_direction=direction,
            )
            
            for handle, placement in builder.generate_placements(molprim):
                molprim.children_by_handle[handle].rigidly_transform(placement)
            
            # Attach chain to universe
            univprim.attach_child(molprim)
            
            yield chain_idx, chain_len
    
    if show_progress:
        with progress_ctx as progress:
            task = progress.add_task("Building chains", total=n_chains)
            for chain_idx, chain_len in _build_chains():
                progress.advance(task)
    else:
        for _ in _build_chains():
            pass
    
    total_atoms = len(univprim.leaves)
    total_residues = sum(len(chain.children) for chain in univprim.children)
    logger.info(
        f"Built system: {n_chains} chains, {total_residues} residues, {total_atoms} atoms"
    )
    
    return univprim


'''
NOTE: This concept is confusing! At a high level, the Factory returns
a function that will have custom arguments passed into it. Each time
we call the Factory function, it instantiates a new call to build_polymer_system()
with the parameters we provide when we invoke the Factory.

Factory Fixture Usage Examples
-------------------------------
The factory fixtures (polyethane_factory, BPA_BPS_factory) allow creating
systems with custom parameters in your tests:

    def test_with_custom_system(polyethane_factory):
        # Single chain with specific length
        small_system = polyethane_factory(chain_len=5, n_chains=1)
        
        # Multiple chains with length variation
        large_system = polyethane_factory(
            chain_len_min=10,
            chain_len_max=20,
            n_chains=50
        )
        
        # Custom parameters (bond_length, angle_max_rad, etc.)
        custom_system = polyethane_factory(
            chain_len=10,
            n_chains=5,
            bond_length=2.0,
            angle_max_rad=np.pi/6
        )
    
    def test_with_BPA_BPS(BPA_BPS_factory):
        # Pure BPS homopolymer
        bps_only = BPA_BPS_factory(
            chain_len=15,
            n_chains=10,
            bps_fraction=1.0
        )
        
        # Custom BPA/BPS ratio
        custom_ratio = BPA_BPS_factory(
            chain_len_min=5,
            chain_len_max=15,
            n_chains=20,
            bps_fraction=0.7  # 70% BPS, 30% BPA
        )        
'''

@pytest.fixture
def polyethane_smiles() -> dict[str, str]:
    """SMILES definitions for polyethane"""
    return {
        'head': '[H:1]-[CH2:2]-*',
        'ethane': '*-[CH2:1][CH2:2]-*',
        'tail': '*-[CH2:1]-[H:2]'
    }

# resname_maps are currently used to assign PDB compliant 3-char codes
# to repeat units
@pytest.fixture
def polyethane_resname_map():
    """Residue name mapping for polyethane systems."""
    return {
        'head': "HEA",
        'ethane': "EAN",
        'tail': "TYL"
    }

@pytest.fixture
def BPA_BPS_smiles() -> dict[str, str]:
    """SMILES definitions for BPA/BPS copolymer"""
    return {
        'head': '[H]-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*',
        'bisphenol_S': '*-[O:1]c1ccc(cc1)S(=O)(=O)c1cc[c:2](cc1)-*',
        'bisphenol_A': '*-[O:1]c1ccc(cc1)C(-C)(-C)c1cc[c:2](cc1)-*',
        'tail': '*-[O:1]c1ccc(cc1)S(=O)(=O)c1ccc(cc1)[O:2]-[H]',
    }

@pytest.fixture
def BPA_BPS_resname_map():
    """Residue name mapping for BPA/BPS copolymer systems."""
    return {
        'head': "HED",
        'bisphenol_S': "BPS",
        'bisphenol_A': "BPA",
        'tail': "TAL"
    }

@pytest.fixture
def polyethane_factory(polyethane_smiles):# -> Callable[..., Primitive]:
    """
    Factory for creating polyethane systems with configurable parameters.
    
    Returns a function that builds polyethane systems with specified:
    - Chain length (number of repeat units)
    - Number of chains
    - Other build parameters
    
    Examples
    --------
    >>> def test_something(polyethane_factory):
    ...     # Single 2-mer chain
    ...     system1 = polyethane_factory(chain_len=2, n_chains=1)
    ...     
    ...     # Multiple chains with varying lengths
    ...     system2 = polyethane_factory(
    ...         chain_len_min=5,
    ...         chain_len_max=10,
    ...         n_chains=20
    ...     )
    """
    def _make_polyethane(
        chain_len: Optional[int] = None,
        chain_len_min: Optional[int] = None,
        chain_len_max: Optional[int] = None,
        n_chains: int = 1,
        random_seed: Optional[int] = 42,
        **kwargs
    ) -> Primitive:
        # Handle single chain_len or min/max range
        if chain_len is not None:
            chain_len_min = chain_len_max = chain_len
        elif chain_len_min is None or chain_len_max is None:
            raise ValueError("Must provide either chain_len or both chain_len_min and chain_len_max")
        
        return build_polymer_system(
            polyethane_smiles,
            mid_distrib={'ethane': 1.0},
            n_chains=n_chains,
            chain_len_min=chain_len_min,
            chain_len_max=chain_len_max,
            random_seed=random_seed,
            show_progress=False,
            **kwargs
        )
    return _make_polyethane

@pytest.fixture
def BPA_BPS_factory(BPA_BPS_smiles):
    """
    Factory for creating BPA/BPS copolymer systems with configurable parameters.
    
    Returns a function that builds BPA/BPS systems with specified:
    - Chain length (number of repeat units)
    - Number of chains
    - BPA/BPS ratio
    - Other build parameters
    
    Examples
    --------
    >>> def test_something(BPA_BPS_factory):
    ...     # 5 chains, 40% BPS / 60% BPA
    ...     system1 = BPA_BPS_factory(
    ...         chain_len_min=5,
    ...         chain_len_max=10,
    ...         n_chains=5,
    ...         bps_fraction=0.4
    ...     )
    ...     
    ...     # Pure BPS homopolymer
    ...     system2 = BPA_BPS_factory(
    ...         chain_len=20,
    ...         n_chains=10,
    ...         bps_fraction=1.0
    ...     )
    """
    def _make_BPA_BPS(
        chain_len: Optional[int] = None,
        chain_len_min: Optional[int] = None,
        chain_len_max: Optional[int] = None,
        n_chains: int = 5,
        bps_fraction: float = 0.4,
        random_seed: Optional[int] = 42,
        **kwargs
    ) -> Primitive:
        # Handle single chain_len or min/max range
        if chain_len is not None:
            chain_len_min = chain_len_max = chain_len
        elif chain_len_min is None or chain_len_max is None:
            raise ValueError("Must provide either chain_len or both chain_len_min and chain_len_max")
        
        # Calculate BPA/BPS distribution
        mid_distrib = {
            'bisphenol_S': bps_fraction,
            'bisphenol_A': 1.0 - bps_fraction
        }
        
        return build_polymer_system(
            BPA_BPS_smiles,
            mid_distrib,
            n_chains=n_chains,
            chain_len_min=chain_len_min,
            chain_len_max=chain_len_max,
            random_seed=random_seed,
            show_progress=False,
            **kwargs
        )
    return _make_BPA_BPS
    
@pytest.fixture
def single_polyethane_2mer(polyethane_factory) -> Primitive:
    """
    Fixture providing a Primitive containing a single molecule of
    polyethane composed of 2 repeat units of ethane.
    Primitive is intended to be SAAMR-compliant.
    [Universe -> Molecule -> Repeat-Units -> Atoms]
    
    * should have:
    - 1 chain
    - 2 repeat units
    - 8 atoms
    - 6 intra-residue bonds
    - 1 inter-residue bond
    """
    return polyethane_factory(chain_len=2, n_chains=1)

@pytest.fixture
def single_polyethane_3mer(polyethane_factory) -> Primitive:
    """
    Fixture providing a Primitive containing a single molecule of
    polyethane composed of 3 repeat units of ethane.
    Primitive is intended to be SAAMR-compliant.
    [Universe -> Molecule -> Repeat-Units -> Atoms]
    
    * should have:
    - 1 chain
    - 3 repeat units
    - 14 atoms
    - 11 intra-residue bonds
    - 2 inter-residue bonds
    """
    return polyethane_factory(chain_len=3, n_chains=1)

@pytest.fixture
def multi_polyethane_system(polyethane_factory) -> Primitive:
    """
    Fixture providing a multi-chain polyethane system with varying chain lengths.
    Useful for testing system-level operations.
    
    * should have:
    - 10 chains
    - 5-10 repeat units per chain
    - Variable total atoms/bonds depending on random chain lengths
    """
    return polyethane_factory(chain_len_min=5, chain_len_max=10, n_chains=10)

@pytest.fixture
def BPA_BPS_copolymer(BPA_BPS_factory) -> Primitive:
    """
    Fixture providing a default BPA/BPS copolymer system Primitive.
    Primitive is intended to be SAAMR-compliant.
    [Universe -> Molecule -> Repeat-Units -> Atoms]
    
    Default configuration: 5 chains, 5-10 repeat units per chain, 40% BPS / 60% BPA
    """
    return BPA_BPS_factory(
        chain_len_min=5,
        chain_len_max=10,
        n_chains=5,
        bps_fraction=0.4
    )

@pytest.fixture
def helium_resname_map():
    """Residue name mapping for Helium SAAMR system."""
    return {
        'He_unit': 'HEL'
    }

@pytest.fixture
def single_helium_atom_saamr() -> Primitive:
    """
    Fixture providing the simplest possible SAAMR-compliant system: a single Helium atom.
    
    This serves as a base case for testing with minimal complexity.
    Hierarchy: [Universe -> Molecule -> Repeat-Unit -> Atom]
    
    * should have:
    - 1 molecule
    - 1 repeat unit
    - 1 atom (He) at position [0, 0, 0]
    """
    from ..geometry.shapes import Sphere
    from periodictable import elements
    
    # Create the atom-level primitive (Helium)
    atom_prim = Primitive(label='He')
    atom_prim.element = elements.He  # Set element to make it an atom
    atom_prim.shape = Sphere(0.49)  # 0.49 Angstrom radius at origin
    
    # Create the repeat unit and attach the atom
    repeat_unit_prim = Primitive(label='He_unit')
    repeat_unit_prim.attach_child(atom_prim)
    
    # Create the molecule and attach the repeat unit
    molecule_prim = Primitive(label='He_molecule')
    molecule_prim.attach_child(repeat_unit_prim)
    
    # Create the universe and attach the molecule
    universe_prim = Primitive(label='universe')
    universe_prim.attach_child(molecule_prim)
    
    return universe_prim

