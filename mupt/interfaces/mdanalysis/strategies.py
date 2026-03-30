"""Strategy implementations for MuPT -> MDAnalysis export."""

__author__ = "Joseph R. Laforet Jr."
__email__ = "jola3134@colorado.edu"

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from collections.abc import Hashable
from typing import Optional

from anytree import PreOrderIter

from ...chemistry.core import BOND_ORDER
from ...mupr.embedding import ConnectorReference
from ...mupr.primitives import Primitive
from ...mupr.roles import PrimitiveRole

LOGGER = logging.getLogger(__name__)


def _pdb_resname(label: Hashable, resname_map: dict[str, str]) -> str:
    """Map a residue label to a PDB-compliant 3-character residue name.

    Parameters
    ----------
    label : Hashable
        Original residue label from the Primitive object.  Non-string
        labels are normalized to ``str(label)`` before lookup and
        length validation.
    resname_map : dict[str, str]
        Mapping from residue labels to 3-character PDB residue names.

    Returns
    -------
    str
        Uppercase, 3-character PDB-compliant residue name.

    Raises
    ------
    ValueError
        If the resulting residue name is not exactly 3 characters long.
    """
    label = str(label)
    if resname_map and label in resname_map:
        name = resname_map[label]
    else:
        name = label

    if len(name) != 3:
        raise ValueError(f"Residue name '{name}' (from '{label}') is not 3 characters long")
    return name.upper()


def _resolve_to_atom(parent: Primitive, conn_ref: ConnectorReference) -> Primitive:
    """Recursively follow external_connectors to find the leaf atom."""
    child = parent.fetch_child(conn_ref.primitive_handle)
    if child.is_atom:
        return child

    next_ref = child.external_connectors[conn_ref.connector_handle]
    return _resolve_to_atom(child, next_ref)


def _bond_order_from_conn_ref(parent: Primitive, conn_ref: ConnectorReference) -> float:
    """Infer bond order from a connection reference at the current parent level."""
    child = parent.fetch_child(conn_ref.primitive_handle)
    connector = child.connectors.get(conn_ref.connector_handle)
    if connector is None:
        return 1.0

    bondtype: Optional[str] = getattr(connector, "bondtype", None)
    return BOND_ORDER.get(bondtype, 1.0)


@dataclass
class MDATopologyData:
    """Container for topology arrays/lists used to build an MDAnalysis Universe."""

    atom_elements: list[str] = field(default_factory=list)
    atom_names: list[str] = field(default_factory=list)
    atom_positions: list[list[float]] = field(default_factory=list)
    atom_resindex: list[int] = field(default_factory=list)
    atom_segindex: list[int] = field(default_factory=list)
    residue_names: list[str] = field(default_factory=list)
    residue_segindex: list[int] = field(default_factory=list)
    residue_ids: list[int] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_orders: list[float] = field(default_factory=list)
    bonds_set: set[tuple[int, int]] = field(default_factory=set)
    num_segments: int = 0


class MDAExportStrategy(ABC):
    """Abstract strategy for collecting MDAnalysis-exportable topology data."""

    @abstractmethod
    def validate(self, root: Primitive) -> None:
        """Validate role assignment and hierarchy preconditions for export."""

    @abstractmethod
    def collect_topology(self, root: Primitive, resname_map: dict[str, str]) -> MDATopologyData:
        """Collect topology attributes from a Primitive hierarchy."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable name for this strategy."""


class AllAtomExportStrategy(MDAExportStrategy):
    """All-atom export strategy based on role-aware hierarchy traversal."""

    @property
    def label(self) -> str:
        """Human-readable strategy name."""
        return "All-atom"

    def validate(self, root: Primitive) -> None:
        """Validate role assignments needed for all-atom export."""
        if root.role != PrimitiveRole.UNIVERSE:
            raise ValueError(
                "Root Primitive must have role=PrimitiveRole.UNIVERSE. "
                "Assign roles via assign_SAAMR_roles() or set them manually."
            )

        segment_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.SEGMENT]
        residue_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.RESIDUE]
        if not segment_nodes:
            raise ValueError(
                "No SEGMENT-role Primitives found in hierarchy. "
                "Assign roles via assign_SAAMR_roles() or set them manually."
            )
        if not residue_nodes:
            raise ValueError(
                "No RESIDUE-role Primitives found in hierarchy. "
                "Assign roles via assign_SAAMR_roles() or set them manually."
            )

        for leaf in root.leaves:
            if leaf.role != PrimitiveRole.PARTICLE:
                raise ValueError(
                    "All leaves must have role=PrimitiveRole.PARTICLE. "
                    "Assign roles via assign_SAAMR_roles() or set them manually."
                )

    def collect_topology(self, root: Primitive, resname_map: dict[str, str]) -> MDATopologyData:
        """Walk the hierarchy and gather MDAnalysis topology arrays/lists."""
        self.validate(root)
        data = MDATopologyData()

        # Build global atom index from DFS pre-order leaf traversal
        leaves = list(root.leaves)
        n_atoms = len(leaves)
        atom_id_to_global: dict[int, int] = {id(atom): idx for idx, atom in enumerate(leaves)}

        # Pre-allocate resindex/segindex arrays so we can write by global index
        data.atom_resindex = [0] * n_atoms
        data.atom_segindex = [0] * n_atoms

        # Collect per-atom data in DFS leaf order
        for atom in leaves:
            if atom.element is None:
                raise ValueError(
                    f"Leaf Primitive '{atom}' has role=PARTICLE but no element assigned. "
                    "AllAtomExportStrategy requires atomic PARTICLE leaves."
                )
            atom_symbol = atom.element.symbol
            data.atom_elements.append(atom_symbol)
            data.atom_names.append(atom_symbol)

            if hasattr(atom, "shape") and atom.shape is not None:
                data.atom_positions.append(list(atom.shape.centroid))
            else:
                data.atom_positions.append([0.0, 0.0, 0.0])

        # Walk segment -> residue -> particle to build hierarchy mapping
        # Use PreOrderIter to find SEGMENT nodes at any depth (not just direct children)
        segment_nodes = [node for node in PreOrderIter(root) if node.role == PrimitiveRole.SEGMENT]
        data.num_segments = len(segment_nodes)
        mapped_count = 0

        for seg_idx, segment in enumerate(segment_nodes):
            resid_counter = 1
            residue_nodes = [node for node in PreOrderIter(segment) if node.role == PrimitiveRole.RESIDUE]

            for residue in residue_nodes:
                data.residue_names.append(_pdb_resname(residue.label, resname_map))
                data.residue_segindex.append(seg_idx)
                data.residue_ids.append(resid_counter)

                res_global_idx = len(data.residue_names) - 1
                particle_leaves = [
                    node
                    for node in PreOrderIter(residue)
                    if node.is_leaf and node.role == PrimitiveRole.PARTICLE
                ]
                for atom in particle_leaves:
                    atom_global_idx = atom_id_to_global[id(atom)]
                    data.atom_resindex[atom_global_idx] = res_global_idx
                    data.atom_segindex[atom_global_idx] = seg_idx
                    mapped_count += 1

                resid_counter += 1

        if mapped_count != n_atoms:
            raise ValueError(
                f"Role hierarchy traversal mapped {mapped_count} of {n_atoms} PARTICLE leaves "
                "to a RESIDUE and SEGMENT. Assign roles via assign_SAAMR_roles() or set them manually."
            )

        for node in PreOrderIter(root):
            if node.is_leaf or not node.internal_connections:
                continue

            for conn_ref_pair in node.internal_connections:
                conn_ref1, conn_ref2 = tuple(conn_ref_pair)
                atom1 = _resolve_to_atom(node, conn_ref1)
                atom2 = _resolve_to_atom(node, conn_ref2)

                idx1 = atom_id_to_global[id(atom1)]
                idx2 = atom_id_to_global[id(atom2)]
                bond_pair = tuple(sorted((idx1, idx2)))
                if bond_pair in data.bonds_set:
                    continue

                data.bonds.append(bond_pair)
                data.bonds_set.add(bond_pair)
                data.bond_orders.append(_bond_order_from_conn_ref(node, conn_ref1))

        return data


class CoarseGrainedExportStrategy(MDAExportStrategy):
    """Placeholder strategy for future coarse-grained MDAnalysis export."""

    @property
    def label(self) -> str:
        """Human-readable strategy name."""
        raise NotImplementedError("Coarse-grained export is not yet implemented")

    def validate(self, root: Primitive) -> None:
        """Validate hierarchy for coarse-grained export."""
        raise NotImplementedError("Coarse-grained export is not yet implemented")

    def collect_topology(self, root: Primitive, resname_map: dict[str, str]) -> MDATopologyData:
        """Collect topology for coarse-grained export."""
        raise NotImplementedError("Coarse-grained export is not yet implemented")
