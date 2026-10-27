import unittest

import numpy as np

from partition import (PieceSource, merge_parts, mesh_piece, strip_parts,
                       vertex_owners)
from synthetic_source import SyntheticMITLSource
from tests.common import SMALL_MESH, comm


class PartitionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.source = SyntheticMITLSource(comm(), num_steps=3, **SMALL_MESH)

    @classmethod
    def tearDownClass(cls):
        del cls.source

    def test_strips_cover_every_element_once(self):
        parts = strip_parts(self.source.points, self.source.tris, 4)
        self.assertEqual(sorted(np.unique(parts)), [0, 1, 2, 3])
        counts = np.bincount(parts)
        self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_merge_parts(self):
        np.testing.assert_array_equal(
            merge_parts(np.arange(8), 2), [0, 0, 0, 0, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            merge_parts(np.arange(4), 8)

    def test_pieces_own_every_entity_once(self):
        src = self.source
        parts = strip_parts(src.points, src.tris, 3)
        elem_seen = np.zeros(len(src.tris), dtype=int)
        vert_owned = np.zeros(len(src.points), dtype=int)
        for part in range(3):
            piece = mesh_piece(src.points, src.tris, parts, part, comm())
            elem_seen[piece.elem_gids] += 1
            vert_owned[piece.vert_gids] += piece.node_owned
            # The "global" tag is the full-mesh numbering, and the piece is
            # the same geometry.
            np.testing.assert_array_equal(piece.mesh.globals(2),
                                          piece.elem_gids)
            np.testing.assert_array_equal(piece.mesh.globals(0),
                                          piece.vert_gids)
            np.testing.assert_array_equal(
                piece.mesh.coords().reshape(-1, 2),
                src.points[piece.vert_gids])
        np.testing.assert_array_equal(elem_seen, 1)
        np.testing.assert_array_equal(vert_owned, 1)

    def test_vertex_owner_is_lowest_touching_part(self):
        tris = np.array([[0, 1, 2], [1, 3, 2]])
        np.testing.assert_array_equal(
            vertex_owners(tris, np.array([1, 0]), 4), [1, 0, 0, 0])

    def test_piece_source_restricts_fields(self):
        src = self.source
        parts = strip_parts(src.points, src.tris, 2)
        piece = mesh_piece(src.points, src.tris, parts, 1, comm())
        local = PieceSource(src, piece).fields(1)
        full = src.fields(1)
        for name, where in src.field_centering.items():
            index = piece.elem_gids if where == "elem" else piece.vert_gids
            np.testing.assert_array_equal(local[name], full[name][index])


if __name__ == "__main__":
    unittest.main()
