"""MeshFix cleaning of the merged Step 07 surface before TetGen."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import generate_open_volmesh_stable as pipeline
from model_vol_septum.clean_step02_surfaces import surface_quality


def intersecting_surface():
    return pv.Cube().triangulate().merge(
        pv.Cube(center=(.4, .3, .2)).triangulate(), merge_points=False)


class Step8CleaningTests(unittest.TestCase):
    def test_intersections_removed_in_saved_copy_with_original_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "subject_test_ES_LVRV_surf.vtk"
            intersecting_surface().save(source)
            checksum = hashlib.sha256(source.read_bytes()).hexdigest()
            paths, info = pipeline.save_step8_cleaned_lvrv_surface(
                root / "step_08_LVRV_surf_clean", source, "subject_test", "ES")
            output = Path(paths["LVRV_surf"])
            self.assertEqual(output.name, source.name)
            self.assertEqual(output.parent.name, "step_08_LVRV_surf_clean")
            self.assertGreater(info["cleaning"]["before"]["self_intersection_pairs"], 0)
            saved = pv.read(output)
            self.assertEqual(surface_quality(saved)["self_intersection_pairs"], 0)
            self.assertEqual(saved.n_open_edges, 0)
            self.assertEqual(saved.points.dtype, np.float64)
            self.assertNotIn("elemTag", saved.cell_data)
            self.assertEqual(checksum, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_invalid_saved_copy_does_not_replace_previous_clean_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "subject_test_ED_LVRV_surf.vtk"
            pv.Cube().triangulate().save(source)
            output_dir = root / "step_08_LVRV_surf_clean"
            output_dir.mkdir()
            previous = output_dir / source.name
            previous.write_bytes(b"previous validated surface")
            with patch("model_vol_septum.clean_step02_surfaces.clean_surface_meshfix",
                       return_value=(intersecting_surface(), {})):
                with self.assertRaisesRegex(RuntimeError, "self_intersection_pairs"):
                    pipeline.save_step8_cleaned_lvrv_surface(output_dir, source, "subject_test", "ED")
            self.assertEqual(previous.read_bytes(), b"previous validated surface")


if __name__ == "__main__":
    unittest.main()
