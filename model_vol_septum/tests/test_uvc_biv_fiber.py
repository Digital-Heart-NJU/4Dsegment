"""Direct Step 09 labels, coordinate gradients, fiber frames, and cache freshness."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_vol_septum import compute_UVC_fiber as uvc
from model_vol_septum import generate_UVC_fiber as driver


def two_chambers():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                       [0., 0., 1.], [0., 0., -1.]])
    cells = np.array([[4, 0, 1, 2, 3], [4, 0, 2, 1, 4]])
    volume = pv.UnstructuredGrid(cells.ravel(), np.full(2, pv.CellType.TETRA), points)
    volume.cell_data['elemTag'] = np.array([1, 0], dtype=np.int32)
    return volume


def write_carp(volume, base):
    base = Path(base)
    np.savetxt(base.with_suffix('.pts'), volume.points, header=str(volume.n_points), comments='', fmt='%.17g')
    rows = np.column_stack((volume.cells_dict[pv.CellType.TETRA], volume.cell_data['elemTag']))
    np.savetxt(base.with_suffix('.elem'), rows, header=str(volume.n_cells), comments='', fmt='Tt %d %d %d %d %d')
    np.savetxt(base.with_suffix('.lon'), np.tile([1., 0., 0.], (volume.n_cells, 1)), header='1', comments='')


class BivFiberTests(unittest.TestCase):
    def test_biv_uses_current_tags_and_lv_priority_at_shared_nodes(self):
        volume = two_chambers()
        before = volume.points.copy(), volume.cells.copy(), volume['elemTag'].copy()
        values, info = uvc.biv_openbase(volume)
        np.testing.assert_array_equal(values, [-1, -1, -1, -1, 1])
        self.assertEqual(info['shared_interface_nodes'], 3)
        volume.cell_data['elemTag'] = np.array([0, 1])
        swapped, _ = uvc.biv_openbase(volume)
        np.testing.assert_array_equal(swapped, [-1, -1, -1, 1, -1])
        np.testing.assert_array_equal(volume.points, before[0])
        np.testing.assert_array_equal(volume.cells, before[1])

    def test_affine_field_gradient_is_exact_on_both_tetrahedra(self):
        volume = two_chambers()
        slope = np.array([2., -3., 4.])
        values = volume.points @ slope + 7.
        gradient = uvc._tetra_nodal_gradient(volume.points, volume.cells_dict[pv.CellType.TETRA], values)
        np.testing.assert_allclose(gradient, np.tile(slope, (2, 1)), atol=1e-14)

    def test_angles_unit_vectors_and_six_component_lon(self):
        volume = two_chambers()
        z = .5 * (volume.points[:, 2] + 1)
        rho = volume.points[:, 0]
        lon, arrays, info = uvc.fiber_openbase(volume, z, rho)
        self.assertEqual(lon.shape, (2, 6))
        # mean(PHO)=1/4 -> alpha=+30 degrees, radial=x, circumferential=y.
        expected = np.array([0., np.cos(np.pi / 6), np.sin(np.pi / 6)])
        np.testing.assert_allclose(lon[:, :3], np.tile(expected, (2, 1)), atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(lon[:, :3], axis=1), 1., atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(lon[:, 3:], axis=1), 1., atol=1e-14)
        np.testing.assert_allclose(np.einsum('ij,ij->i', lon[:, :3], lon[:, 3:]), 0., atol=1e-14)
        beta = np.deg2rad(-42.5)
        radial = np.tile([1., 0., 0.], (2, 1))
        expected_sheet = np.cos(beta) * radial + np.sin(beta) * np.cross(lon[:, :3], radial)
        np.testing.assert_allclose(lon[:, 3:], expected_sheet, atol=1e-14)
        self.assertEqual(info['repaired_element_count'], 0)
        np.testing.assert_array_equal(arrays['fiber_repair_mask'], [0, 0])

    def repair_fixture(self):
        tetra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        points = np.vstack((tetra, tetra + [20., 0., 0.], tetra + [2., 0., 0.]))
        cells = np.column_stack((np.full(3, 4), np.arange(12).reshape(3, 4)))
        volume = pv.UnstructuredGrid(cells.ravel(), np.full(3, pv.CellType.TETRA), points)
        volume.cell_data['elemTag'] = np.array([1, 1, 0])
        rho = np.r_[np.full(4, .5), tetra[:, 0], tetra[:, 1]]
        return volume, points[:, 2], rho

    def test_flat_pho_gradients_borrow_only_from_the_same_ventricle(self):
        volume, z, rho = self.repair_fixture()
        lon, arrays, info = uvc.fiber_openbase(volume, z, rho)
        np.testing.assert_allclose(lon[0, :3], [0., 1., 0.], atol=1e-14)
        self.assertEqual(info['repairs']['LV']['PHO_gradient'], 1)
        self.assertEqual(info['repairs']['RV']['PHO_gradient'], 0)
        self.assertEqual(int(arrays['fiber_repair_mask'][0]), 1)

    def test_no_valid_gradient_in_lv_cannot_borrow_an_rv_direction(self):
        volume, z, rho = self.repair_fixture()
        rho[:8] = .5
        with self.assertRaisesRegex(RuntimeError, 'LV cannot build a coordinate frame'):
            uvc.fiber_openbase(volume, z, rho)

    def test_working_conversion_detects_tag_changes_without_geometry_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = two_chambers()
            source = root / 'source.vtk'
            volume.save(source)
            def convert(_tool, input_base, output_base, **_kwargs):
                write_carp(pv.read(Path(input_base).with_suffix('.vtk')), output_base)
                return True
            with patch.object(driver.meshtool, 'convert_mesh', side_effect=convert) as mocked:
                _, first = driver.prepare_working_mesh(source, root, 'mesh', '/unused')
                volume.cell_data['elemTag'] = np.array([0, 1], dtype=np.int32)
                volume.save(source)
                _, second = driver.prepare_working_mesh(source, root, 'mesh', '/unused')
                self.assertTrue(first and second)
                self.assertEqual(mocked.call_count, 2)

    def test_working_conversion_accepts_local_tetrahedron_orientation_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            volume = two_chambers()
            source = root / 'source.vtk'
            volume.save(source)
            def convert(_tool, input_base, output_base, **_kwargs):
                write_carp(pv.read(Path(input_base).with_suffix('.vtk')), output_base)
                rows = np.asarray(driver.meshIO.read_elems(basename=output_base), dtype=int)
                rows[0, [0, 1]] = rows[0, [1, 0]]
                np.savetxt(Path(output_base).with_suffix('.elem'), rows,
                           header=str(volume.n_cells), comments='', fmt='Tt %d %d %d %d %d')
                return True
            with patch.object(driver.meshtool, 'convert_mesh', side_effect=convert) as mocked:
                _, first = driver.prepare_working_mesh(source, root, 'mesh', '/unused')
                _, second = driver.prepare_working_mesh(source, root, 'mesh', '/unused')
                self.assertTrue(first)
                self.assertFalse(second)
                self.assertEqual(mocked.call_count, 1)

    def test_fiber_cache_invalidates_when_pho_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            subject = data / 'subject_11'
            inputs = driver.pipeline_frame_paths(subject, 'ED')
            inputs['volume'].parent.mkdir(parents=True)
            volume = two_chambers()
            volume.save(inputs['volume'])
            frame = subject / 'UVC_fiber_Outputs/ED'
            uvc_dir = frame / 'UVC_i'
            uvc_dir.mkdir(parents=True)
            def field(name, values):
                np.savetxt(uvc_dir / (name + '.dat'), values, fmt='%.17g')
                mesh = volume.copy()
                mesh.point_data[name] = values
                mesh.save(uvc_dir / (name + '.vtk'))
            field('COORDS_Z', .5 * (volume.points[:, 2] + 1))
            field('PHI', np.zeros(volume.n_points))
            field('PHO', volume.points[:, 0])
            base = frame / 'subject_11_mesh_all'
            write_carp(volume, base)
            with patch.object(uvc, 'fiber_openbase', wraps=uvc.fiber_openbase) as generated:
                first = driver.run_frame_fiber('subject_11', 'ED', data)
                initial_lon = np.loadtxt(Path(first['fiber_mesh']).with_suffix('.lon'), skiprows=1)
                cached = driver.run_frame_fiber('subject_11', 'ED', data)
                self.assertTrue(cached['skipped'])
                field('PHO', .5 * volume.points[:, 0])
                changed = driver.run_frame_fiber('subject_11', 'ED', data)
                self.assertNotEqual(first['input_sha256']['PHO'], changed['input_sha256']['PHO'])
                updated_lon = np.loadtxt(Path(changed['fiber_mesh']).with_suffix('.lon'), skiprows=1)
                self.assertFalse(np.allclose(initial_lon, updated_lon))
                self.assertEqual(generated.call_count, 2)
                final = pv.read(changed['fiber_mesh'])
                np.testing.assert_array_equal(final.points, volume.points)
                np.testing.assert_array_equal(final.cells, volume.cells)
                np.testing.assert_array_equal(final.cell_data['elemTag'], volume.cell_data['elemTag'])
                for name in ('COORDS_Z', 'PHI', 'PHO', 'biv'):
                    self.assertIn(name, final.point_data)
                self.assertEqual(set(final.point_data['biv']), {-1, 1})


if __name__ == '__main__':
    unittest.main()
