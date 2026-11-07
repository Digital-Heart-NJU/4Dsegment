"""Build a synthetic RV wall in XY slices, without replacing nonzero labels."""
import argparse
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def checked_labels(data, allowed):
    labels = np.asarray(data)
    if not np.isfinite(labels).all() or not np.isin(labels, allowed).all():
        raise ValueError('Unexpected, noninteger, or nonfinite segmentation labels.')
    return labels.astype(np.int16, copy=True)


def remap_nnunet_labels(data):
    """Native label 3 is RV blood, not RV myocardium."""
    native = checked_labels(data, (0, 1, 2, 3))
    mapped = native.copy()
    mapped[native == 3] = 4
    return mapped


def expand_rv_myo_xy(data, affine, radius_mm=3.0, spatial_unit='mm', assume_mm=False):
    labels = checked_labels(data, (0, 1, 2, 3, 4))
    if labels.ndim != 3:
        raise ValueError('XY expansion requires a 3D label array.')
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError('The radius in millimeters must be positive and finite.')
    matrix = np.asarray(affine, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError('Invalid spatial affine.')
    units = {'mm': 1.0, 'meter': 1000.0, 'micron': 0.001}
    assumed = spatial_unit == 'unknown' and assume_mm
    if spatial_unit not in units and not assumed:
        raise ValueError('Spatial units are unknown; explicitly allow a millimeter assumption if appropriate.')
    basis = matrix[:3, :2] * units.get(spatial_unit, 1.0)
    spacing = np.linalg.norm(basis, axis=0)
    orthogonal = abs(np.dot(basis[:, 0], basis[:, 1])) <= 1e-7 * np.prod(spacing)
    output = labels.copy()
    added = np.zeros(labels.shape, dtype=bool)
    structure = ndimage.generate_binary_structure(2, 1)
    records = []
    for z in range(labels.shape[2]):
        plane = labels[:, :, z]
        rv = plane == 4
        if not rv.any():
            records.append({'z': z, 'RV_voxels': 0, 'added_voxels': 0, 'skipped': True})
            continue
        if orthogonal:
            distances = ndimage.distance_transform_edt(~rv, sampling=spacing)
        else:
            # Shear invalidates diagonal EDT sampling; use the actual in-plane
            # affine basis to measure distances between voxel centers instead.
            source = np.argwhere(rv) @ basis.T
            targets = np.argwhere(plane == 0)
            distances = np.full(plane.shape, np.inf)
            if len(targets):
                values = cKDTree(source).query(targets @ basis.T)[0]
                distances[tuple(targets.T)] = values
            distances[rv] = 0.0
        candidate = (plane == 0) & (distances > 0) & (distances <= radius_mm + 1e-8)
        connected = ndimage.binary_propagation(rv, structure=structure,
                                              mask=rv | candidate | (plane == 3))
        new = candidate & connected
        output[:, :, z][new] = 3
        added[:, :, z] = new
        border = new | rv
        records.append({'z': z, 'RV_voxels': int(rv.sum()), 'added_voxels': int(new.sum()),
                        'disconnected_candidates_removed': int((candidate & ~connected).sum()),
                        'max_added_center_distance_mm': float(distances[new].max()) if new.any() else None,
                        'growth_or_RV_touches_image_edge': bool(border[0].any() or border[-1].any() or
                                                               border[:, 0].any() or border[:, -1].any())})
    protected = labels != 0
    if not np.array_equal(output[protected], labels[protected]):
        raise RuntimeError('Expansion changed an existing anatomical label.')
    report = {'method': 'XY_only_background_growth_from_RV_blood_voxel_centers',
              'radius_mm': float(radius_mm), 'inplane_spacing_mm': spacing.tolist(),
              'distance_method': '2D_spacing_aware_EDT' if orthogonal else 'inplane_world_coordinate_KDTree',
              'input_spatial_unit': spatial_unit, 'assumed_mm': assumed,
              'added_voxels': int(added.sum()), 'protected_labels_unchanged': True,
              'no_Z_growth': True, 'slices': records,
              'label_counts': {str(int(v)): int(n) for v, n in zip(*np.unique(output, return_counts=True))},
              'limitation': 'Voxel-center dilation radius, not a certified continuous surface thickness; obstacles and image edges limit growth.'}
    return output, added, report


def label_image(labels, reference):
    image = nib.Nifti1Image(np.asarray(labels, dtype=np.int16), reference.affine, header=reference.header.copy())
    image.set_qform(reference.get_qform(), code=int(reference.header['qform_code']))
    image.set_sform(reference.get_sform(), code=int(reference.header['sform_code']))
    if not int(reference.header['qform_code']) and not int(reference.header['sform_code']):
        image.set_sform(reference.affine, code=2)
    image.header.set_data_dtype(np.int16)
    image.header.set_slope_inter(1, 0)
    return image


def save_labels(path, labels, reference, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.labels-', dir=path.parent) as tmp:
        candidate = Path(tmp) / path.name
        nib.save(label_image(labels, reference), candidate)
        saved = nib.load(candidate)
        if not np.array_equal(np.asarray(saved.dataobj), labels) or not np.allclose(saved.affine, reference.affine, atol=1e-5):
            raise RuntimeError('Saved labels or geometry changed unexpectedly.')
        candidate.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Already-remapped DMACS labels, with RV blood=4.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--radius-mm', default=3.0, type=float)
    parser.add_argument('--assume-mm-if-unknown', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('The input file must not be overwritten.')
    source = nib.load(args.input)
    data = np.asarray(source.dataobj)
    singleton = data.ndim == 4 and data.shape[-1] == 1
    output, added, report = expand_rv_myo_xy(data[..., 0] if singleton else data, source.affine,
                                           args.radius_mm, source.header.get_xyzt_units()[0],
                                           args.assume_mm_if_unknown)
    save_labels(args.output, output[..., None] if singleton else output, source, args.overwrite)
    summary = args.output.with_name(args.output.name + '.json')
    summary.write_text(json.dumps(report, indent=2) + '\n')
    print('Saved:', args.output, 'added RV myocardium voxels:', int(added.sum()))


if __name__ == '__main__':
    main()
