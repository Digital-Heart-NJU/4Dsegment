"""Confirm Open3D intersection candidates using double-precision triangles."""

import fcl
import numpy as np
import open3d as o3d


def confirmed_self_intersections(points, faces):
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points), o3d.utility.Vector3iVector(faces),
    )
    candidates = np.asarray(mesh.get_self_intersecting_triangles(), dtype=np.int64).reshape(-1, 2)
    objects = {}

    def triangle_object(index):
        if index not in objects:
            model = fcl.BVHModel()
            model.beginModel()
            model.addSubModel(points[faces[index]], np.array([[0, 1, 2]], dtype=np.int32))
            model.endModel()
            objects[index] = fcl.CollisionObject(model)
        return objects[index]

    confirmed = []
    for a, b in candidates:
        result = fcl.DistanceResult()
        distance = fcl.distance(triangle_object(int(a)), triangle_object(int(b)), fcl.DistanceRequest(), result)
        if not np.isfinite(distance):
            raise RuntimeError("Triangle intersection verification returned a non-finite distance.")
        if distance <= 0.0:
            confirmed.append((a, b))
    return np.asarray(confirmed, dtype=np.int64).reshape(-1, 2)
