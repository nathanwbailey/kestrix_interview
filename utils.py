"""Functions to use in the main file."""

import os
from copy import deepcopy
from random import randint
from typing import Literal

import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull


def project_point_to_plane(
    normal_vector: np.ndarray, d_value: np.float64, point_to_project: np.ndarray
) -> np.ndarray:
    """Project a 3D point onto a plane."""
    k_value = (d_value - np.dot(point_to_project, normal_vector)) / np.dot(
        normal_vector, normal_vector
    )
    point_projected = point_to_project + k_value * normal_vector
    return point_projected


def choose_point_in_plane(
    plane_equation: np.ndarray, x_val: int | float, y_val: int | float
) -> np.ndarray:
    """Find a point in a given plane defined by a plane equation."""
    normal_vector = plane_equation[:3]
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    cross = np.asarray([0, 0, 0])
    point_to_return = np.asarray([0, 0, 0])
    while np.all(cross == 0):
        x = x_val
        y = y_val
        z = (
            plane_equation[-1] - x * plane_equation[0] - y * plane_equation[1]
        ) / plane_equation[2]
        point_to_return = np.asarray([x, y, z])
        # Check for non-colinear with the normal vector of the plane
        cross = np.cross(normal_vector, point_to_return)
    return point_to_return


def obtain_orthonormal_basis_gram_schmidt(
    plane_equation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a plane equation obtain an orthonormal basis using the gram-schmidt process."""
    # Choose starting x and y values so that the 2 points are linearly independent
    v1 = choose_point_in_plane(plane_equation, 0, 1)
    v2 = choose_point_in_plane(plane_equation, 1, 0)
    # Compute the orthonormal basis using the gram-schmidt process (https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
    u1 = v1
    u1 = u1 / np.linalg.norm(u1)
    proj_u1_v2 = np.dot(v2, u1) * u1
    u2 = v2 - proj_u1_v2
    u2 = u2 / np.linalg.norm(u2)
    # Norm (length) should be one, but check against very close value due to fp errors
    assert np.linalg.norm(u1) >= 0.99999999999999
    assert np.linalg.norm(u2) >= 0.99999999999999
    # Dot product should be zero, but check against very small value due to fp errors
    assert np.dot(u1, u2) < 1e-10

    return u1, u2


def obtain_orthonormal_basis(
    plane_equation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a plane equation obtain an orthonormal basis."""
    normal_vector = plane_equation[:3]
    cross = np.asarray([0, 0, 0])
    v1 = np.asarray([0, 0, 0])
    # Find a valid point on the plane that is non-colinear with the normal vector
    while np.all(cross == 0):
        x = randint(1, 5)
        y = randint(1, 5)
        z = (
            plane_equation[-1] - x * plane_equation[0] - y * plane_equation[1]
        ) / plane_equation[2]
        v1 = np.asarray([x, y, z])
        # Check for non-colinear
        cross = np.cross(normal_vector, v1)
    # Find the orthonormal basis
    v2 = np.cross(normal_vector, v1)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    # Norm (length) should be one, but check against very close value due to fp errors
    assert np.linalg.norm(v1) >= 0.99999999999999
    assert np.linalg.norm(v2) >= 0.99999999999999
    # Dot product should be zero, but check against very small value due to fp errors
    assert np.dot(v1, v2) < 1e-10
    return v1, v2


def transform_3d_point_to_2d(
    point_to_transform: np.ndarray, vector_v1: np.ndarray, vector_v2: np.ndarray
) -> np.ndarray:
    """Given an orthonormal_basis, transform a 3D point to a 2D point."""
    return np.array(
        [np.dot(point_to_transform, vector_v1), np.dot(point_to_transform, vector_v2)]
    )


def is_valid_plane(
    plane_to_validate: o3d.geometry.PointCloud,
    plane_to_validate_equation: np.ndarray,
    centroid_to_compare: np.ndarray | None = None,
    min_area: int = 6,
    max_area: int = 25,
    centroid_threshold: int = 2,
) -> tuple[bool, np.ndarray | None]:
    """Given a plane, determine if it is valid, given parameters."""
    # Compare the plane to a centroid, if needed
    if centroid_to_compare is not None:
        plane_centroid = plane_to_validate.get_center().tolist()[2]
        compare_centroid = centroid_to_compare.tolist()[2]
        if plane_centroid - compare_centroid < centroid_threshold:
            return False, None
    # Transform the points onto the plane, and then onto 2D space
    plane_points = np.asarray(plane_to_validate.points)
    v1, v2 = obtain_orthonormal_basis(plane_to_validate_equation)
    transformed_points = []
    for point in plane_points:
        # Transform the points onto the plane
        point_projected = project_point_to_plane(
            plane_to_validate_equation[:3], plane_to_validate_equation[-1], point
        )
        # Transform the points from 3D to 2D
        point_2d = transform_3d_point_to_2d(point_projected, v1, v2)
        transformed_points.append(point_2d)
    # Get the points as a numpy array
    transformed_points = np.stack(transformed_points, axis=0)
    # Find the convex_hull of the 2D points
    convex_hull = ConvexHull(transformed_points)
    # Get the area of the convex hull
    # 2D co-ordinates, so use the volume which gives the area
    plane_area = convex_hull.volume
    # Find the points of the convex hull (perimeter points)
    convex_hull_points = transformed_points[convex_hull.vertices]
    # Reject or accept the plane based on area requirements
    if plane_area <= min_area or plane_area >= max_area:
        return False, None
    print(f"Found valid plane with area: {plane_area}")
    return True, convex_hull_points


def save_plane_to_file(
    plane_to_save: o3d.geometry.PointCloud,
    plane_number: int,
    plane_type: Literal["roof", "wall"],
    save_as_point_cloud: bool = True,
    save_as_mesh: bool = True,
) -> None:
    """Save a point cloud to file, can also save it as a mesh too using the complex hull."""
    os.makedirs(plane_type + "s", exist_ok=True)
    if save_as_point_cloud:
        o3d.io.write_point_cloud(
            str(plane_type)
            + "s/"
            + str(plane_type)
            + "_plane_"
            + str(plane_number)
            + ".ply",
            plane_to_save,
        )
    if save_as_mesh:
        mesh_to_save, _ = plane_to_save.compute_convex_hull()
        o3d.io.write_triangle_mesh(
            str(plane_type)
            + "s/"
            + str(plane_type)
            + "_plane_"
            + str(plane_number)
            + "_as_mesh.ply",
            mesh_to_save,
        )


def remove_outliers_from_pcd(
    pcd: o3d.geometry.PointCloud, nb_neighbours: int = 20, std_ratio: float = 2.0
) -> o3d.geometry.PointCloud:
    """Given a PCD file, remove statistical outliers and non finite points."""
    # Remove statistical outliers
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbours, std_ratio=std_ratio
    )
    pcd_to_return = pcd.select_by_index(ind)
    removed_points = pcd.select_by_index(ind, invert=True)
    # Remove non-finite points
    pcd_to_return = pcd_to_return.remove_non_finite_points()
    return pcd_to_return, np.asarray(removed_points.points)


def find_planes_ransac(
    pcd: o3d.geometry.PointCloud,
    min_area: int,
    max_area: int,
    plane_type: Literal["roof", "wall"],
    centroid_to_compare: np.ndarray | None = None,
    centroid_threshold: int = 2,
    ransac_distance_threshold: float = 0.08,
    ransac_number: int = 3,
    ransac_num_iterations: int = 5000,
    min_points_for_plane: int = 700,
    nb_neighbours: int = 30,
    std_ratio: float = 0.7,
    cluster_exclude_num: int | None = None,
) -> tuple[list[o3d.geometry.PointCloud], list[np.ndarray], o3d.geometry.PointCloud]:
    """Find valid planes using the RANSAC Algorithm."""
    remaining_points = deepcopy(pcd)
    num = 0
    convex_hull_points = []
    planes = []
    # Find planes until we reach a number of points threshold
    while True:
        # Get the plane from RANSAC
        plane_eq, inliners = remaining_points.segment_plane(
            distance_threshold=ransac_distance_threshold,
            ransac_n=ransac_number,
            num_iterations=ransac_num_iterations,
        )
        # If the number of points found in the plane is less than the specified threshold, break finding planes
        if len(inliners) < min_points_for_plane:
            break
        # Extract the plane from the point cloud
        plane = remaining_points.select_by_index(inliners)
        # Remove the plane from the overall point cloud
        remaining_points = remaining_points.select_by_index(inliners, invert=True)
        # Pre process the plane to remove outilers
        plane, removed_points = remove_outliers_from_pcd(
            plane, nb_neighbours=nb_neighbours, std_ratio=std_ratio
        )
        # Add the removed points back to the overall point cloud
        remaining_points_numpy = np.concatenate(
            (np.asarray(remaining_points.points), removed_points), axis=0
        )
        remaining_points = o3d.geometry.PointCloud()
        remaining_points.points = o3d.utility.Vector3dVector(remaining_points_numpy)

        # Optionally we can exclude planes if they have greater than a certain number of clusters
        if cluster_exclude_num:
            if (
                np.unique(
                    np.asarray(plane.cluster_dbscan(eps=0.5, min_points=3))
                ).shape[0]
                > cluster_exclude_num
            ):
                continue

        # Valid planes are compared to area bounds and optionally a centroid
        if centroid_to_compare is not None:
            valid_plane = is_valid_plane(
                plane_to_validate=plane,
                plane_to_validate_equation=plane_eq,
                centroid_to_compare=centroid_to_compare,
                min_area=min_area,
                max_area=max_area,
                centroid_threshold=centroid_threshold,
            )
        else:
            valid_plane = is_valid_plane(
                plane_to_validate=plane,
                plane_to_validate_equation=plane_eq,
                min_area=min_area,
                max_area=max_area,
            )

        if valid_plane[0] and valid_plane[1] is not None:
            planes.append(plane)
            convex_hull_points.append(valid_plane[1])
            plane.paint_uniform_color([1, 0, 0])
            # Save the plane to a PLY file if valid
            save_plane_to_file(
                plane_to_save=plane,
                plane_number=num,
                plane_type=plane_type,
            )
            num += 1
            # Visualize the plane in the point cloud
            # remaining_points.paint_uniform_color([0, 1, 0])
            # o3d.visualization.draw_geometries([plane, remaining_points])

    return planes, convex_hull_points, remaining_points
