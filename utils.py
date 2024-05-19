"""Functions to use in the main file."""
import numpy as np
from random import randint
from typing import Literal
from scipy.spatial import ConvexHull
import open3d as o3d

def project_point_to_plane(normal_vector: np.ndarray, d_value: np.float64, point_to_project: np.ndarray) -> np.ndarray:
    """Project a 3D point onto a plane."""
    k_value = (d_value - np.dot(point_to_project, normal_vector))/np.dot(normal_vector, normal_vector)
    point_projected = point_to_project + k_value*normal_vector
    return point_projected

def choose_point_in_plane(plane_equation: np.ndarray, x_val: int | float, y_val: int | float) -> np.ndarray:
    """Find a point in a given plane defined by a plane equation."""
    normal_vector = plane_equation[:3]
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    cross = np.asarray([0,0,0])
    point_to_return = np.asarray([0,0,0])
    while np.all(cross == 0):
        x = x_val
        y = y_val
        z = (plane_equation[-1] - x*plane_equation[0] - y*plane_equation[1])/plane_equation[2]
        point_to_return = np.asarray([x,y,z])
        #Check for non-colinear with the normal vector of the plane
        cross = np.cross(normal_vector, point_to_return)
    return point_to_return

def obtain_orthonormal_basis_gram_schmidt(plane_equation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a plane equation obtain an orthonormal basis using the gram-schmidt process."""
    # Choose starting x and y values so that the 2 points are linearly independent
    v1 = choose_point_in_plane(plane_equation,0,1)
    v2 = choose_point_in_plane(plane_equation,1,0)
    # Compute the orthonormal basis using the gram-schmidt process (https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
    u1 = v1
    u1 = u1 / np.linalg.norm(u1)
    proj_u1_v2 = (np.dot(v2, u1)*u1)
    u2 = v2 - proj_u1_v2
    u2 = u2 / np.linalg.norm(u2)
    # Norm (length) should be one, but check against very close value due to fp errors
    assert np.linalg.norm(u1) >= 0.99999999999999
    assert np.linalg.norm(u2) >= 0.99999999999999
    # Dot product should be zero, but check against very small value due to fp errors
    assert np.dot(u1, u2) < 1e-10

    return u1, u2

def obtain_orthonormal_basis(plane_equation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a plane equation obtain an orthonormal basis."""
    normal_vector = plane_equation[:3]
    cross = np.asarray([0,0,0])
    v1 = np.asarray([0,0,0])
    #Find a valid point on the plane that is non-colinear with the normal vector
    while np.all(cross == 0):
        x = randint(1, 5)
        y = randint(1, 5)
        z = (plane_equation[-1] - x*plane_equation[0] - y*plane_equation[1])/plane_equation[2]
        v1 = np.asarray([x,y,z])
        #Check for non-colinear
        cross = np.cross(normal_vector, v1)
    #Find the orthonormal basis
    v2 = np.cross(normal_vector, v1)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    # Norm (length) should be one, but check against very close value due to fp errors
    assert np.linalg.norm(v1) >= 0.99999999999999
    assert np.linalg.norm(v2) >= 0.99999999999999
    # Dot product should be zero, but check against very small value due to fp errors
    assert np.dot(v1, v2) < 1e-10
    return v1, v2

def transform_3d_point_to_2d(point_to_transform: np.ndarray, vector_v1: np.ndarray, vector_v2: np.ndarray) -> np.ndarray:
    """Given an orthonormal_basis, transform a 3D point to a 2D point."""
    return np.array([np.dot(point_to_transform, vector_v1), np.dot(point_to_transform, vector_v2)])

def is_valid_plane(plane_to_validate: o3d.geometry.PointCloud, plane_to_validate_equation: np.ndarray, centroid_to_compare: np.ndarray | None = None,  min_area: int = 6, max_area: int = 25, centroid_threshold: int = 2) -> tuple[bool, np.ndarray | None]:
    """Given a plane, determine if it is valid, given parameters."""
    #Compare the plane to a centroid, if needed
    if centroid_to_compare is not None:
        plane_centroid = plane_to_validate.get_center().tolist()[2]
        compare_centroid = centroid_to_compare.tolist()[2]
        if plane_centroid-compare_centroid < centroid_threshold:
            return False, None
    #Transform the points onto the plane, and then onto 2D space
    plane_points = np.asarray(plane_to_validate.points)
    v1, v2 = obtain_orthonormal_basis(plane_to_validate_equation)
    transformed_points = []
    for point in plane_points:
        #Transform the points onto the plane
        point_projected = project_point_to_plane(plane_to_validate_equation[:3], plane_to_validate_equation[-1], point)
        #Transform the points from 3D to 2D
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
    # Reject ior accept the plane based on area requirements
    if plane_area <= min_area or plane_area >= max_area:
        return False, None
    return True, convex_hull_points

def save_plane_to_file(plane_to_save: o3d.geometry.PointCloud, plane_number: int, plane_type: Literal['roof', 'wall'], save_as_point_cloud: bool = True, save_as_mesh: bool = True) -> None:
    """Save a point cloud to file, can also save it as a mesh too using the complex hull."""
    if save_as_point_cloud:
        o3d.io.write_point_cloud(str(plane_type)+'_plane_'+str(plane_number)+'.ply', plane_to_save)
    if save_as_mesh:
        mesh_to_save, _ = plane_to_save.compute_convex_hull()
        o3d.io.write_triangle_mesh(str(plane_type)+'_plane_'+str(plane_number)+'_as_mesh.ply', mesh_to_save)