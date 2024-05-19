"""Main file for kestrix interview code."""
from copy import deepcopy
from random import randint
import open3d as o3d
import numpy as np
import pdal
from scipy.spatial import ConvexHull


#Input the mesh file and convert it to a point cloud using open3D
mesh = o3d.io.read_triangle_mesh('property.ply', print_progress=True)
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Faces:')
print(np.asarray(mesh.triangles))
print('Colors:')
print(np.asarray(mesh.vertex_colors))
bounding_box_mesh = mesh.get_axis_aligned_bounding_box()
print(f"Min bound: {bounding_box_mesh.min_bound}")
print(f"Max bound: {bounding_box_mesh.max_bound}")
pcd = mesh.sample_points_uniformly(number_of_points=25000)
pcd = mesh.sample_points_poisson_disk(number_of_points=10000, pcl=pcd)
# o3d.visualization.draw_geometries([pcd])
#Print some summary information about the PCD
bounding_box = pcd.get_axis_aligned_bounding_box()
print(f"Min bound: {bounding_box.min_bound}")
print(f"Max bound: {bounding_box.max_bound}")
centroid = pcd.get_center()
print(f"Centroid: {centroid}")


#Preprocess the point cloud
#Downsample into voxels
voxel_size=0.003
pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
#Recompute normals, use a radius neighbourhood of 2 voxels and 30 NN
pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2.0, max_nn=30))
#Remove some outliers
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_down = pcd_down.select_by_index(ind)
pcd_down = pcd_down.remove_non_finite_points()
#DBSCAN
#Compute fpfh
# fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,max_nn=100))

#Save file for PDAL
o3d.io.write_point_cloud('pdal_point_cloud_input.ply', pcd_down)

#PDAL pipeline to remove the ground points
pipeline = '''
[
    {
        "type": "readers.ply",
        "filename": "pdal_point_cloud_input.ply"
    },
    {
        "type":"filters.csf"
    },
    {
        "type":"filters.range",
        "limits":"Classification![2:2]"
    },
    {
        "type": "writers.ply",
        "filename": "pdal_point_cloud_output.ply"
    }
]
'''

pipeline_pdal = pdal.Pipeline(pipeline)
pipeline_pdal.execute()

# pcd_down = o3d.io.read_point_cloud('pdal_point_cloud_output.ply', print_progress=True)
# o3d.visualization.draw_geometries([point_cloud])

pre_processed_centroid = pcd_down.get_center()
print(f"Pre Processed Centroid: {pre_processed_centroid}")

def project_point_to_plane(normal_vector: np.ndarray, d_value: np.float64, point_to_project: np.ndarray) -> np.ndarray:
    """Project a 3D point onto a plane."""
    k_value = (d_value - np.dot(point_to_project, normal_vector))/np.sum(np.square(normal_vector))
    point_projected = point_to_project + k_value*normal_vector
    return point_projected

def choose_point_in_plane(plane_equation: np.ndarray, x_val: int | float, y_val: int | float) -> np.ndarray:
    """Find a point in a given plane defined by a plane equation."""
    normal_vector = plane_equation[:3]
    cross = np.asarray([0,0,0])
    point_to_return = np.asarray([0,0,0])
    while np.all(cross == 0):
        x = x_val
        y = y_val
        z = (plane_equation[-1] - x*plane_equation[0] - y*plane_equation[1])/plane_equation[2]
        point_to_return = np.asarray([x,y,z])
        #Check for non-colinear
        cross = np.cross(normal_vector, point_to_return)
    return point_to_return

def obtain_orthonormal_basis(plane_equation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a plane equation obtain an orthonormal_basis."""
    # Choose starting x and y values so that the 2 points are linearly independent
    v1 = choose_point_in_plane(plane_equation, 1, 0)
    v2 = choose_point_in_plane(plane_equation, 0, 1)
    # Compute the orthonormal basis using the gram-schmidt process (https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process)
    u1 = v1
    proj_u1_v2 = (np.dot(v2, u1)*u1)/np.dot(u1,u1)
    u2 = v2 - proj_u1_v2
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)

    # Norm (length) should be one, but check against very close value due to fp errors
    assert np.linalg.norm(u1) >= 0.99999999999999
    assert np.linalg.norm(u2) >= 0.99999999999999
    # Dot product should be zero, but check against very small value due to fp errors
    assert np.dot(u1, u2) < 1e-10
    return u1, u2



def transform_3d_point_to_2d(point_to_transform: np.ndarray, vector_v1: np.ndarray, vector_v2: np.ndarray) -> np.ndarray:
    """Given an orthonormal_basis, transform a 3D point to a 2D point."""
    return np.array([np.dot(point_to_transform, vector_v1), np.dot(point_to_transform, vector_v2)])

def is_valid_roof_plane(plane: o3d.geometry.PointCloud, centroid_to_compare: np.ndarray, plane_eq: np.ndarray, min_area: int = 10, max_area: int = 25, centroid_threshold: int = 2) -> bool:
    """Given a plane, determine if it is valid, given parameters."""
    plane_centroid = plane.get_center().tolist()[2]
    compare_centroid = centroid_to_compare.tolist()[2]
    if plane_centroid-compare_centroid < centroid_threshold:
        return False
    # convex_hull, _ = plane.compute_convex_hull()
    # print(convex_hull.get_surface_area())
    # # o3d.visualization.draw_geometries([convex_hull])

    plane_points = np.asarray(plane.points)
    v1, v2 = obtain_orthonormal_basis(plane_eq)
    transformed_points = []
    for point in plane_points:
        point_projected = project_point_to_plane(plane_eq[:3], plane_eq[3], point)
        point_2d = transform_3d_point_to_2d(point_projected, v1, v2)
        transformed_points.append(point_2d)

    transformed_points = np.stack(transformed_points, axis=0)

    plane_area = ConvexHull(transformed_points).area
    print(plane_area)
    if plane_area <= min_area or plane_area >= max_area:
        return False
    return True

#Extract the planes using RANSAC
remaining_points = deepcopy(pcd_down)
pcd_down_copy = deepcopy(pcd_down)
planes = []
for _ in range(10):
    plane_eq, inliners = remaining_points.segment_plane(distance_threshold=0.07, ransac_n=3, num_iterations=1000)
    plane = remaining_points.select_by_index(inliners)

    """
    A valid plane must have a centroid greater than 2M from the pre_processed_centroid.
    It must also must be greater than 10m2 and less than 25m2
    """
    planes.append(plane)
    remaining_points = remaining_points.select_by_index(inliners, invert=True)

    if is_valid_roof_plane(plane, pre_processed_centroid, plane_eq):
        plane.paint_uniform_color([1,0,0])
        remaining_points_plane = pcd_down_copy.select_by_index(inliners, invert=True)
        remaining_points.paint_uniform_color([0,1,0])
        #o3d.visualization.draw_geometries([plane, remaining_points])

