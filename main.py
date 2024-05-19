"""Main file for kestrix interview code."""
from copy import deepcopy
from random import randint
from typing import Literal
import open3d as o3d
import numpy as np
import pdal
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


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
voxel_size=0.005
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
        #Check for non-colinear
        cross = np.cross(normal_vector, point_to_return)
    return point_to_return

def obtain_orthonormal_basis(plane_equation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given a plane equation obtain an orthonormal basis."""
    normal_vector = plane_equation[:3]
    cross = np.asarray([0,0,0])
    v1 = np.asarray([0,0,0])
    while np.all(cross == 0):
        x = randint(1, 5)
        y = randint(1, 5)
        z = (plane_equation[-1] - x*plane_equation[0] - y*plane_equation[1])/plane_equation[2]
        v1 = np.asarray([x,y,z])
        #Check for non-colinear
        cross = np.cross(normal_vector, v1)
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
    if centroid_to_compare is not None:
        plane_centroid = plane_to_validate.get_center().tolist()[2]
        compare_centroid = centroid_to_compare.tolist()[2]
        if plane_centroid-compare_centroid < centroid_threshold:
            return False, None
    plane_points = np.asarray(plane_to_validate.points)
    v1, v2 = obtain_orthonormal_basis(plane_to_validate_equation)
    transformed_points = []
    for point in plane_points:
        point_projected = project_point_to_plane(plane_to_validate_equation[:3], plane_to_validate_equation[-1], point)
        point_2d = transform_3d_point_to_2d(point_projected, v1, v2)
        transformed_points.append(point_2d)

    transformed_points = np.stack(transformed_points, axis=0)
    # 2D co-ordinates, so use the volume which gives the area
    convex_hull = ConvexHull(transformed_points)
    plane_area = convex_hull.volume
    convex_hull_points = transformed_points[convex_hull.vertices]
    if plane_area <= min_area or plane_area >= max_area:
        return False, None
    return True, convex_hull_points

def save_plane_to_file(plane_to_save: o3d.geometry.PointCloud, plane_number: int, plane_type: Literal['roof', 'wall'], save_as_point_cloud: bool = True, save_as_mesh: bool = True) -> None:
    """Save a point cloud to file, can also save it as a mesh too."""
    if save_as_point_cloud:
        o3d.io.write_point_cloud(str(plane_type)+'_plane_'+str(plane_number)+'.ply', plane_to_save)
    if save_as_mesh:
        mesh_to_save, _ = plane_to_save.compute_convex_hull()
        o3d.io.write_triangle_mesh(str(plane_type)+'_plane_'+str(plane_number)+'_as_mesh.ply', mesh_to_save)

#Extract the planes using RANSAC
remaining_points = deepcopy(pcd_down)
pcd_down_copy = deepcopy(pcd_down)
planes = []
roof_num = 0
facade_num = 0
for _ in range(10):
    plane_eq, inliners = remaining_points.segment_plane(distance_threshold=0.08, ransac_n=3, num_iterations=5000)
    plane = remaining_points.select_by_index(inliners)
    planes.append(plane)
    remaining_points = remaining_points.select_by_index(inliners, invert=True)

    # A valid roof plane must have a centroid greater than 2M from the pre_processed_centroid
    # It must also must be greater than 10m2 and less than 25m2
    valid_roof_plane = is_valid_plane(plane_to_validate=plane, plane_to_validate_equation=plane_eq, centroid_to_compare=pre_processed_centroid, min_area=10, max_area=25, centroid_threshold=2)
    if valid_roof_plane[0]:
        roof_num += 1

    # A valid facade must be greater than 10m2 and less than 30m2
    valid_facade_plane = (False, None)
    if not valid_roof_plane[0]:
        valid_facade_plane = is_valid_plane(plane_to_validate=plane, plane_to_validate_equation=plane_eq, min_area=10, max_area=30) # type: ignore[assignment]

    if valid_facade_plane[0]:
        facade_num += 1

    if valid_roof_plane[0] or valid_facade_plane[0]:
        plane.paint_uniform_color([1,0,0])
        save_plane_to_file(plane_to_save=plane, plane_number=(roof_num if valid_roof_plane[0] else facade_num), plane_type=('roof' if valid_roof_plane[0] else 'wall'))
        remaining_points_plane = pcd_down_copy.select_by_index(inliners, invert=True)
        remaining_points.paint_uniform_color([0,1,0])
        #o3d.visualization.draw_geometries([plane, remaining_points])

    # Bonus Exercise
    # Plane outline extraction
    # We can take the convex hull found above which finds the smallest convex polygon encompassing all the points
    # Scatter these points in a graph and connect the points to obtain the outline of the planes
    if valid_roof_plane[1] is not None or valid_facade_plane[1] is not None:
        convex_hull_plane_points = valid_roof_plane[1] if valid_roof_plane[1] is not None else valid_facade_plane[1]
        #Extract X and Y points
        roof_plane_x_points = [points[0] for points in convex_hull_plane_points]
        roof_plane_y_points = [points[1] for points in convex_hull_plane_points]
        #Add the first point to fully connect the points
        roof_plane_x_points += [roof_plane_x_points[0]]
        roof_plane_y_points += [roof_plane_y_points[0]]
        # Scatter points and plot the line through the points
        # Scatter points stay above the line
        plt.scatter(roof_plane_x_points, roof_plane_y_points, zorder=2)
        plt.plot(roof_plane_x_points, roof_plane_y_points, 'b-', zorder=1)
        plane_number=(roof_num if valid_roof_plane[0] else facade_num)
        plane_type=('roof' if valid_roof_plane[0] else 'wall')
        plt.savefig(str(plane_type)+'_outline_'+str(plane_number)+'.png')
        plt.clf()
