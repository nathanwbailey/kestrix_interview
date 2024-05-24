"""Main file for kestrix interview code."""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pdal

from utils import find_planes_ransac, remove_outliers_from_pcd

# Input the mesh file
mesh = o3d.io.read_triangle_mesh("property.ply", print_progress=True)
# Print some summary information about the mesh: vertices, color, faces
print("Vertices:")
print(np.asarray(mesh.vertices))
print("Faces:")
print(np.asarray(mesh.triangles))
print("Colors:")
print(np.asarray(mesh.vertex_colors))
bounding_box_mesh = mesh.get_axis_aligned_bounding_box()
print(f"Min bound: {bounding_box_mesh.min_bound}")
print(f"Max bound: {bounding_box_mesh.max_bound}")

# Convert the mesh to a point cloud
pcd = mesh.sample_points_uniformly(number_of_points=25000)
pcd = mesh.sample_points_poisson_disk(number_of_points=10000, pcl=pcd)

# Print some summary information about the PCD
bounding_box = pcd.get_axis_aligned_bounding_box()
print(f"Min bound: {bounding_box.min_bound}")
print(f"Max bound: {bounding_box.max_bound}")
centroid = pcd.get_center()
print(f"Centroid: {centroid}")

# Preprocess the point cloud
# Downsample into voxels
VOXEL_SIZE = 0.005
pcd_down = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
# Recompute normals, use a radius neighbourhood of 2 voxels and 30 NN
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=VOXEL_SIZE * 2.0, max_nn=30
    )
)
# Remove outliers
pcd_down, _ = remove_outliers_from_pcd(pcd_down, nb_neighbours=20, std_ratio=2.0)

# Get the centroid after the pcd has been processed
pre_processed_centroid = pcd_down.get_center()
print(f"Pre Processed Centroid: {pre_processed_centroid}")

# Get the roof planes from RANSAC
# A valid roof plane must have a centroid greater than 2M from the pre_processed_centroid
# It must also must be greater than 10m2 and less than 25m2
roof_planes, roof_plane_convex_hull_points, remaining_points = find_planes_ransac(
    pcd=pcd_down,
    min_area=10,
    max_area=25,
    plane_type="roof",
    centroid_to_compare=pre_processed_centroid,
    centroid_threshold=2,
    ransac_distance_threshold=0.08,
    ransac_number=3,
    ransac_num_iterations=5000,
    nb_neighbours=30,
    std_ratio=0.7,
)

# Process the PCD ready to detect wall planes
# We remove the roof planes from the PCD such that they do not get picked up in the upcoming RANSAC process
remaining_points = deepcopy(pcd_down)
remaining_points_numpy = np.asarray(remaining_points.points)
for roof_plane in roof_planes:
    roof_point = np.asarray(roof_plane.points)
    roof_point_set = set([tuple(x) for x in roof_point])
    remaining_points_set = set([tuple(x) for x in remaining_points_numpy])
    remaining_points_numpy = np.array(list(remaining_points_set - roof_point_set))

# Construct the PCD from the remaining points
remaining_points = o3d.geometry.PointCloud()
remaining_points.points = o3d.utility.Vector3dVector(remaining_points_numpy)
remaining_points.paint_uniform_color([0, 1, 0])

# Save file for PDAL
o3d.io.write_point_cloud("pdal_point_cloud_input.ply", remaining_points)

# PDAL pipeline to remove the ground points
pipeline = """
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
"""
# Execute the PDAL pipeline and obtain the output point cloud
pipeline_pdal = pdal.Pipeline(pipeline)
pipeline_pdal.execute()
remaining_points = o3d.io.read_point_cloud(
    "pdal_point_cloud_output.ply", print_progress=True
)
# Remove outliers from the PCD
remaining_points, _ = remove_outliers_from_pcd(
    pcd=remaining_points, nb_neighbours=20, std_ratio=1.0
)


# Get the wall planes from RANSAC
# A valid wall plane must be greater than 10m2 and less than 30m2
_, facade_plane_convex_hull_points, _ = find_planes_ransac(
    pcd=remaining_points,
    min_area=10,
    max_area=30,
    plane_type="wall",
    ransac_distance_threshold=0.15,
    ransac_number=3,
    ransac_num_iterations=5000,
    nb_neighbours=70,
    std_ratio=0.7,
    cluster_exclude_num=1,
)


# Bonus Exercise
# Plane outline extraction
# We can take the convex hull found above which finds the smallest convex polygon encompassing all the points
# Scatter these points in a graph and connect the points to obtain the outline of the planes
convex_hull_dict = {
    "roof": roof_plane_convex_hull_points,
    "wall": facade_plane_convex_hull_points,
}
for key, value in convex_hull_dict.items():
    os.makedirs(key + "s", exist_ok=True)
    for idx, convex_hull_plane_points in enumerate(value):
        # Extract X and Y points
        roof_plane_x_points = [points[0] for points in convex_hull_plane_points]
        roof_plane_y_points = [points[1] for points in convex_hull_plane_points]
        # Add the first point to fully connect the points
        roof_plane_x_points += [roof_plane_x_points[0]]
        roof_plane_y_points += [roof_plane_y_points[0]]
        # Scatter points and plot the line through the points
        # Scatter points stay above the line
        plt.scatter(roof_plane_x_points, roof_plane_y_points, zorder=2)
        plt.plot(roof_plane_x_points, roof_plane_y_points, "b-", zorder=1)
        plane_number = idx
        plane_type = key
        plt.savefig(
            key + "s/" + str(plane_type) + "_outline_" + str(plane_number) + ".png"
        )
        plt.clf()
