"""Main file for kestrix interview code."""

from copy import deepcopy

import numpy as np
import open3d as o3d
import pdal


# Input the mesh file and convert it to a point cloud using open3D
mesh = o3d.io.read_triangle_mesh("property.ply", print_progress=True)
print(mesh)
print("Vertices:")
print(np.asarray(mesh.vertices))
print("Faces:")
print(np.asarray(mesh.triangles))
print("Colors:")
print(np.asarray(mesh.vertex_colors))

# o3d.visualization.draw_geometries([mesh])


bounding_box_mesh = mesh.get_axis_aligned_bounding_box()
print(f"Min bound: {bounding_box_mesh.min_bound}")
print(f"Max bound: {bounding_box_mesh.max_bound}")
pcd = mesh.sample_points_uniformly(number_of_points=25000)
pcd = mesh.sample_points_poisson_disk(number_of_points=10000, pcl=pcd)
voxel_size = 0.005
pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
# Recompute normals, use a radius neighbourhood of 2 voxels and 30 NN
pcd_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2.0, max_nn=30
    )
)
# Remove some outliers
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_down = pcd_down.select_by_index(ind)
pcd_down = pcd_down.remove_non_finite_points()
# DBSCAN
# Compute fpfh
fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100)
)
pre_processed_centroid = pcd_down.get_center()
print(f"Pre Processed Centroid: {pre_processed_centroid}")
# o3d.visualization.draw_geometries([pcd_down])

o3d.io.write_point_cloud("pdal_point_cloud_input.ply", pcd_down)

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

pipeline_pdal = pdal.Pipeline(pipeline)
pipeline_pdal.execute()

point_cloud = o3d.io.read_point_cloud(
    "pdal_point_cloud_output.ply", print_progress=True
)
o3d.visualization.draw_geometries([point_cloud])
