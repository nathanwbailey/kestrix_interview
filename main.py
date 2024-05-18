"""Main file for kestrix interview code."""
import open3d as o3d
import numpy as np
import pdal
from copy import deepcopy


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
o3d.io.write_point_cloud("property_point_cloud.ply", pcd_down)
pre_processed_centroid = pcd_down.get_center()
print(f"Pre Processed Centroid: {pre_processed_centroid}")

#Save file for PDAL
o3d.io.write_point_cloud('pdal_point_cloud_input.ply', pcd_down)

#PDAL pipeline
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
        "limits":"Classification![2:2]" //Remove the ground points
    },
    {
        "type": "writers.ply",
        "filename": "pdal_point_cloud_output.ply"
    }
]
'''

pipeline_pdal = pdal.Pipeline(pipeline)
pipeline_pdal.execute()

point_cloud = o3d.io.read_point_cloud('pdal_point_cloud_output.ply', print_progress=True)
o3d.visualization.draw_geometries([point_cloud])

#TODO
def is_valid_plane(plane: o3d.geometry.PointCloud, centroid_to_compare: np.ndarray,  min_area: int = 10, max_area: int = 25, centroid_threshold: int = 2) -> bool:
    return True

#Extract the planes using RANSAC
remaining_points = deepcopy(pcd_down)
planes = []
for _ in range(10):
    plane_eq, inliners = remaining_points.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=10000, probability=0.99999999)
    plane = remaining_points.select_by_index(inliners)

    """
    A valid plane must have a centroid greater than 2M from the pre_processed_centroid.
    It must also must be greater than 10m2 and less than 25m2
    """
    planes.append(plane)
    remaining_points = remaining_points.select_by_index(inliners, invert=True)

    if is_valid_plane(plane, pre_processed_centroid):
        plane.paint_uniform_color([1,0,0])
        remaining_points.paint_uniform_color([0,1,0])
        o3d.visualization.draw_geometries([plane, remaining_points])

