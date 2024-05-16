"""Main file for kestrix interview code."""
import open3d as o3d
import numpy as np


#Input the mesh file and convert it to a point cloud using open3D
mesh = o3d.io.read_triangle_mesh('property.ply', print_progress=True)
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=25000)
pcd = mesh.sample_points_poisson_disk(number_of_points=10000, pcl=pcd)
# o3d.visualization.draw_geometries([pcd])

#Preprocess the point cloud
#Downsample into voxels
voxel_size=0.05
pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
#Recompute normals, use a radius neighbourhood of 2 voxels and 30 NN
pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2.0, max_nn=30))
#Remove some outliers
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_down = pcd_down.select_by_index(ind)
#DBSCAN
#Compute fpfh
fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,max_nn=100))

o3d.visualization.draw_geometries([pcd_down])


#Extract some planes
