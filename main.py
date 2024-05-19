"""Main file for kestrix interview code."""
from copy import deepcopy
import open3d as o3d
import numpy as np
import pdal
import matplotlib.pyplot as plt
from utils import is_valid_plane
from utils import save_plane_to_file


#Input the mesh file
mesh = o3d.io.read_triangle_mesh('property.ply', print_progress=True)
#Print some summary information about the Mesh
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Faces:')
print(np.asarray(mesh.triangles))
print('Colors:')
print(np.asarray(mesh.vertex_colors))
bounding_box_mesh = mesh.get_axis_aligned_bounding_box()
print(f"Min bound: {bounding_box_mesh.min_bound}")
print(f"Max bound: {bounding_box_mesh.max_bound}")

# Convert the mesh to a point cloud
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
#Remove outliers
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_down = pcd_down.select_by_index(ind)
pcd_down = pcd_down.remove_non_finite_points()


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
#Execute the PDAL pipeline and obtain the output point cloud
pipeline_pdal = pdal.Pipeline(pipeline)
pipeline_pdal.execute()
pcd_down = o3d.io.read_point_cloud('pdal_point_cloud_output.ply', print_progress=True)

#Get the centroid after the pcd has been processed
pre_processed_centroid = pcd_down.get_center()
print(f"Pre Processed Centroid: {pre_processed_centroid}")

#Extract the planes using RANSAC
remaining_points = deepcopy(pcd_down)
pcd_down_copy = deepcopy(pcd_down)
planes = []
roof_num = 0
facade_num = 0
for _ in range(10):
    #Get the plane from RANSAC
    plane_eq, inliners = remaining_points.segment_plane(distance_threshold=0.08, ransac_n=3, num_iterations=5000)
    #Extract the plane from the point cloud
    plane = remaining_points.select_by_index(inliners)
    planes.append(plane)
    #Remove the plane from the overall point cloud
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
    
    #Save the plane to a PLY file if valid
    if valid_roof_plane[0] or valid_facade_plane[0]:
        plane.paint_uniform_color([1,0,0])
        save_plane_to_file(plane_to_save=plane, plane_number=(roof_num if valid_roof_plane[0] else facade_num), plane_type=('roof' if valid_roof_plane[0] else 'wall'))
        remaining_points_plane = pcd_down_copy.select_by_index(inliners, invert=True)
        remaining_points.paint_uniform_color([0,1,0])

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
