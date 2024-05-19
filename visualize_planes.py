import open3d as o3d

mesh = o3d.io.read_point_cloud('wall_plane_2.ply', print_progress=True)
o3d.visualization.draw_geometries([mesh])