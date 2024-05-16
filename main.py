"""Main file for kestrix interview code."""
import open3d as o3d
import numpy as np

mesh = o3d.io.read_triangle_mesh('property.ply')
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))
