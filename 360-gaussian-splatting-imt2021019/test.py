import open3d as o3d
import numpy as np

# Create a simple point cloud
points = np.random.rand(100, 3)
colors = np.random.rand(100, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])