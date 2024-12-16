import open3d as o3d

# Load the .ply file as a point cloud
ply_file = "./output.ply"
point_cloud = o3d.io.read_point_cloud(ply_file)
print(point_cloud.has_colors())

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])
