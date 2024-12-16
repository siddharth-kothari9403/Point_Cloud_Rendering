import open3d as o3d
from plyfile import PlyData, PlyElement
import numpy as np

sh_coeff = 0.28209479177387814

def SH2RGB(sh):
    return sh * sh_coeff + 0.5

def sigmoid(x):
    return 1/ (1+np.exp(-x))

def SH_to_RGB(f_dc_0, f_dc_1, f_dc_2):
    # This is an approximation. You might need more terms for higher accuracy.
    
    # The spherical harmonics basis functions for each coefficient
    # Here we assume a simple case where we use just the 0th order (constant) term
    # and the first order terms.
    
    # Convert the spherical harmonics coefficients back to RGB values (simplified)
    # The coefficients are weighted contributions of the RGB channels
    rgb = np.array([0.5+sh_coeff*f_dc_0, 0.5+sh_coeff*f_dc_1, 0.5+sh_coeff*f_dc_2])
    
    # Normalize the colors to the [0, 1] range (or [0, 255] if needed)
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

# Load the .ply file as a point cloud
ply_file1 = "./output_fruits_white/final/point_cloud/iteration_30000/point_cloud.ply"
# ply_file2 = "./output/final/input.ply"


plydata = PlyData.read(ply_file1)

xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),  axis=1)

features_dc = np.zeros((xyz.shape[0], 3, 1))
features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

rgb = SH2RGB(features_dc[..., 0])
rgb = np.maximum(rgb, 0)

opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
opacities = sigmoid(opacities)
opacity_mask = (opacities>0.005).squeeze(1)

xyz = xyz[opacity_mask]
rgb = rgb[opacity_mask]

max_rgb = np.max(rgb, axis=1)
max_rgb = np.maximum(max_rgb, 1)
rgb = rgb/max_rgb[:, np.newaxis]

# print(rgb)
# print(xyz)

# Extract normal vectors (nx, ny, nz)
normals = np.stack((
    np.asarray(plydata.elements[0]["nx"]),
    np.asarray(plydata.elements[0]["ny"]),
    np.asarray(plydata.elements[0]["nz"])
), axis=1)
normals = normals[opacity_mask]

print("Shape of xyz:", xyz.shape)
print("Shape of rgb:", rgb.shape)
print("Shape of normals: ", normals.shape)

xyz = xyz.astype(np.float32)
rgb = rgb.astype(np.float32)
normals = normals.astype(np.float32)

print("NaN in xyz:", np.isnan(xyz).any())
print("NaN in rgb:", np.isnan(rgb).any())
print("NaN in normals:", np.isnan(normals).any())

n_points = xyz.shape[0]

# try:
#     pcd = o3d.geometry.PointCloud()
#     print("here1")
#     pcd.points = o3d.utility.Vector3dVector(xyz[:100])
#     print("here2")
#     pcd.colors = o3d.utility.Vector3dVector(rgb[:100])
#     print("here3")
#     pcd.normals = o3d.utility.Vector3dVector(normals[:100])
#     o3d.io.write_point_cloud("test.ply", pcd)
#     print("here")

#     point_cloud = o3d.io.read_point_cloud("test.ply")
#     o3d.visualization.draw_geometries([point_cloud])

# except Exception as e:
#     print(f"Error during assignment: {e}")

vertices = np.array([(xyz[i, 0], xyz[i, 1], xyz[i, 2], 
                      rgb[i, 0]*255, rgb[i, 1]*255, rgb[i, 2]*255, 
                      normals[i, 0], normals[i, 1], normals[i, 2]) 
                      for i in range(n_points)], dtype=[('x', 'f4'), 
                                                      ('y', 'f4'), 
                                                      ('z', 'f4'), 
                                                      ('red', 'u1'), 
                                                      ('green', 'u1'), 
                                                      ('blue', 'u1'),
                                                      ('nx', 'f4'),
                                                      ('ny', 'f4'),
                                                      ('nz', 'f4')])

# Create the PLY element
vertex_element = PlyElement.describe(vertices, 'vertex')

# Write to PLY file
ply_file = 'output_fruits_white.ply'
ply_data = PlyData([vertex_element], text=True)
ply_data.write(ply_file)
