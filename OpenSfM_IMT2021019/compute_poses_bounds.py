import numpy as np
import json
import cv2

def axis_angle_to_rotation_matrix(axis_angle):
    # OpenCV's Rodrigues function converts axis-angle to a rotation matrix
    R, _ = cv2.Rodrigues(axis_angle)
    return R

def compute_poses_bounds(reconstruction_file):
    with open(reconstruction_file) as f:
        data = json.load(f)
    
    poses_bounds = []
    
    for shot in data[0]['shots'].values():
        # Camera extrinsics
        R = axis_angle_to_rotation_matrix(np.array(shot['rotation']))
        t = np.array(shot['translation'])
        C = -np.dot(R.T, t)  # Camera center
        
        # Visible 3D points
        points = [np.array(data[0]['points'][pt]['coordinates']) 
                  for pt in data[0]['points']]
        
        if not points:
            continue
        
        points = np.array(points)
        distances = np.linalg.norm(points - C, axis=1)
        
        # Compute depth bounds
        min_depth = np.percentile(distances, 5)  # 5th percentile
        max_depth = np.percentile(distances, 95)  # 95th percentile
        
        # Camera intrinsics
        height = data[0]['cameras'][shot['camera']]['height']
        width = data[0]['cameras'][shot['camera']]['width']
        focal = data[0]['cameras'][shot['camera']]['focal']
        
        # Prepare pose matrix (3x5)
        intrinsics = np.array([height, width, focal]).reshape(3, 1)
        pose_matrix = np.hstack([R, t[:, None], intrinsics])  # Shape (3, 5)
        
        # Flatten the pose matrix and append depth bounds
        pose_bounds = np.hstack([pose_matrix.flatten(), [min_depth, max_depth]])        
        poses_bounds.append(pose_bounds)
    
    # Convert to numpy array
    poses_bounds = np.array(poses_bounds)
    print(poses_bounds.shape)
    return poses_bounds


# Usage
parent="./data/fruits"
reconstruction_file = parent+'/reconstruction.json'
poses_bounds = compute_poses_bounds(reconstruction_file)

# Save as poses_bounds.npy
np.save(parent+'/poses_bounds.npy', poses_bounds)
