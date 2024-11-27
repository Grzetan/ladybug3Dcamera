import numpy as np
import cv2

def generate_ray(pixel, K, D):
    normalized_coords = cv2.fisheye.undistortPoints(
        np.array([[[pixel[0], pixel[1]]]], dtype=np.float32), K, D
    ).reshape(-1)
    ray_1 = np.array([normalized_coords[0], normalized_coords[1], 1.0])
    return ray_1

def euler_to_ray(euler_angles):
    """
    Converts Euler angles to a 3D ray in Cartesian coordinates.

    Args:
        euler_angles: A tuple or list of three angles (roll, pitch, yaw) in radians.
                      - roll: Rotation around the X-axis.
                      - pitch: Rotation around the Y-axis.
                      - yaw: Rotation around the Z-axis.

    Returns:
        A 3D unit vector (ray) as a numpy array in Cartesian coordinates.
    """
    roll, pitch, yaw = euler_angles

    # Calculate the combined rotation matrix from Euler angles
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combine the rotations: R = R_z * R_y * R_x (yaw -> pitch -> roll)
    R = R_z @ R_y @ R_x

    # The ray starts as a unit vector along the Z-axis in the camera's local frame
    ray = np.array([0, 0, 1])

    # Apply the rotation to the ray
    rotated_ray = R @ ray

    # Ensure the ray is a unit vector
    rotated_ray /= np.linalg.norm(rotated_ray)

    return rotated_ray

def ray_to_euler(ray):
    # Step 2: Compute the Euler angles from the 3D ray
    x, y, z = ray
    
    # Compute Pitch, Yaw, and Roll (XYZ convention)
    pitch = np.arctan2(y, np.sqrt(x**2 + z**2))  # Rotation around X-axis
    yaw = np.arctan2(-x, z)  # Rotation around Y-axis
    roll = 0  # No roll for direction vector, assuming it is along a single axis

    # Return the Euler angles in radians
    return pitch, yaw, roll

def compute_3d_rays(pixel_1, pixel_2, K1, D1, K2, D2, T, theta):
    """
    Compute 3D rays for two distorted pixels in a shared coordinate system.

    Args:
        pixel_1: (u1, v1) Distorted pixel coordinates in camera 1.
        pixel_2: (u2, v2) Distorted pixel coordinates in camera 2.
        K1: Intrinsic matrix for camera 1.
        D1: Distortion coefficients for camera 1.
        K2: Intrinsic matrix for camera 2.
        D2: Distortion coefficients for camera 2.
        T: Translation vector from camera 2 to camera 1 (3x1 vector).
        theta_y: Rotation angle between the cameras around the Y-axis (in radians).

    Returns:
        ray_1: 3D ray in camera 1's coordinate system.
        ray_2_in_cam1: 3D ray from camera 2 transformed to camera 1's coordinate system.
    """

    # Undistort pixel coordinates to normalized coordinates for camera 1
    normalized_coords_1 = cv2.fisheye.undistortPoints(
        np.array([[[pixel_1[0], pixel_1[1]]]], dtype=np.float32), K1, D1
    ).reshape(-1)
    print(normalized_coords_1)
    ray_1 = np.array([normalized_coords_1[0], normalized_coords_1[1], 1.0])

    # Undistort pixel coordinates to normalized coordinates for camera 2
    normalized_coords_2 = cv2.fisheye.undistortPoints(
        np.array([[[pixel_2[0], pixel_2[1]]]], dtype=np.float32), K2, D2
    ).reshape(-1)
    ray_2 = np.array([normalized_coords_2[0], normalized_coords_2[1], 1.0])
    ray2_euler = ray_to_euler(ray_2)
    print("orig", ray2_euler)
    ray2 = euler_to_ray(ray_2)
    print("orig", ray2)
    # Transform ray_2 into camera 1's coordinate system
    return ray_1, ray2


# Example Usage
if __name__ == "__main__":
    # Input values

    # pixel_1 = (600, 418)  # top left corner of ad
    # pixel_2 = (265, 470)  # top left corner of ad

    # pixel_1 = (607, 671)  # top left corner of ad
    # pixel_2 = (273, 639)  # top left corner of ad

    # pixel_1 = (915, 445)  # top left corner of ad
    # pixel_2 = (392, 503)  # top left corner of ad

    pixel_1 = (933, 664)  # top left corner of ad
    pixel_2 = (401, 627)  # top left corner of ad

    K1 = np.array([[610.4367424810383, 0.0, 504.6321581261914], [0.0, 610.2550573171218, 604.2666914429974], [0.0, 0.0, 1.0]])
    D1 = np.array([[0.006859811423695183], [-0.05971827153369545], [0.1148248634988694], [-0.06113158099265619]])

    K2 = np.array([[623.9013847638885, 0.0, 505.82461862738677], [0.0, 623.5620633553356, 611.0252926333887], [0.0, 0.0, 1.0]])
    D2 = np.array([[-0.000680666865227473], [0.0695711526486394], [-0.24818647566497218], [0.2315876435875059]])

    print(ray_to_euler(generate_ray(pixel_1, K1, D1)))
    print(ray_to_euler(generate_ray(pixel_2, K2, D2)))

    # rays = []
    # width = 1024  # Image width
    # height = 1232  # Image height

    # # Loop only through edge pixels
    # for x in range(width):
    #     for y in [0, height - 1]:  # Only the top and bottom edges
    #         r = ray((x, y), K1, D1)
    #         for i in range(10):
    #             rays.append(r * i)

    # for y in range(height):
    #     for x in [0, width - 1]:  # Only the left and right edges
    #         r = ray((x, y), K1, D1)
    #         for i in range(10):
    #             rays.append(r * i)

    # with open("output.ply", "w") as f:
    #     f.write("ply\n")
    #     f.write("format ascii 1.0\n")
    #     f.write(f"element vertex {len(rays)}\n")
    #     f.write("property float x\n")
    #     f.write("property float y\n")
    #     f.write("property float z\n")
    #     f.write("element face 0\n")
    #     f.write("property list uchar int vertex_index\n")
    #     f.write("end_header\n")

    #     # Write the rays data line by line
    #     for r in rays:
    #         f.write(f"{r[0]} {r[1]} {r[2]}\n")

    T = np.array([15, 0.0, 0.0])  # Translation vector (camera 2 relative to camera 1)
    theta_z = np.deg2rad(-72)  # 10 degrees rotation around Z-axis

    # # Compute 3D rays
    ray_1, ray_2_in_cam1 = compute_3d_rays(pixel_1, pixel_2, K1, D1, K2, D2, T, theta_z)
    print(ray_to_euler(ray_2_in_cam1))

    # # Output results
    # print("Ray in camera 1's coordinate system:", ray_1)
    # print("Ray from camera 2 in camera 1's coordinate system:", ray_2_in_cam1)
