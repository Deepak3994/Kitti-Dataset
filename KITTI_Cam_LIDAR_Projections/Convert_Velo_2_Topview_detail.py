import numpy as np
import matplotlib.pyplot as plt


def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

# bin file -> numpy array
velo_points = load_from_bin('/home/deepaknayak/Documents/kitti-datasets/KITTI_Cam_LIDAR_Projections/2011_09_26_drive_0001_sync/velodyne_points/data/0000000089.bin')

print(velo_points.shape)


def normalize_depth(val, min_v, max_v):
    """
    print 'normalized depth value'
    normalize values to 0-255 & close distance value has high value. (similar to stereo vision's disparity map)
    """
    return (((max_v - val) / (max_v - min_v)) * 255).astype(np.uint8)


def in_range_points(points, x, y, z, x_range, y_range, z_range):
    """ extract in-range points """
    return points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], \
                                         y < y_range[1], z > z_range[0], z < z_range[1]))]


def velo_points_2_top_view(points, x_range, y_range, z_range, scale):
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2)

    # extract in-range points
    x_lim = in_range_points(x, x, y, z, x_range, y_range, z_range)
    y_lim = in_range_points(y, x, y, z, x_range, y_range, z_range)
    dist_lim = in_range_points(dist, x, y, z, x_range, y_range, z_range)

    # * x,y,z range are based on lidar coordinates
    x_size = int((y_range[1] - y_range[0]))
    y_size = int((x_range[1] - x_range[0]))

    # convert 3D lidar coordinates(vehicle coordinates) to 2D image coordinates
    # Velodyne coordinates info : http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
    # scale - for high resolution
    x_img = -(y_lim * scale).astype(np.int32)
    y_img = -(x_lim * scale).astype(np.int32)

    # shift negative points to positive points (shift minimum value to 0)
    x_img += int(np.trunc(y_range[1] * scale))
    y_img += int(np.trunc(x_range[1] * scale))

    # normalize distance value & convert to depth map
    max_dist = np.sqrt((max(x_range) ** 2) + (max(y_range) ** 2))
    dist_lim = normalize_depth(dist_lim, min_v=0, max_v=max_dist)

    # array to img
    img = np.zeros([y_size * scale + 1, x_size * scale + 1], dtype=np.uint8)
    img[y_img, x_img] = dist_lim

    return img


# Plot result
top_image = velo_points_2_top_view(velo_points, x_range=(-20, 20), y_range=(-20, 20), z_range=(-2, 2), scale=10)
plt.subplots(1, 1, figsize=(5, 5))
plt.imshow(top_image)
plt.axis('off')


""" different range's result """

top_image = velo_points_2_top_view(velo_points, x_range=(-10, 20), y_range=(-10, 20), z_range=(-2, 2), scale=10)
plt.subplots(1,1, figsize = (4,4))
plt.imshow(top_image)
plt.axis('off')


import glob
import cv2

lidar_points = glob.glob('/home/deepaknayak/Documents/kitti-datasets/KITTI_Cam_LIDAR_Projections/2011_09_26_drive_0001_sync/velodyne_points/data/*.bin')

# pre define range for image size
x_range, y_range, z_range, scale = (-20,20), (-20,20), (-2, 2), 10
size = int((max(x_range)-min(x_range)) * scale), int((max(y_range)-min(y_range)) * scale)

""" save top view video """
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid = cv2.VideoWriter('topview.avi', fourcc, 25.0, size, False)

for point in lidar_points:
    velo = load_from_bin(point)
    img = velo_points_2_top_view(velo, x_range=x_range, y_range=y_range, z_range=z_range, scale=scale)
    vid.write(img)

print('video saved')
vid.release()