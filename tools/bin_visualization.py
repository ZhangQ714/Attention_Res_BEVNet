# 此代码用于可视化激光雷达数据  ; 单帧
import numpy as np
import mayavi.mlab
import os

velodyne_path = '../data/semantic_kitti_4class_100x100/08/velodyne/03809.bin'
pointcloud = np.fromfile(str(velodyne_path), dtype=np.float32, count=-1).reshape([-1, 4])

x = -pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point

r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

degr = np.degrees(np.arctan(z / d))

vals = 'height'
if vals == "height":
    col = z
else:
    col = d

fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080))
mayavi.mlab.points3d(x, y, z,
                     col,  # Values used for Color
                     mode="point",
                     colormap='spectral',  # 'bone', 'copper', 'gnuplot'
                     # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     )

mayavi.mlab.show()