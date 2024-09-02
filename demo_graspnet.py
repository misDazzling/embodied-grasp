import pybullet as p
import pybullet_data
import random
import time
import cv2
import math
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from graspnet.graspnet import GraspBaseline
import copy
from scipy.spatial.transform import Rotation as R
from robot import UR5Robotiq85

graspNet = GraspBaseline()

p.connect(p.GUI)  # 使用GUI模式以便可视化
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置数据路径

p.loadURDF("plane.urdf")
tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0.0], useFixedBase=True)

p.setGravity(0,0,-10)


robot = UR5Robotiq85((0.2, -0.7, 0.6), (0, 0, 0))
robot.load()

robot.step_simulation = p.stepSimulation
robot.reset()

# 设置摄像头参数
width = 640
height = 480

fov = 80
aspect = width / height
near = 0.02
far = 1

view_matrix = p.computeViewMatrix([0.23, -0.2, 1.0], [0.23, 0.05, 0.3], [0, 1, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

# 设置Debug摄像头视角
p.resetDebugVisualizerCamera( cameraDistance=2.7, cameraYaw=63,
cameraPitch=-44, cameraTargetPosition=[0,0,0])


# 加载物体，随机位置和姿态
ori = p.getQuaternionFromEuler((0, 0, random.uniform(-np.pi/2, np.pi/2)))
pos = [random.uniform(0.1, 0.3), random.uniform(0, 0.2), 0.626]
hammerId = p.loadURDF("ycb_objects/YcbHammer/model.urdf", basePosition=pos, baseOrientation=ori)



robot.move_ee([0.2, -0.5, 1]+[0, np.pi/2, np.pi/2], 'end')
# 初始化
for i in range(1000):
    p.stepSimulation()

# https://github.com/bulletphysics/bullet3/issues/1924
def get_point_cloud(width, height, view_matrix, proj_matrix):
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

    # get a depth image
    # "infinite" depths will have a value close to 1
    image_arr = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    depth = image_arr[3] # H*W*1
    color = image_arr[2] # H*W*4

    # # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    # view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    view_matrix = np.eye(4)
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
    # tran_pix_world = np.eye(4)

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points, depth, color

pts, depth, color = get_point_cloud(width, height, view_matrix, projection_matrix)

pcd = o3d.geometry.PointCloud()
# 将NumPy数组设置为点云的点
pcd.points = o3d.utility.Vector3dVector(pts)


# pybullet camera 到graspnet camera坐标变换，将摄像头的z轴变换到面对table，以供graspnet进行抓取检测
trans_camera = np.eye(4)
trans_camera = np.array([[-1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])

# pybullet camera to world
trans_camera_world = np.asarray(view_matrix).reshape([4, 4], order="F")

def get_grasp_world(grasp):
    translation = grasp.translation
    rot = grasp.rotation_matrix

    grasp_trans = np.eye(4)
    grasp_trans[:3, :3] = rot
    grasp_trans[:3, -1] = translation

    # 将grasp转换到世界坐标系
    grasp_trans_world = np.linalg.inv(trans_camera_world).dot(np.linalg.inv(trans_camera).dot(grasp_trans))
    return grasp_trans_world

# 运行graspNet，需要将pybullet camera的点云转换到摄像头z轴面向table
gg = graspNet.run(copy.deepcopy(pcd).transform(trans_camera), vis=False)

gg.nms()
gg.sort_by_score()
grasp = gg[0]

# 获取grasp在world坐标下的变换
grasp_world = get_grasp_world(grasp)

trans_x_neg = np.eye(4)
trans_x_neg[0, -1] = -0.15

pre_grasp_world = grasp_world.dot(trans_x_neg)




rot = R.from_matrix(pre_grasp_world[:3, :3])
euler = rot.as_euler('xyz')

robot.move_ee(pre_grasp_world[:3, -1].tolist()+euler.tolist(), 'end')
for i in range(200):
    p.stepSimulation()
    time.sleep(1/240)

robot.move_gripper(0)
for i in range(200):
    p.stepSimulation()
    time.sleep(1/240)

mesh_pre_grasp = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]).transform(pre_grasp_world)

mesh_grasp = grasp.to_open3d_geometry().transform(np.linalg.inv(trans_camera_world).dot(np.linalg.inv(trans_camera)))


viewer = o3d.visualization.Visualizer()
viewer.create_window()

viewer.add_geometry(pcd.transform(np.linalg.inv(trans_camera_world)))
viewer.add_geometry(mesh_grasp)
viewer.add_geometry(mesh_pre_grasp)

viewer.run()
viewer.destroy_window()