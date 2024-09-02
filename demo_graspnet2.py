import pybullet as p
import pybullet_data
import random
import time
import cv2
import math
import numpy as np
import open3d as o3d
import requests
import base64
from matplotlib import pyplot as plt
from openai import OpenAI
from graspnet.graspnet import GraspBaseline
import copy
import re
from scipy.spatial.transform import Rotation as R
from robot import UR5Robotiq85


class EmboidedGrasp:
    def __init__(self):
        self.graspNet = GraspBaseline() # 加载抓取检测模型
        p.connect(p.GUI)  # 使用GUI模式以便可视化
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置数据路径

        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=[0, 0, 0.0], useFixedBase=True)

        

        p.setGravity(0,0,-10) # 设置重力

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

        self.robot = UR5Robotiq85((0.2, -0.6, 0.6), (0, 0, 0))
        self.robot.load()

        self.robot.step_simulation = p.stepSimulation
        self.robot.reset()

        self.setup_debug_camera()
        self.setup_capture_camera()

        self.robot.move_ee([0.2, -0.5, 1]+[0, np.pi/2, np.pi/2], 'end') # 机械臂移动到初始位置
        # 初始化
        for i in range(1000):
            p.stepSimulation()

        self.load_objects()
        for i in range(2000):
            p.stepSimulation()

        self.llm_client = OpenAI(api_key=open('keys.txt').readline().strip(), base_url="https://api.deepseek.com")

    def setup_debug_camera(self):
        # 设置Debug摄像头视角
        p.resetDebugVisualizerCamera( cameraDistance=2.7, cameraYaw=63, cameraPitch=-44, cameraTargetPosition=[0,0,0])


    def setup_capture_camera(self):
        # 设置采集摄像头参数
        self.width = 640
        self.height = 480

        self.fov = 100
        self.aspect = self.width / self.height
        self.near = 0.02
        self.far = 1

        self.camera_pos = [0, 0, 1.1]
        self.view_matrix = p.computeViewMatrix(self.camera_pos, [0., 0., 0.6], [0, 1, 0])
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

    def load_objects(self):

        
        ori = p.getQuaternionFromEuler((0, 0, np.pi*1.02))
        pos = [0.1, 0, 0.626]
        p.loadURDF("ycb_objects/YcbBanana/model.urdf", basePosition=pos, baseOrientation=ori, globalScaling=1.5)

        ori = p.getQuaternionFromEuler((0, 0, 0))
        pos = [-0.15, -0.12, 0.66]
        p.loadURDF("ycb_objects/green_bowl/model.urdf", basePosition=pos, baseOrientation=ori, globalScaling=1.5)


    def rgbd2points_camera(self): # 获取pybullet摄像头坐标系下的点云
        # https://github.com/bulletphysics/bullet3/issues/1924

        image_arr = p.getCameraImage(width=self.width, height=self.height, viewMatrix=self.view_matrix, projectionMatrix=self.projection_matrix, lightDirection=self.camera_pos)
        depth = image_arr[3] # H*W*1
        color = image_arr[2] # H*W*4

        color_r = color[..., 0].reshape(-1)
        color_g = color[..., 1].reshape(-1)
        color_b = color[..., 2].reshape(-1)

        projection_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(projection_matrix)

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.height, -1:1:2 / self.width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        rgbs = np.stack([color_r, color_g, color_b], axis=1).astype('float32')/255.0
        rgbs = rgbs[z < 0.99]
        
        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        self.points_camera = points
        self.rgbs = rgbs
        self.depth = depth
        self.color = color

    def grasp2pixel(self, grasp_poses): #将world坐标系下的3维点投影到camera像素坐标系
        projection_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")

        grasp_points = []
        for grasp in grasp_poses['grasp_world']:
            grasp_point = grasp[:, -1].reshape((-1, 4))
            grasp_points.append(grasp_point)

        grasp_points = np.concatenate(grasp_points, axis=0)

        grasp_points = grasp_points @ np.transpose(projection_matrix @ view_matrix)

        points_ndc = grasp_points / grasp_points[:, 3:4]
        
        points_ndc[:, 1] *= -1 # negative y
        points_screen = (points_ndc[:, :2] * 0.5 + 0.5) * np.array([self.width, self.height])

        return points_screen[:, :2]

    def detect_by_text(self, color, class_text):
        tmp_image_path = 'tmp.jpg'
        cv2.imwrite(tmp_image_path, color[..., :3][..., ::-1]) # 保存RGB图像，以供后续调用yolo-world进行开放词汇检测
        # 读取图片并转换为 Base64 编码
        with open(tmp_image_path, 'rb') as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # 准备请求数据
        data = {
            'image': image_base64,
            'classes': [class_text]
        }

        # 发送 POST 请求
        response = requests.post('http://127.0.0.1:5000/detect', json=data)

        # 处理响应
        if response.status_code == 200:
            detections = response.json().get('detections', [])
            print('Detections:', detections)
            return detections
        else:
            print('Error:', response.text)
            return []

    def filter_grasp_by_text(self, class_text, grasp_poses, vis=False): # 根据2d检测bbox，对grasp_poses进行filter

        detections = self.detect_by_text(self.color, class_text)
        grasp_points = self.grasp2pixel(grasp_poses)

        if len(detections):
            det = detections[0]
            x1, y1, x2, y2, score, class_name = det

            grasp_candidate = []
            for i, grasp_p in enumerate(grasp_points):
                # 根据grasp_points是否在bbox进行filter
                if grasp_p[0]>x1 and grasp_p[0]<x2 and grasp_p[1]>y1 and grasp_p[1]<y2:
                    grasp_candidate.append(i)

            if len(grasp_candidate):
                
                scores = [grasp_poses['score'][i] for i in grasp_candidate]
                grasp_points = [grasp_points[i] for i in grasp_candidate]
                grasps = [grasp_poses['grasp_world'][i] for i in grasp_candidate]

                # 简单规则，更偏向垂直方向的grasp
                grasps_rule_score = np.array([abs(grasp_poses['grasp_world'][i][2, 0]) for i in grasp_candidate])
                max_id = np.argmax(np.array(scores) + grasps_rule_score)

                if vis:
                    plt.imshow(self.color)

                    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red'))
                    plt.gca().text(x1, y1 - 5, f'{score:.2f}', color='red', fontsize=10, backgroundcolor='none')

                    plt.plot(grasp_points[max_id][0], grasp_points[max_id][1], 'go')
                    plt.title(f'{class_text} Grasp Score: {scores[max_id]:.2f}')
                    plt.show()

                return [grasps[max_id]]
            
        return []

    def get_grasp(self, samples=10, vis=True):
        self.rgbd2points_camera()
        pcd = o3d.geometry.PointCloud()
        # 将NumPy数组设置为点云的点
        pcd.points = o3d.utility.Vector3dVector(self.points_camera)
        pcd.colors = o3d.utility.Vector3dVector(self.rgbs)

        # pybullet camera 到graspnet camera坐标变换，将摄像头的z轴变换到面对table，以供graspnet进行抓取检测
        trans_camera = np.array([[-1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])

        # pybullet camera to world
        trans_camera_world = np.asarray(self.view_matrix).reshape([4, 4], order="F")

        def get_grasp_world(grasp):
            translation = grasp.translation
            rot = grasp.rotation_matrix

            grasp_trans = np.eye(4)
            grasp_trans[:3, :3] = rot
            grasp_trans[:3, -1] = translation

            # 将grasp转换到世界坐标系
            grasp_trans_world = np.linalg.inv(trans_camera_world).dot(np.linalg.inv(trans_camera).dot(grasp_trans))
            return grasp_trans_world
                
        gg = self.graspNet.run(copy.deepcopy(pcd).transform(trans_camera), vis=False)

        gg.nms()
        gg.sort_by_score()

        grasp_poses = {'score':[], 'grasp_world':[]}

        for grasp in gg[:samples]:
            # 获取grasp在world坐标下的变换
            grasp_world = get_grasp_world(grasp)

            grasp_poses['score'].append(grasp.score)
            grasp_poses['grasp_world'].append(grasp_world)

        if vis:
            viewer = o3d.visualization.Visualizer()
            viewer.create_window()
            viewer.add_geometry(pcd.transform(np.linalg.inv(trans_camera_world)))

            for grasp in gg[:samples]:
                mesh_grasp = grasp.to_open3d_geometry().transform(np.linalg.inv(trans_camera_world).dot(np.linalg.inv(trans_camera)))
                viewer.add_geometry(mesh_grasp)

            viewer.run()
            viewer.destroy_window()

        return grasp_poses

    def execute_grasp(self, grasp_world):

        # pre grasp offset，以供后续夹爪进行抓取
        trans_x_neg = np.eye(4)
        trans_x_neg[0, -1] = -0.3
        pre_grasp_world1 = grasp_world.dot(trans_x_neg)

        def get_xyzrpy(grasp):
            print(grasp)
            rot = R.from_matrix(grasp[:3, :3])
            euler = rot.as_euler('xyz')
            xyzrpy = np.array(grasp[:3, -1].tolist()+euler.tolist())
            return xyzrpy


        self.robot.move_ee(get_xyzrpy(pre_grasp_world1)[:3].tolist()+[0, np.pi/2, np.pi/2], 'end')
        for i in range(1000):
            p.stepSimulation()
            time.sleep(1/240)

        self.robot.move_ee(get_xyzrpy(pre_grasp_world1), 'end')
        for i in range(1000):
            p.stepSimulation()
            time.sleep(1/240)
        
        trans_x_neg[0, -1] = -0.12 
        pre_grasp_world2 = grasp_world.dot(trans_x_neg)

        self.robot.move_ee(get_xyzrpy(pre_grasp_world2), 'end')
        for i in range(1000):
            p.stepSimulation()
            time.sleep(1/240)

        self.grasp()

        self.robot.move_ee(get_xyzrpy(pre_grasp_world1), 'end')
        for i in range(500):
            p.stepSimulation()
            time.sleep(1/240)

        up_z = get_xyzrpy(pre_grasp_world1)
        up_z = [up_z[0], up_z[1], up_z[2]]+[0, np.pi/2, np.pi/2]
        self.robot.move_ee(up_z, 'end')
        for i in range(500):
            p.stepSimulation()
            time.sleep(1/240)

        up_z[2] += 0.1
        self.robot.move_ee(up_z, 'end')
        for i in range(500):
            p.stepSimulation()
            time.sleep(1/240)


    def move_xyz(self, xyz):
        self.robot.move_ee(xyz+[0, np.pi/2, np.pi/2], 'end')
        for i in range(500):
            p.stepSimulation()
            time.sleep(1/240)

    def ungrasp(self):
        self.robot.move_gripper(0.08)
        for i in range(500):
            p.stepSimulation()
            time.sleep(1/240)
        
    def grasp(self):
        self.robot.move_gripper(0.0)
        for i in range(500):
            p.stepSimulation()
            time.sleep(1/240)

    def plan_from_llm(self, task):
        template = """你是一个机器人，你拥有的技能API如下：
        1. get_grasp_by_name(name_text): 输入类别文本（注意是英文，要简短），返回检测候选抓取的List
        2. execute_grasp(grasp): 输入候选抓取，然后执行抓取
        现在需要你根据你所拥有的技能API，编写python代码完成给你的任务，只输出plan函数，不要输出其他代码以为的内容。你的任务是“KKKK”。
        """

        response = self.llm_client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": template.replace('KKKK', task)},
            ],
            stream=False
        )

        return response.choices[0].message.content

em_grasp = EmboidedGrasp()

# grasp_poses = em_grasp.get_grasp(samples=100, vis=True)

# grasp = em_grasp.filter_grasp_by_text('banana', grasp_poses)

# if len(grasp):
#     em_grasp.execute_grasp(grasp[0])
# while True:
#     p.stepSimulation()

# 封装api，以供LLM调用

def get_grasp_by_name(name_text):
    grasp_poses = em_grasp.get_grasp(samples=100, vis=False)
    grasp = em_grasp.filter_grasp_by_text(name_text, grasp_poses)
    return grasp

def execute_grasp(grasp):
    em_grasp.execute_grasp(grasp)

print('!!!!!!!!\n\n')
task = input("请输入您的指令:")
# task = '帮我拿一下香蕉吧'
print('好的！这是我编写的代码：')
code = em_grasp.plan_from_llm(task)

pattern = r"`python\n([\s\S]*?)`"
match = re.search(pattern, code, re.DOTALL)

if match:
    # 提取代码块内容
    code_block = match.group(1)
    print(code_block)

    exec(code_block  + '\nplan()')

while True:
    p.stepSimulation()