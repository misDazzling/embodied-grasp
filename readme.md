## Emboided Grasping In One
Grasp检测代码从[https://github.com/graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
移植而来，不过官方原本代码需要cuda 10.1和pytorch 1.6.很多新卡无法安装cuda 10.1，并且cuda 11后的api进行了变化，所以把修改后的knn代码集成在了本仓库，位置在libs下面。测试环境为Ubuntu 20.04 RTX3090 cuda 11.8
依赖：
```
torch==2.1.0
pybullet
open3d
transforms3d==0.4.2
ultralytics==8.2.32
numpy==1.24.4
flask
opencv-python
grasp_nms
```
### 安装
首先安装GraspNet 1B需要的KNN、Pointnet2:
```
cd libs/knn
python setup.py install
cd libs/pointnet2
python setup.py install
```
然后安装所需的库
```
pip install -r requirements.txt
```
### 抓取模块
6D抓取姿态生成采用GraspNet 1B，输入是点云，输出带有Score的多个抓取姿态
![](assets/demo.png#errorMessage=unknown%20error&id=U8CDa&originalType=binary&ratio=1&rotation=0&showTitle=false&status=error&style=none)
![demo.png](https://cdn.nlark.com/yuque/0/2024/png/22340347/1725260050519-20d1aada-e4ca-406b-8244-4668bd4702dc.png#averageHue=%2325f821&clientId=uf429b261-eb91-4&from=ui&id=u7fc6df38&originHeight=807&originWidth=1504&originalType=binary&ratio=0.8955223880597015&rotation=0&showTitle=false&size=449626&status=done&style=none&taskId=u86932640-e383-4446-b7b5-4e4ec5c1097&title=)
### 开放词汇目标检测
开放词汇目标检测采用Ultralytics的Yolov8-world-v2版本，使用Flask将目标检测封装成一个Http接口。然后客户端将图片base64编码以及指定的类别文本传给服务端，执行检测

执行一下脚本将检测服务启动：
```
cd yolo_world
python yolo_world.py
```
服务启动之后可以运行```python demo.py```进行测试:
```python
image_path = '1.jpg'  # 替换为你的图片路径
classes = ['banana']  # 替换为你需要的类别
detections = send_detection_request(image_path, classes)
draw_detections(image_path, detections)
```
得到结果：
![](assets/yolow.png#errorMessage=unknown%20error&id=GuwJI&originalType=binary&ratio=1&rotation=0&showTitle=false&status=error&style=none)![yolow.png](https://cdn.nlark.com/yuque/0/2024/png/22340347/1725260057895-13966656-c596-4635-a7f3-a136335620e1.png#averageHue=%23d4c3ac&clientId=uf429b261-eb91-4&from=ui&id=u6d07c327&originHeight=427&originWidth=574&originalType=binary&ratio=0.8955223880597015&rotation=0&showTitle=false&size=266577&status=done&style=none&taskId=u7a67486c-7052-4078-822f-94facbf4a5a&title=)
### LLM调用API
使用DeepSeek-v2的api进行调用，申请api-key([https://platform.deepseek.com/api_keys)之后保存到keys.txt里面，调用的Prompt模板为：](https://platform.deepseek.com/api_keys)之后保存到keys.txt里面，调用的Prompt模板为：)
```
你是一个机器人，你拥有的技能API如下：
1.get_grasp_by_name(name_text): 输入类别文本（注意是英文，要简短），返回检测候选抓取的list
2.execute_grasp(grasp): 输入候选抓取，然后执行抓取
现在需要你根据你所拥有的技能API，编写python代码完成给你的任务，只输出plan函数，不要输出其他代码以为的内容。你的任务是“帮我拿一下香蕉吧”。
```
下面是一个示例：
输入：
```
你是一个机器人，你拥有的技能API如下：
1.get_grasp_by_name(name_text): 输入类别文本（注意是英文，要简短），返回检测候选抓取的list
2.execute_grasp(grasp): 输入候选抓取，然后执行抓取
现在需要你根据你所拥有的技能API，编写python代码完成给你的任务，只输出plan函数，不要输出其他代码以为的内容。你的任务是“”。
```
输出：
```
def plan():
    # 定义类别文本为"banana"
    name_text = "banana"
    # 调用get_grasp_by_name API获取候选抓取列表
    grasps = get_grasp_by_name(name_text)
    # 调用execute_grasp API执行抓取
    execute_grasp(grasps[0])
```
然后程序会自动提取plan函数，进行执行
完整运行：
先启动yolo_world服务：
```
cd yolo_world
python yolo_world.py
```
再开一个新终端
```
python demo_graspnet2.py
```
环境可以参考如下environment.yml：
```
name: embodied-grasp
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/fastai/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_gnu
  - _sysroot_linux-64_curr_repodata_hack=3=h69a702a_16
  - binutils_impl_linux-64=2.40=ha1999f0_7
  - ca-certificates=2024.8.30=hbcca054_0
  - gcc=14.1.0=h6f9ffa1_1
  - gcc_impl_linux-64=14.1.0=h3c94d91_0
  - kernel-headers_linux-64=3.10.0=h4a8ded7_16
  - ld_impl_linux-64=2.40=hf3520f5_7
  - libffi=3.4.4=h6a678d5_1
  - libgcc=5.2.0=0
  - libgcc-devel_linux-64=14.1.0=h5d3d1c9_100
  - libgcc-ng=14.1.0=h77fa898_0
  - libgomp=14.1.0=h77fa898_0
  - libsanitizer=14.1.0=hcba0ae0_0
  - libstdcxx-ng=14.1.0=hc0a3c3a_0
  - ncurses=6.4=h6a678d5_0
  - openssl=3.3.1=hb9d3cd8_3
  - pip=24.2=py38h06a4308_0
  - python=3.8.19=h955ad1f_0
  - readline=8.2=h5eee18b_0
  - setuptools=72.1.0=py38h06a4308_0
  - sqlite=3.45.3=h5eee18b_0
  - sysroot_linux-64=2.17=h4a8ded7_16
  - tk=8.6.14=h39e8969_0
  - wheel=0.43.0=py38h06a4308_0
  - xz=5.4.6=h5eee18b_1
  - zlib=1.2.13=h5eee18b_1
  - pip:
      - addict==2.4.0
      - annotated-types==0.7.0
      - anyio==4.4.0
      - asttokens==2.4.1
      - attrs==24.2.0
      - backcall==0.2.0
      - blinker==1.8.2
      - certifi==2024.8.30
      - charset-normalizer==3.3.2
      - click==8.1.7
      - clip==1.0
      - comm==0.2.2
      - configargparse==1.7
      - contourpy==1.1.1
      - cycler==0.12.1
      - cython==3.0.11
      - dash==2.17.1
      - dash-core-components==2.0.0
      - dash-html-components==2.0.0
      - dash-table==5.0.0
      - decorator==5.1.1
      - distro==1.9.0
      - exceptiongroup==1.2.2
      - executing==2.0.1
      - fastjsonschema==2.20.0
      - filelock==3.15.4
      - flask==3.0.3
      - fonttools==4.53.1
      - fsspec==2024.6.1
      - ftfy==6.2.3
      - grasp-nms==1.0.2
      - h11==0.14.0
      - httpcore==1.0.5
      - httpx==0.27.2
      - idna==3.8
      - importlib-metadata==8.4.0
      - importlib-resources==6.4.4
      - ipython==8.12.3
      - ipywidgets==8.1.5
      - itsdangerous==2.2.0
      - jedi==0.19.1
      - jinja2==3.1.4
      - jiter==0.5.0
      - joblib==1.4.2
      - jsonschema==4.23.0
      - jsonschema-specifications==2023.12.1
      - jupyter-core==5.7.2
      - jupyterlab-widgets==3.0.13
      - kiwisolver==1.4.5
      - knn-pytorch==0.1
      - markupsafe==2.1.5
      - matplotlib==3.7.5
      - matplotlib-inline==0.1.7
      - mpmath==1.3.0
      - nbformat==5.10.4
      - nest-asyncio==1.6.0
      - networkx==3.1
      - numpy==1.24.4
      - nvidia-cublas-cu12==12.1.3.1
      - nvidia-cuda-cupti-cu12==12.1.105
      - nvidia-cuda-nvrtc-cu12==12.1.105
      - nvidia-cuda-runtime-cu12==12.1.105
      - nvidia-cudnn-cu12==9.1.0.70
      - nvidia-cufft-cu12==11.0.2.54
      - nvidia-curand-cu12==10.3.2.106
      - nvidia-cusolver-cu12==11.4.5.107
      - nvidia-cusparse-cu12==12.1.0.106
      - nvidia-nccl-cu12==2.20.5
      - nvidia-nvjitlink-cu12==12.6.68
      - nvidia-nvtx-cu12==12.1.105
      - open3d==0.18.0
      - openai==1.43.0
      - opencv-python==4.10.0.84
      - packaging==24.1
      - pandas==2.0.3
      - parso==0.8.4
      - pexpect==4.9.0
      - pickleshare==0.7.5
      - pillow==10.4.0
      - pkgutil-resolve-name==1.3.10
      - platformdirs==4.2.2
      - plotly==5.24.0
      - pointnet2==0.0.0
      - prompt-toolkit==3.0.47
      - psutil==6.0.0
      - ptyprocess==0.7.0
      - pure-eval==0.2.3
      - py-cpuinfo==9.0.0
      - pybullet==3.2.6
      - pydantic==2.8.2
      - pydantic-core==2.20.1
      - pygments==2.18.0
      - pyparsing==3.1.4
      - pyquaternion==0.9.9
      - python-dateutil==2.9.0.post0
      - pytz==2024.1
      - pyyaml==6.0.2
      - referencing==0.35.1
      - regex==2024.7.24
      - requests==2.32.3
      - retrying==1.3.4
      - rpds-py==0.20.0
      - scikit-learn==1.3.2
      - scipy==1.10.1
      - seaborn==0.13.2
      - six==1.16.0
      - sniffio==1.3.1
      - stack-data==0.6.3
      - sympy==1.13.2
      - tenacity==9.0.0
      - threadpoolctl==3.5.0
      - torch==2.4.0
      - torchvision==0.19.0
      - tqdm==4.66.5
      - traitlets==5.14.3
      - transforms3d==0.4.2
      - triton==3.0.0
      - typing-extensions==4.12.2
      - tzdata==2024.1
      - ultralytics==8.2.32
      - ultralytics-thop==2.0.5
      - urllib3==2.2.2
      - wcwidth==0.2.13
      - werkzeug==3.0.4
      - widgetsnbextension==4.0.13
      - zipp==3.20.1
prefix: /usr/local/anacoda3/envs/embodied-grasp
```
