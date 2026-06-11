# YOLO Object Detection Project

这个项目当前用于训练 YOLO 模型识别 3 类物体：

- `Cleaner_bottle`: Cleaner bottle
- `Salt_box`: Salt box
- `tomato_soup_can`: Tomato soup can

## 1. 安装环境

建议在 WSL/Linux 中使用 Python 虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果你已有 Conda 环境，也可以直接：

```bash
pip install -r requirements.txt
```

## 2. 数据集目录

YOLO 需要图片和对应的 `.txt` 标注文件。目录已经按训练、验证、测试划分好：

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

每张图片都需要一个同名 `.txt` 标注文件，例如：

```text
dataset/images/train/img001.jpg
dataset/labels/train/img001.txt
```

YOLO 标注格式为：

```text
class_id x_center y_center width height
```

所有坐标都必须是 0 到 1 之间的归一化值。类别编号如下：

```text
0 Cleaner_bottle
1 Salt_box
2 tomato_soup_can
```

## OpenCV 轮廓 yaw 估计

如果已经有目标检测框，可以先把检测框内的物体裁出来，再用 OpenCV 做二值化、轮廓提取和 yaw 估计：

```bash
python yaw_estimator.py --image path/to/image.jpg --bbox x1 y1 x2 y2 --debug-dir debug_yaw
```

其中 `bbox` 是检测框左上角和右下角坐标，单位是像素。脚本会输出图像平面内的 yaw 角度，并在 `debug_yaw/` 保存裁剪图、二值图和轮廓调试图。

注意：`white_cube` 和 `black_cube` 如果从相机视角看起来接近正方形，仅靠外轮廓无法稳定判断 yaw。除非 cube 上有明显标记、纹理、缺口，或者使用角点/ArUco marker，否则 0 度和 90 度等方向在视觉上可能是等价的。

## 3. 拍摄图片建议

每个类别建议至少拍 100 到 300 张图片。为了让模型更稳定，图片要覆盖：

- 不同角度：正面、侧面、俯视、斜视
- 不同距离：近距离、中距离、远一点
- 不同光照：强光、弱光、阴影
- 不同背景：桌面、地面、架子、杂物旁边
- 物体组合：单个物体、多个物体同框

如果物体外观很相似，比如大番茄罐头和小番茄罐头，一定要多拍尺寸对比明显、角度多样的图片。

## 4. 标注图片

推荐使用以下任意工具：

- CVAT: <https://www.cvat.ai/>
- Roboflow: <https://roboflow.com/>
- Label Studio: <https://labelstud.io/>
- labelImg

导出格式选择 YOLO / YOLOv8 / YOLO txt。

## 5. 训练模型

把图片和标注放好后运行：

```bash
python train.py
```

默认使用 `yolov8n.pt`，速度快，适合先验证流程。训练结果会保存在：

```text
runs/detect/three_objects/
```

训练好的模型通常在：

```text
runs/detect/three_objects/weights/best.pt
```

## 6. 测试预测

对单张图片或一个文件夹预测：

```bash
python predict.py --source path/to/image_or_folder --weights runs/detect/three_objects/weights/best.pt
```

预测结果会保存在 `runs/detect/predict/` 下。

## 7. 常见问题

如果训练时提示没有图片，请检查图片是否放在 `dataset/images/train/` 和 `dataset/images/val/`。

如果训练时提示 label 不存在，请确认每张图片都有同名 `.txt` 文件，空图片也可以有空 `.txt` 文件。

如果识别效果不好，通常不是代码问题，而是数据问题。优先增加图片数量、提高标注质量、加入更多背景和光照变化。
