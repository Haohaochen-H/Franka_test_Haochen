# YOLO Object Detection Project

This project is used to train a YOLO model for detecting five object classes:

- `Cleaner_bottle`: Cleaner bottle
- `Salt_box`: Salt box
- `tomato_soup_can`: Tomato soup can
- `Orange_cube`: Orange cube
- `Yellow_cube`: Yellow cube

## 1. Environment Setup

Using a Python virtual environment on WSL/Linux is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you already have a Conda environment, you can also install the requirements directly:

```bash
pip install -r requirements.txt
```

## 2. Dataset Layout

YOLO requires images and matching `.txt` label files. The dataset is already split into training, validation, and test folders:

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

Each image needs a label file with the same stem, for example:

```text
dataset/images/train/img001.jpg
dataset/labels/train/img001.txt
```

YOLO label format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized to the range 0 to 1. Class IDs:

```text
0 Cleaner_bottle
1 Salt_box
2 tomato_soup_can
3 Orange_cube
4 Yellow_cube
```

## OpenCV Contour Yaw Estimation

If an object detection bounding box is available, the object can be cropped from the bounding box and processed with OpenCV thresholding, contour extraction, and yaw estimation:

```bash
python yaw_estimator.py --image path/to/image.jpg --bbox x1 y1 x2 y2 --debug-dir debug_yaw
```

Here `bbox` is the top-left and bottom-right bounding box coordinates in pixels. The script prints the image-plane yaw angle and saves the crop, binary mask, and contour debug images under `debug_yaw/`.

Note: if `white_cube` or `black_cube` looks nearly square from the camera viewpoint, its yaw cannot be estimated reliably from the outer contour alone. Unless the cube has a clear marker, texture, notch, corner feature, or ArUco marker, 0 degrees and 90 degrees may be visually equivalent.

## 3. Image Capture Suggestions

Capture at least 100 to 300 images per class. For a more robust model, cover:

- Different viewpoints: front, side, top-down, and oblique views
- Different distances: close, medium, and farther views
- Different lighting: bright light, dim light, and shadows
- Different backgrounds: tabletop, floor, shelves, and cluttered scenes
- Object combinations: single-object and multi-object images

If two objects look similar, such as large and small tomato cans, capture many images with clear scale differences and varied viewpoints.

## 4. Image Annotation

Recommended annotation tools:

- CVAT: <https://www.cvat.ai/>
- Roboflow: <https://roboflow.com/>
- Label Studio: <https://labelstud.io/>
- labelImg

Export annotations in YOLO / YOLOv8 / YOLO txt format.

## 5. Model Training

After placing the images and labels in the dataset folders, run:

```bash
python train.py
```

The default model is `yolov8n.pt`, which is fast and suitable for validating the workflow first. Training outputs are saved under:

```text
runs/detect/five_objects/
```

The trained model is usually located at:

```text
runs/detect/five_objects/weights/best.pt
```

## 6. Prediction Test

Run prediction on one image or a folder:

```bash
python predict.py --source path/to/image_or_folder --weights runs/detect/three_objects/weights/best.pt
```

Prediction results are saved under `runs/detect/predict/`.

## 7. Troubleshooting

If training reports that no images were found, check that images are placed under `dataset/images/train/` and `dataset/images/val/`.

If training reports missing labels, make sure every image has a matching `.txt` file. Empty images may use an empty `.txt` file.

If detection quality is poor, the issue is usually the dataset rather than the code. First improve the number of images, annotation quality, background diversity, and lighting variation.