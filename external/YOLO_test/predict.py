import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "runs" / "detect" / "three_objects" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction.")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained YOLO weights.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image, video, camera index, or folder to predict.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="Save YOLO txt predictions under runs/detect/predict*/labels.",
    )
    parser.add_argument(
        "--save-conf",
        action="store_true",
        help="Include confidence values in saved prediction txt files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )

    for result in results:
        print(f"\nimage: {result.path}")
        if result.boxes is None or len(result.boxes) == 0:
            print("  no detections")
            continue

        names = result.names
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        conf = result.boxes.conf.cpu().numpy()
        for i, (box, class_id, score) in enumerate(zip(xyxy, cls, conf), start=1):
            x1, y1, x2, y2 = box
            class_name = names.get(class_id, str(class_id))
            print(
                f"  {i}: {class_name} conf={score:.3f} "
                f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )


if __name__ == "__main__":
    main()
