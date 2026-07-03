# This Python file uses the following encoding: utf-8
from __future__ import annotations

import argparse
import shutil

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train and export Hyakkiyakou patch detector")
    parser.add_argument("--data", required=True, help="YOLO data.yaml path")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default="ram", help="Ultralytics image cache mode: ram / disk / true / false")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="hya_patch")
    parser.add_argument("--output", required=True, help="Destination ONNX path")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(
            "缺少训练依赖。请先在训练 Python 环境安装：python -m pip install ultralytics onnx"
        ) from exc

    data_path = Path(args.data)
    project_dir = Path(args.project)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)

    print(f"data: {data_path}")
    print(f"base model: {args.model}")
    print(f"project: {project_dir}")
    print(f"output: {output_path}")

    model = YOLO(args.model)
    train_result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=args.cache,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
    )

    save_dir = Path(getattr(train_result, "save_dir", project_dir / args.name))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise SystemExit(f"训练完成但没有找到 best.pt: {best_pt}")

    print(f"best: {best_pt}")
    trained = YOLO(str(best_pt))
    export_result = trained.export(format="onnx", imgsz=args.imgsz, opset=12)
    exported = Path(export_result)
    if not exported.exists():
        exported = best_pt.with_suffix(".onnx")
    if not exported.exists():
        raise SystemExit(f"导出完成但没有找到 ONNX: {export_result}")

    shutil.copy2(exported, output_path)
    print(f"onnx: {exported}")
    print(f"copied: {output_path}")
    print("done")


if __name__ == "__main__":
    main()
