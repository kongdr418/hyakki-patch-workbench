# This Python file uses the following encoding: utf-8
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


WORKBENCH_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_FILE = WORKBENCH_ROOT / "config.local.json"
DEFAULT_OAS_ROOT = WORKBENCH_ROOT.parent / "OnmyojiAutoScript-easy-install"


def load_workbench_config() -> dict:
    if not LOCAL_CONFIG_FILE.exists():
        return {}
    try:
        with LOCAL_CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


WORKBENCH_CONFIG = load_workbench_config()


def is_oas_root(path: Path) -> bool:
    return (
        (path / "toolkit" / "python.exe").exists()
        and (path / "module" / "config" / "config.py").exists()
        and (path / "tasks" / "Hyakkiyakou").exists()
    )


def discover_oas_root() -> Path:
    explicit = [
        os.environ.get("HYAKKI_OAS_ROOT"),
        WORKBENCH_CONFIG.get("oas_root"),
    ]
    candidates = [Path(path).expanduser() for path in explicit if path]
    candidates.append(DEFAULT_OAS_ROOT)
    try:
        for child in WORKBENCH_ROOT.parent.iterdir():
            if child.is_dir():
                candidates.append(child)
    except OSError:
        pass

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if is_oas_root(resolved):
            return resolved
    return candidates[0].resolve() if candidates else DEFAULT_OAS_ROOT.resolve()


OAS_ROOT = discover_oas_root()
DEFAULT_OAS_CONFIG = str(
    os.environ.get("HYAKKI_OAS_CONFIG")
    or WORKBENCH_CONFIG.get("config_name")
    or ""
).strip()
if str(OAS_ROOT) not in sys.path:
    sys.path.insert(0, str(OAS_ROOT))
if OAS_ROOT.exists():
    os.chdir(OAS_ROOT)

from oashya import labels as legacy_labels

try:
    from oashya.tracker import Tracker as LegacyTracker
    LEGACY_IMPORT_ERROR = ""
except Exception as exc:
    LegacyTracker = None
    LEGACY_IMPORT_ERROR = str(exc)


PROJECT_ROOT = WORKBENCH_ROOT
HYAKKI_DIR = WORKBENCH_ROOT
STATIC_DIR = HYAKKI_DIR / "collector_static"
DEFAULT_DATASET_ROOT = HYAKKI_DIR / "datasets" / "patch"
MODELS_DIR = HYAKKI_DIR / "models"
PATCH_LABELS_FILE = MODELS_DIR / "hya_patch_labels.json"
PATCH_MODEL_FILE = MODELS_DIR / "hya_patch_fp32.onnx"
TRAIN_SCRIPT = HYAKKI_DIR / "train_patch_model.py"
TRAIN_RUNS_DIR = HYAKKI_DIR / "runs"
PATCH_PT_MODEL_FILE = TRAIN_RUNS_DIR / "hya_patch" / "weights" / "best.pt"
TRAIN_LOG_FILE = TRAIN_RUNS_DIR / "hya_patch_train.log"
INSTALL_LOG_FILE = TRAIN_RUNS_DIR / "hya_patch_install.log"
VENV_TRAIN_PYTHON = PROJECT_ROOT / ".venv-yolo" / "Scripts" / "python.exe"
VENV_SITE_PACKAGES = PROJECT_ROOT / ".venv-yolo" / "Lib" / "site-packages"
OAS_TRAIN_PYTHON = OAS_ROOT / "toolkit" / "python.exe"
DEFAULT_TRAIN_PYTHON = VENV_TRAIN_PYTHON
CUDA_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
TSINGHUA_PYPI_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_DOWNLOAD_TIMEOUT = "300"
PIP_RETRIES = "10"
ENV_CHECK_TIMEOUT = 120
ENV_CHECK_CACHE_SECONDS = 300

RARITIES = ("buff", "sp", "ssr", "sr", "r", "n", "g")
LEGACY_MAP = {
    "buff": legacy_labels.buff,
    "sp": legacy_labels.sp,
    "ssr": legacy_labels.ssr,
    "sr": legacy_labels.sr,
    "r": legacy_labels.r,
    "n": legacy_labels.n,
    "g": legacy_labels.g,
}
LEGACY_MAX_ID = legacy_labels.CLASSINDEX.MAX_SP

_legacy_tracker: LegacyTracker | None = None
_legacy_tracker_args: tuple[float, float] | None = None
_legacy_lock = threading.Lock()
_patch_model = None
_patch_model_mtime: float | None = None
_patch_model_path: Path | None = None
_patch_lock = threading.Lock()
_train_process: subprocess.Popen | None = None
_train_command: list[str] = []
_train_started_at: float | None = None
_install_process: subprocess.Popen | None = None
_install_command: list[str] = []
_install_started_at: float | None = None
_environment_cache: dict[str, tuple[float, dict]] = {}


class Store:
    def __init__(self, root: Path = DEFAULT_DATASET_ROOT):
        self.root = root
        self.ensure()

    @property
    def classes_path(self) -> Path:
        return self.root / "classes.json"

    def set_root(self, root: str | Path):
        path = Path(root)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        self.root = path
        self.ensure()

    def ensure(self):
        for split in ("train", "val"):
            (self.root / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.root / "labels" / split).mkdir(parents=True, exist_ok=True)
        if not self.classes_path.exists():
            self.save_classes(self._initial_classes())

    def _initial_classes(self) -> list[dict]:
        if PATCH_LABELS_FILE.exists():
            with PATCH_LABELS_FILE.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                raw = raw.get("labels", [])
            return [normalize_class_item(item) for item in raw]
        return []

    def load_classes(self) -> list[dict]:
        self.ensure()
        with self.classes_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return [normalize_class_item(item) for item in raw]

    def save_classes(self, classes: list[dict]):
        self.root.mkdir(parents=True, exist_ok=True)
        with self.classes_path.open("w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=2)


store = Store()
app = FastAPI(title="Hyakkiyakou Collector", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SettingsIn(BaseModel):
    root: str


class ClassIn(BaseModel):
    rarity: Literal["buff", "sp", "ssr", "sr", "r", "n", "g"]
    name: str = Field(min_length=1)
    label: str | None = None
    id: int | None = None


class CaptureIn(BaseModel):
    config_name: str = ""
    split: Literal["train", "val"] = "train"
    prefix: Literal["cap", "rec"] = "cap"


class RecordIn(CaptureIn):
    seconds: float = Field(default=5, ge=0.1, le=120)
    interval: float = Field(default=0.3, ge=0.05, le=10)


class VideoIn(BaseModel):
    path: str
    split: Literal["train", "val"] = "train"
    interval: float = Field(default=0.2, ge=0.01, le=30)
    max_frames: int = Field(default=0, ge=0)


class TrainStartIn(BaseModel):
    python: str | None = None
    model: str = "yolov8n.pt"
    epochs: int = Field(default=120, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=64, le=2048)
    batch: int = Field(default=16, ge=-1, le=256)
    device: str = "cpu"
    workers: int = Field(default=4, ge=0, le=16)
    cache: str = Field(default="ram")
    name: str = "hya_patch"
    force: bool = False


class TrainDepsIn(BaseModel):
    python: str | None = None
    mode: Literal["default", "cuda"] = "default"


class LegacyDetectIn(BaseModel):
    image: str
    source: Literal["legacy", "patch", "both"] = "legacy"
    conf_threshold: float = Field(default=0.25, ge=0.01, le=1)
    iou_threshold: float = Field(default=0.7, ge=0.01, le=1)


class LegacyBatchDetectIn(BaseModel):
    split: Literal["train", "val"] = "train"
    images: list[str] | None = None
    source: Literal["legacy", "patch", "both"] = "legacy"
    conf_threshold: float = Field(default=0.25, ge=0.01, le=1)
    iou_threshold: float = Field(default=0.7, ge=0.01, le=1)


class BoxIn(BaseModel):
    label: str
    x: float
    y: float
    w: float
    h: float


class AnnotationIn(BaseModel):
    image: str
    boxes: list[BoxIn]


class MoveFramesIn(BaseModel):
    images: list[str] = Field(min_length=1)
    target: Literal["train", "val"]


class DeleteFramesIn(BaseModel):
    images: list[str] = Field(min_length=1)


def normalize_class_item(item: dict) -> dict:
    label = item.get("label") or item.get("class")
    if not label:
        raise HTTPException(status_code=400, detail=f"Missing label in {item!r}")
    rarity = (item.get("rarity") or str(label).split("_", 1)[0]).lower()
    if rarity not in RARITIES:
        raise HTTPException(status_code=400, detail=f"Invalid rarity: {rarity}")
    return {
        "id": int(item.get("id", item.get("class_id", 0))),
        "label": str(label),
        "name": str(item.get("name", label)),
        "rarity": rarity,
    }


def next_label(rarity: str, classes: list[dict]) -> str:
    max_index = 0
    for label in LEGACY_MAP[rarity].keys():
        max_index = max(max_index, int(label.rsplit("_", 1)[1]))
    for item in classes:
        label = item["label"]
        if label.startswith(f"{rarity}_"):
            try:
                max_index = max(max_index, int(label.rsplit("_", 1)[1]))
            except ValueError:
                pass
    return f"{rarity}_{max_index + 1:03d}"


def next_class_id(classes: list[dict]) -> int:
    max_id = LEGACY_MAX_ID
    for item in classes:
        max_id = max(max_id, int(item["id"]))
    return max_id + 1


def legacy_id_for_label(label: str) -> int | None:
    for class_id in range(LEGACY_MAX_ID + 1):
        try:
            if legacy_labels.id2label(class_id) == label:
                return class_id
        except Exception:
            continue
    return None


def class_item_for_label(label: str) -> dict | None:
    rarity = label.split("_", 1)[0].lower() if "_" in label else ""
    if rarity not in LEGACY_MAP or label not in LEGACY_MAP[rarity]:
        return None
    class_id = legacy_id_for_label(label)
    if class_id is None:
        return None
    return {
        "id": class_id,
        "label": label,
        "name": LEGACY_MAP[rarity][label],
        "rarity": rarity,
    }


def oas_config_names() -> list[str]:
    config_dir = OAS_ROOT / "config"
    if not config_dir.exists():
        return []
    excluded = {"template"}
    names = []
    for path in config_dir.glob("*.json"):
        if path.stem.lower() in excluded:
            continue
        names.append(path.stem)
    return sorted(names, key=str.lower)


def default_oas_config_name() -> str:
    if DEFAULT_OAS_CONFIG:
        return DEFAULT_OAS_CONFIG
    names = oas_config_names()
    if "oas1" in names:
        return "oas1"
    return names[0] if names else ""


def resolve_oas_config_name(config_name: str | None) -> str:
    name = (config_name or "").strip()
    if name:
        return name
    name = default_oas_config_name()
    if not name:
        raise HTTPException(
            status_code=400,
            detail=f"No OAS config found in {OAS_ROOT / 'config'}",
        )
    return name


def ensure_annotation_classes(classes: list[dict], labels: list[str]) -> tuple[list[dict], list[dict]]:
    existing = {item["label"] for item in classes}
    added = []
    for label in labels:
        if label in existing:
            continue
        item = class_item_for_label(label)
        if item is None:
            raise HTTPException(status_code=400, detail=f"Unknown label: {label}")
        classes.append(item)
        existing.add(label)
        added.append(item)
    if added:
        store.save_classes(classes)
    return classes, added


def image_path(image_rel: str) -> tuple[Path, str]:
    rel = Path(image_rel)
    parts = rel.parts
    if len(parts) != 2 or parts[0] not in ("train", "val"):
        raise HTTPException(status_code=400, detail="Image path must be train/name.png or val/name.png")
    path = store.root / "images" / parts[0] / parts[1]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_rel}")
    return path, parts[0]


def label_path_for_image(image_rel: str) -> Path:
    img, split = image_path(image_rel)
    return store.root / "labels" / split / f"{img.stem}.txt"


def get_legacy_tracker(conf_threshold: float, iou_threshold: float) -> LegacyTracker:
    global _legacy_tracker, _legacy_tracker_args, LEGACY_IMPORT_ERROR

    if LegacyTracker is None:
        raise HTTPException(
            status_code=500,
            detail=f"OAS 原模型暂不可用，onnxruntime 导入失败: {LEGACY_IMPORT_ERROR}",
        )

    args_key = (round(conf_threshold, 4), round(iou_threshold, 4))
    if _legacy_tracker is not None and _legacy_tracker_args == args_key:
        return _legacy_tracker

    args = {
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "precision": "fp32",
        "inference_engine": "onnxruntime",
        "debug": False,
    }
    try:
        _legacy_tracker = LegacyTracker(args=args)
    except Exception as exc:
        LEGACY_IMPORT_ERROR = str(exc)
        raise HTTPException(
            status_code=500,
            detail=f"OAS 原模型初始化失败: {exc}",
        ) from exc
    _legacy_tracker_args = args_key
    return _legacy_tracker


def legacy_rarity(class_id: int) -> str:
    try:
        label = legacy_labels.id2label(class_id)
    except Exception:
        return "unknown"
    return str(label).split("_", 1)[0]


def detect_legacy_image(path: Path, conf_threshold: float, iou_threshold: float) -> list[dict]:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {path.name}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    with _legacy_lock:
        tracker = get_legacy_tracker(conf_threshold, iou_threshold)
        tracker.clear_tracks()
        raw_detections = tracker.detect(image_rgb)

    detections = []
    for index, detection in enumerate(raw_detections):
        if len(detection) == 3:
            class_id, conf, box = detection
            cx, cy, box_w, box_h = box
            track_id = index
            velocity = 0.0
        elif len(detection) == 8:
            track_id, class_id, conf, cx, cy, box_w, box_h, velocity = detection
        else:
            continue
        x1 = max(0.0, min(float(width), float(cx) - float(box_w) / 2))
        y1 = max(0.0, min(float(height), float(cy) - float(box_h) / 2))
        x2 = max(0.0, min(float(width), float(cx) + float(box_w) / 2))
        y2 = max(0.0, min(float(height), float(cy) + float(box_h) / 2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "track_id": int(track_id),
            "class_id": int(class_id),
            "label": legacy_labels.id2label(int(class_id)),
            "name": legacy_labels.id2name(int(class_id)),
            "rarity": legacy_rarity(int(class_id)),
            "source": "legacy",
            "conf": float(conf),
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "velocity": float(velocity),
        })
    detections.sort(key=lambda item: item["conf"], reverse=True)
    return detections


def load_patch_classes() -> list[dict]:
    labels_path = PATCH_LABELS_FILE if PATCH_LABELS_FILE.exists() else store.classes_path
    if not labels_path.exists():
        return []
    with labels_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("labels", [])
    return [normalize_class_item(item) for item in raw]


def patch_predict_model_file() -> Path | None:
    if PATCH_PT_MODEL_FILE.exists():
        return PATCH_PT_MODEL_FILE
    if PATCH_MODEL_FILE.exists():
        return PATCH_MODEL_FILE
    return None


def get_patch_model():
    global _patch_model, _patch_model_mtime, _patch_model_path

    model_path = patch_predict_model_file()
    if model_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"还没有生成训练模型: {PATCH_PT_MODEL_FILE}",
        )

    model_mtime = model_path.stat().st_mtime
    if (
        _patch_model is not None
        and _patch_model_mtime == model_mtime
        and _patch_model_path == model_path
    ):
        return _patch_model

    try:
        if VENV_SITE_PACKAGES.exists() and str(VENV_SITE_PACKAGES) not in sys.path:
            sys.path.insert(0, str(VENV_SITE_PACKAGES))
        from ultralytics import YOLO
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="缺少 ultralytics，先在训练面板安装独立训练依赖",
        ) from exc

    _patch_model = YOLO(str(model_path))
    _patch_model_mtime = model_mtime
    _patch_model_path = model_path
    return _patch_model


def patch_class_item(class_index: int, classes: list[dict], names: dict | list | None) -> dict:
    if 0 <= class_index < len(classes):
        return classes[class_index]
    fallback_label = str(class_index)
    if isinstance(names, dict):
        fallback_label = str(names.get(class_index, names.get(str(class_index), fallback_label)))
    elif isinstance(names, list) and 0 <= class_index < len(names):
        fallback_label = str(names[class_index])
    rarity = fallback_label.split("_", 1)[0] if "_" in fallback_label else "unknown"
    return {
        "id": class_index,
        "label": fallback_label,
        "name": fallback_label,
        "rarity": rarity,
    }


def detect_patch_image(path: Path, conf_threshold: float, iou_threshold: float) -> list[dict]:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {path.name}")

    classes = load_patch_classes()
    with _patch_lock:
        model = get_patch_model()
        try:
            results = model.predict(
                source=image_bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"训练模型识别失败: {exc}") from exc

    detections = []
    if not results:
        return detections

    names = getattr(model, "names", None)
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return detections

    for index, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            class_index = int(box.cls[0])
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        item = patch_class_item(class_index, classes, names)
        detections.append({
            "track_id": index,
            "class_id": int(item["id"]),
            "label": item["label"],
            "name": item["name"],
            "rarity": item["rarity"],
            "source": "patch",
            "conf": conf,
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "velocity": 0.0,
        })
    detections.sort(key=lambda item: item["conf"], reverse=True)
    return detections


def detect_reference_image(path: Path, source: str, conf_threshold: float, iou_threshold: float) -> tuple[list[dict], list[str]]:
    if source == "legacy":
        return detect_legacy_image(path, conf_threshold, iou_threshold), []
    if source == "patch":
        return detect_patch_image(path, conf_threshold, iou_threshold), []
    if source == "both":
        detections = []
        warnings = []
        for label, detector in (
            ("OAS 原模型", detect_legacy_image),
            ("训练模型", detect_patch_image),
        ):
            try:
                detections.extend(detector(path, conf_threshold, iou_threshold))
            except HTTPException as exc:
                warnings.append(f"{label}失败: {exc.detail}")
        detections.sort(key=lambda item: item["conf"], reverse=True)
        return detections, warnings
    raise HTTPException(status_code=400, detail=f"Unknown detect source: {source}")


def split_stats(split: str) -> dict:
    image_dir = store.root / "images" / split
    label_dir = store.root / "labels" / split
    images = sorted(image_dir.glob("*.png"))
    labeled_images = 0
    boxes = 0
    for image_file in images:
        label_file = label_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            continue
        lines = [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            labeled_images += 1
            boxes += len(lines)
    return {
        "images": len(images),
        "labeled_images": labeled_images,
        "boxes": boxes,
    }


def dataset_training_stats() -> dict:
    classes = store.load_classes()
    train = split_stats("train")
    val = split_stats("val")
    warnings = []
    if not classes:
        warnings.append("还没有导出任何式神标签")
    if train["boxes"] < 20:
        warnings.append("train 标注框偏少，建议至少 30-50 个框再训练")
    if val["boxes"] < 5:
        warnings.append("val 验证框偏少，建议至少 5-10 个框")
    return {
        "classes": classes,
        "train": train,
        "val": val,
        "warnings": warnings,
        "ready": bool(classes) and train["boxes"] >= 20 and val["boxes"] >= 5,
    }


def clean_python_path(path: str | None) -> str:
    if path:
        return path.strip().strip('"')
    env_python = os.environ.get("HYAKKI_TRAIN_PYTHON")
    if env_python:
        return env_python
    return str(DEFAULT_TRAIN_PYTHON)


def executable_exists(path: str) -> bool:
    return Path(path).exists() or shutil.which(path) is not None


def python_version_info(path: str | Path) -> dict:
    if not executable_exists(str(path)):
        return {"exists": False, "version": "", "major": 0, "minor": 0, "ok310": False}
    code = (
        "import json,sys;"
        "print(json.dumps({'version': sys.version, 'major': sys.version_info.major, "
        "'minor': sys.version_info.minor}, ensure_ascii=False))"
    )
    try:
        result = subprocess.run(
            [str(path), "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return {"exists": True, "version": str(exc), "major": 0, "minor": 0, "ok310": False}
    if result.returncode != 0:
        return {
            "exists": True,
            "version": (result.stderr or result.stdout).strip(),
            "major": 0,
            "minor": 0,
            "ok310": False,
        }
    data = json.loads(result.stdout)
    data["exists"] = True
    data["ok310"] = data.get("major") == 3 and data.get("minor") == 10
    return data


def ensure_train_venv(python_path: str) -> str:
    if not same_path(python_path, VENV_TRAIN_PYTHON):
        return python_path
    if VENV_TRAIN_PYTHON.exists():
        version = python_version_info(VENV_TRAIN_PYTHON)
        if not version["ok310"]:
            raise HTTPException(status_code=400, detail=f"训练虚拟环境不是 Python 3.10: {version['version']}")
        return str(VENV_TRAIN_PYTHON)

    seed = OAS_TRAIN_PYTHON
    version = python_version_info(seed)
    if not version["ok310"]:
        raise HTTPException(status_code=400, detail=f"没有可用的 Python 3.10 来创建训练环境: {version['version']}")

    venv_dir = PROJECT_ROOT / ".venv-yolo"
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(seed), "-m", "venv", str(venv_dir)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "创建训练虚拟环境失败"
        raise HTTPException(status_code=500, detail=detail)
    return str(VENV_TRAIN_PYTHON)


def nvidia_gpus() -> list[dict]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=8,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": int(parts[2]),
                "memory_used_mb": int(parts[3]),
            })
        except ValueError:
            continue
    return gpus


def empty_torch_info() -> dict:
    return {
        "version": "",
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }


def check_training_environment(path: str) -> dict:
    modules = ["ultralytics", "torch", "onnx"]
    try:
        cache_key = str(Path(path).resolve())
    except OSError:
        cache_key = path
    cached = _environment_cache.get(cache_key)
    if cached and time.time() - cached[0] < ENV_CHECK_CACHE_SECONDS:
        return cached[1]

    def finish(data: dict) -> dict:
        _environment_cache[cache_key] = (time.time(), data)
        return data

    hardware_gpus = nvidia_gpus()
    if not executable_exists(path):
        return finish({
            "python": path,
            "exists": False,
            "ok": False,
            "modules": {module: False for module in modules},
            "torch": empty_torch_info(),
            "gpu": {
                "hardware": hardware_gpus,
                "usable": False,
                "reason": "Python 不存在",
            },
            "error": "Python 不存在",
        })
    code = (
        "import importlib.util,json,sys;"
        "mods=['ultralytics','torch','onnx'];"
        "modules={m: importlib.util.find_spec(m) is not None for m in mods};"
        "torch_info={'version':'','cuda_available':False,'cuda_version':None,'device_count':0,'devices':[]};"
        "exec(\"if modules['torch']:\\n"
        "    import torch\\n"
        "    torch_info.update({'version': torch.__version__, 'cuda_available': torch.cuda.is_available(), "
        "'cuda_version': torch.version.cuda, 'device_count': torch.cuda.device_count(), "
        "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})\");"
        "print(json.dumps({'python':sys.executable,'modules':modules,'torch':torch_info}, ensure_ascii=False))"
    )
    try:
        result = subprocess.run(
            [path, "-c", code],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=ENV_CHECK_TIMEOUT,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return finish({
            "python": path,
            "exists": True,
            "ok": False,
            "modules": {module: False for module in modules},
            "torch": empty_torch_info(),
            "gpu": {
                "hardware": hardware_gpus,
                "usable": False,
                "reason": str(exc),
            },
            "error": str(exc),
        })
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip()
        return finish({
            "python": path,
            "exists": True,
            "ok": False,
            "modules": {module: False for module in modules},
            "torch": empty_torch_info(),
            "gpu": {
                "hardware": hardware_gpus,
                "usable": False,
                "reason": error,
            },
            "error": error,
        })
    data = json.loads(result.stdout)
    installed = data["modules"]
    torch_info = data.get("torch") or empty_torch_info()
    gpu_usable = bool(hardware_gpus and torch_info.get("cuda_available") and torch_info.get("device_count", 0) > 0)
    if gpu_usable:
        gpu_reason = ""
    elif hardware_gpus and installed.get("torch"):
        gpu_reason = "检测到 NVIDIA 显卡，但当前 PyTorch 是 CPU 版或 CUDA 不可用"
    elif hardware_gpus:
        gpu_reason = "检测到 NVIDIA 显卡，但缺少 PyTorch"
    else:
        gpu_reason = "未检测到 NVIDIA 显卡"
    return finish({
        "python": data["python"],
        "exists": True,
        "ok": all(installed.values()),
        "modules": installed,
        "torch": torch_info,
        "gpu": {
            "hardware": hardware_gpus,
            "usable": gpu_usable,
            "reason": gpu_reason,
        },
        "error": "",
    })


def same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


def training_commands(python_path: str, device: str = "cpu") -> dict:
    prepare = []
    if same_path(python_path, VENV_TRAIN_PYTHON):
        prepare.append(f"{OAS_TRAIN_PYTHON} -m venv {PROJECT_ROOT / '.venv-yolo'}")
    prepare.append(
        f"{python_path} -m pip install -i {TSINGHUA_PYPI_INDEX_URL} "
        f"--timeout {PIP_DOWNLOAD_TIMEOUT} --retries {PIP_RETRIES} ultralytics torch onnx"
    )
    cuda_prepare = [
        f"{python_path} -m pip install --upgrade --force-reinstall "
        f"--timeout {PIP_DOWNLOAD_TIMEOUT} --retries {PIP_RETRIES} "
        f"--no-deps torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url {CUDA_TORCH_INDEX_URL}"
    ]

    train_command = [
        python_path,
        str(TRAIN_SCRIPT),
        "--data", str(store.root / "data.yaml"),
        "--model", "yolov8n.pt",
        "--epochs", "120",
        "--imgsz", "640",
        "--batch", "16",
        "--device", device,
        "--workers", "4",
        "--cache", "ram",
    ]
    return {
        "prepare": prepare,
        "cuda_prepare": cuda_prepare,
        "train": " ".join(train_command),
    }


def train_log_tail(max_lines: int = 80) -> str:
    if not TRAIN_LOG_FILE.exists():
        return ""
    lines = TRAIN_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def install_log_tail(max_lines: int = 80) -> str:
    if not INSTALL_LOG_FILE.exists():
        return ""
    lines = INSTALL_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def current_train_job() -> dict:
    global _train_process

    if _train_process is None:
        return {
            "running": False,
            "exit_code": None,
            "command": _train_command,
            "started_at": _train_started_at,
        }
    exit_code = _train_process.poll()
    return {
        "running": exit_code is None,
        "exit_code": exit_code,
        "command": _train_command,
        "started_at": _train_started_at,
    }


def current_install_job() -> dict:
    global _install_process

    if _install_process is None:
        return {
            "running": False,
            "exit_code": None,
            "command": _install_command,
            "started_at": _install_started_at,
        }
    exit_code = _install_process.poll()
    return {
        "running": exit_code is None,
        "exit_code": exit_code,
        "command": _install_command,
        "started_at": _install_started_at,
    }


def training_status(path: str | None = None, device: str = "cpu") -> dict:
    python_path = clean_python_path(path)
    return {
        "dataset": dataset_training_stats(),
        "environment": check_training_environment(python_path),
        "default_python": str(DEFAULT_TRAIN_PYTHON),
        "oas_python": str(OAS_TRAIN_PYTHON),
        "venv_python": str(VENV_TRAIN_PYTHON),
        "commands": training_commands(python_path, device),
        "job": current_train_job(),
        "install": current_install_job(),
        "log": train_log_tail(),
        "install_log": install_log_tail(),
        "model": {
            "path": str(PATCH_MODEL_FILE),
            "exists": PATCH_MODEL_FILE.exists(),
            "mtime": PATCH_MODEL_FILE.stat().st_mtime if PATCH_MODEL_FILE.exists() else None,
            "predict_path": str(patch_predict_model_file()) if patch_predict_model_file() else None,
            "predict_exists": patch_predict_model_file() is not None,
        },
    }


def save_rgb_image(image, split: str, prefix: str = "cap") -> str:
    name = f"{prefix}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.png"
    path = store.root / "images" / split / name
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return f"{split}/{name}"


def save_bgr_image(image, split: str, prefix: str = "frame") -> str:
    name = f"{prefix}_{datetime.now().strftime('%Y%m%dT%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.png"
    path = store.root / "images" / split / name
    cv2.imwrite(str(path), image)
    return f"{split}/{name}"


def read_annotations(image_rel: str, classes: list[dict]) -> list[dict]:
    path, _split = image_path(image_rel)
    image = cv2.imread(str(path))
    if image is None:
        return []
    height, width = image.shape[:2]
    label_path = label_path_for_image(image_rel)
    if not label_path.exists():
        return []
    result = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        class_index = int(float(fields[0]))
        if class_index < 0 or class_index >= len(classes):
            continue
        xc, yc, bw, bh = [float(v) for v in fields[1:]]
        result.append({
            "label": classes[class_index]["label"],
            "x": (xc - bw / 2) * width,
            "y": (yc - bh / 2) * height,
            "w": bw * width,
            "h": bh * height,
        })
    return result


def frame_item(path: Path, split: str, classes: list[dict]) -> dict:
    rel = f"{split}/{path.name}"
    return {
        "image": rel,
        "name": path.name,
        "split": split,
        "mtime": path.stat().st_mtime,
        "boxes": read_annotations(rel, classes),
    }


def move_frame_item(image_rel: str, target: str) -> dict | None:
    source_image, source_split = image_path(image_rel)
    if source_split == target:
        return None

    source_label = store.root / "labels" / source_split / f"{source_image.stem}.txt"
    target_image = store.root / "images" / target / source_image.name
    target_label = store.root / "labels" / target / f"{source_image.stem}.txt"
    if target_image.exists():
        raise HTTPException(status_code=409, detail=f"目标分组已有图片: {target}/{source_image.name}")
    if source_label.exists() and target_label.exists():
        raise HTTPException(status_code=409, detail=f"目标分组已有标注: {target}/{target_label.name}")

    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_image), str(target_image))
    if source_label.exists():
        shutil.move(str(source_label), str(target_label))
    return {
        "from": image_rel,
        "to": f"{target}/{target_image.name}",
    }


def delete_frame_item(image_rel: str) -> dict:
    image_file, split = image_path(image_rel)
    label_file = store.root / "labels" / split / f"{image_file.stem}.txt"
    image_file.unlink()
    label_deleted = False
    if label_file.exists():
        label_file.unlink()
        label_deleted = True
    return {
        "image": image_rel,
        "label_deleted": label_deleted,
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/state")
def state():
    classes = store.load_classes()
    return {
        "root": str(store.root),
        "classes": classes,
        "frames": {
            "train": len(list((store.root / "images" / "train").glob("*.png"))),
            "val": len(list((store.root / "images" / "val").glob("*.png"))),
        },
        "export": {
            "patch_labels": str(PATCH_LABELS_FILE),
            "data_yaml": str(store.root / "data.yaml"),
        },
        "patch_model": {
            "path": str(PATCH_MODEL_FILE),
            "exists": PATCH_MODEL_FILE.exists(),
            "mtime": PATCH_MODEL_FILE.stat().st_mtime if PATCH_MODEL_FILE.exists() else None,
            "predict_path": str(patch_predict_model_file()) if patch_predict_model_file() else None,
            "predict_exists": patch_predict_model_file() is not None,
        },
        "legacy_model": {
            "available": LegacyTracker is not None,
            "error": LEGACY_IMPORT_ERROR,
        },
        "oas": {
            "root": str(OAS_ROOT),
            "exists": OAS_ROOT.exists(),
            "config_dir": str(OAS_ROOT / "config"),
            "configs": oas_config_names(),
            "default_config": default_oas_config_name(),
            "toolkit_python": str(OAS_TRAIN_PYTHON),
            "toolkit_python_exists": OAS_TRAIN_PYTHON.exists(),
            "local_config": str(LOCAL_CONFIG_FILE),
        },
    }


@app.post("/api/settings")
def settings(payload: SettingsIn):
    store.set_root(payload.root)
    return state()


@app.get("/api/frames")
def frames(split: Literal["train", "val"] = "train"):
    classes = store.load_classes()
    files = sorted((store.root / "images" / split).glob("*.png"), key=lambda p: p.stat().st_mtime)
    return {"frames": [frame_item(path, split, classes) for path in files]}


@app.post("/api/frames/move")
def move_frames(payload: MoveFramesIn):
    moved = []
    for image_rel in payload.images:
        item = move_frame_item(image_rel, payload.target)
        if item is not None:
            moved.append(item)
    return {
        "target": payload.target,
        "moved": moved,
        "state": state(),
    }


@app.post("/api/frames/delete")
def delete_frames(payload: DeleteFramesIn):
    deleted = []
    for image_rel in payload.images:
        deleted.append(delete_frame_item(image_rel))
    return {
        "deleted": deleted,
        "state": state(),
    }


@app.get("/api/image")
def image(image: str = Query(...)):
    path, _split = image_path(image)
    return FileResponse(path)


@app.post("/api/legacy-detect")
def legacy_detect(payload: LegacyDetectIn):
    path, _split = image_path(payload.image)
    detections, warnings = detect_reference_image(
        path,
        payload.source,
        payload.conf_threshold,
        payload.iou_threshold,
    )
    return {
        "image": payload.image,
        "source": payload.source,
        "detections": detections,
        "warnings": warnings,
    }


@app.post("/api/legacy-detect-batch")
def legacy_detect_batch(payload: LegacyBatchDetectIn):
    if payload.images:
        image_items = []
        for image_rel in payload.images:
            path, split = image_path(image_rel)
            if split != payload.split:
                raise HTTPException(status_code=400, detail=f"{image_rel} is not in {payload.split}")
            image_items.append((image_rel, path))
    else:
        files = sorted((store.root / "images" / payload.split).glob("*.png"), key=lambda p: p.stat().st_mtime)
        image_items = [(f"{payload.split}/{path.name}", path) for path in files]

    predictions = {}
    warnings = []
    total = 0
    for image_rel, path in image_items:
        detections, image_warnings = detect_reference_image(
            path,
            payload.source,
            payload.conf_threshold,
            payload.iou_threshold,
        )
        predictions[image_rel] = detections
        warnings.extend(f"{image_rel}: {warning}" for warning in image_warnings)
        total += len(detections)
    return {
        "split": payload.split,
        "source": payload.source,
        "images": len(image_items),
        "detections": total,
        "predictions": predictions,
        "warnings": warnings,
    }


@app.post("/api/classes")
def add_class(payload: ClassIn):
    classes = store.load_classes()
    label = payload.label.strip() if payload.label else next_label(payload.rarity, classes)
    if label in LEGACY_MAP[payload.rarity]:
        raise HTTPException(status_code=400, detail=f"{label} is a legacy class")
    if any(item["label"] == label for item in classes):
        raise HTTPException(status_code=400, detail=f"{label} already exists")
    class_id = payload.id if payload.id is not None else next_class_id(classes)
    if class_id <= LEGACY_MAX_ID:
        raise HTTPException(status_code=400, detail=f"Patch class id must be greater than {LEGACY_MAX_ID}")
    if any(item["id"] == class_id for item in classes):
        raise HTTPException(status_code=400, detail=f"Class id {class_id} already exists")
    item = {"id": class_id, "label": label, "name": payload.name.strip(), "rarity": payload.rarity}
    classes.append(item)
    store.save_classes(classes)
    return {"classes": classes, "added": item}


def make_oas_device(config_name: str):
    from module.config.config import Config
    from module.device.device import Device

    config = Config(resolve_oas_config_name(config_name))
    return Device(config)


def oas_screenshot_with_retry(config_name: str, device=None, attempts: int = 2):
    last_error = None
    active_device = device
    for attempt in range(attempts):
        try:
            if active_device is None or attempt > 0:
                active_device = make_oas_device(config_name)
            return active_device.screenshot(), active_device
        except HTTPException:
            raise
        except Exception as exc:
            last_error = exc
            active_device = None
            if attempt < attempts - 1:
                time.sleep(1)
    raise HTTPException(
        status_code=502,
        detail=f"OAS截图失败，已重试{attempts}次：{type(last_error).__name__}: {last_error}",
    )


@app.post("/api/capture")
def capture(payload: CaptureIn):
    image, _device = oas_screenshot_with_retry(payload.config_name)
    rel = save_rgb_image(image, payload.split, prefix=payload.prefix)
    return {"created": [rel]}


@app.post("/api/record")
def record(payload: RecordIn):
    device = None
    created = []
    end_time = time.time() + payload.seconds
    while time.time() < end_time:
        image, device = oas_screenshot_with_retry(payload.config_name, device=device)
        created.append(save_rgb_image(image, payload.split, prefix="rec"))
        time.sleep(payload.interval)
    return {"created": created}


@app.post("/api/import-video")
def import_video(payload: VideoIn):
    video_path = Path(payload.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = max(1, int(round(fps * payload.interval)))
    count_frame = 0
    saved = 0
    created = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count_frame += 1
        if count_frame % frame_skip != 0:
            continue
        height, width = frame.shape[:2]
        if width != 1280 or height != 720:
            frame = cv2.resize(frame, (1280, 720))
        created.append(save_bgr_image(frame, payload.split, prefix=video_path.stem))
        saved += 1
        if payload.max_frames and saved >= payload.max_frames:
            break
    cap.release()
    return {"created": created}


@app.post("/api/annotations")
def save_annotations(payload: AnnotationIn):
    path, _split = image_path(payload.image)
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {payload.image}")
    height, width = image.shape[:2]
    classes = store.load_classes()
    classes, added_classes = ensure_annotation_classes(classes, [box.label for box in payload.boxes])
    class_index = {item["label"]: index for index, item in enumerate(classes)}
    lines = []
    for box in payload.boxes:
        if box.label not in class_index:
            raise HTTPException(status_code=400, detail=f"Unknown label: {box.label}")
        x1 = max(0, min(width, box.x))
        y1 = max(0, min(height, box.y))
        x2 = max(0, min(width, box.x + box.w))
        y2 = max(0, min(height, box.y + box.h))
        if x2 <= x1 or y2 <= y1:
            continue
        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{class_index[box.label]} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    label_path = label_path_for_image(payload.image)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return {
        "saved": payload.image,
        "boxes": read_annotations(payload.image, classes),
        "classes": classes,
        "added_classes": added_classes,
    }


@app.post("/api/export")
def export_dataset():
    classes = store.load_classes()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    patch_labels = [
        {"id": item["id"], "label": item["label"], "name": item["name"], "rarity": item["rarity"]}
        for item in classes
    ]
    with PATCH_LABELS_FILE.open("w", encoding="utf-8") as f:
        json.dump(patch_labels, f, ensure_ascii=False, indent=2)

    names = "\n".join(f"  {index}: {item['label']}" for index, item in enumerate(classes))
    data_yaml = (
        f"path: {store.root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n"
    )
    data_yaml_path = store.root / "data.yaml"
    data_yaml_path.write_text(data_yaml, encoding="utf-8")
    return {
        "patch_labels": str(PATCH_LABELS_FILE),
        "data_yaml": str(data_yaml_path),
        "classes": len(classes),
    }


@app.get("/api/train/status")
def train_status(python: str | None = Query(None), device: str = Query("cpu")):
    return training_status(python, device)


@app.post("/api/train/install-deps")
def train_install_deps(payload: TrainDepsIn):
    global _install_process, _install_command, _install_started_at

    if current_train_job()["running"]:
        raise HTTPException(status_code=409, detail="训练运行中，不能安装依赖")
    if current_install_job()["running"]:
        raise HTTPException(status_code=409, detail="依赖正在安装")

    requested_python = clean_python_path(payload.python)
    python_path = ensure_train_venv(requested_python)
    if not executable_exists(python_path):
        raise HTTPException(status_code=400, detail=f"Python 不存在: {python_path}")

    TRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if payload.mode == "cuda":
        _install_command = [
            python_path,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--timeout",
            PIP_DOWNLOAD_TIMEOUT,
            "--retries",
            PIP_RETRIES,
            "--no-deps",
            "torch==2.5.1+cu121",
            "torchvision==0.20.1+cu121",
            "--index-url",
            CUDA_TORCH_INDEX_URL,
        ]
    else:
        _install_command = [
            python_path,
            "-m",
            "pip",
            "install",
            "-i",
            TSINGHUA_PYPI_INDEX_URL,
            "--timeout",
            PIP_DOWNLOAD_TIMEOUT,
            "--retries",
            PIP_RETRIES,
            "ultralytics",
            "torch",
            "onnx",
        ]
    _install_started_at = time.time()
    INSTALL_LOG_FILE.write_text(
        "Hyakkiyakou training dependency install\n"
        f"python: {python_path}\n"
        f"mode: {payload.mode}\n"
        f"command: {' '.join(_install_command)}\n\n",
        encoding="utf-8",
    )
    log_handle = INSTALL_LOG_FILE.open("a", encoding="utf-8", errors="replace")
    try:
        _install_process = subprocess.Popen(
            _install_command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()

    return training_status(payload.python)


@app.post("/api/train/start")
def train_start(payload: TrainStartIn):
    global _train_process, _train_command, _train_started_at

    job = current_train_job()
    if job["running"]:
        raise HTTPException(status_code=409, detail="训练已经在运行")

    export_dataset()
    stats = dataset_training_stats()
    if stats["warnings"] and not payload.force:
        raise HTTPException(status_code=400, detail="；".join(stats["warnings"]))

    python_path = clean_python_path(payload.python)
    environment = check_training_environment(python_path)
    if not environment["ok"]:
        missing = [name for name, ok in environment["modules"].items() if not ok]
        detail = "训练环境缺少依赖"
        if missing:
            detail += f": {', '.join(missing)}"
        if environment["error"]:
            detail += f"\n{environment['error']}"
        raise HTTPException(status_code=400, detail=detail)
    if payload.device != "cpu" and not environment["gpu"]["usable"]:
        raise HTTPException(status_code=400, detail=environment["gpu"]["reason"] or "当前环境不能使用显卡训练")

    TRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _train_command = [
        environment["python"],
        str(TRAIN_SCRIPT),
        "--data", str(store.root / "data.yaml"),
        "--model", payload.model,
        "--epochs", str(payload.epochs),
        "--imgsz", str(payload.imgsz),
        "--batch", str(payload.batch),
        "--device", payload.device,
        "--workers", str(payload.workers),
        "--cache", payload.cache,
        "--project", str(TRAIN_RUNS_DIR),
        "--name", payload.name,
        "--output", str(PATCH_MODEL_FILE),
    ]
    _train_started_at = time.time()
    TRAIN_LOG_FILE.write_text(
        "Hyakkiyakou patch training\n"
        f"dataset: {store.root}\n"
        f"command: {' '.join(_train_command)}\n\n",
        encoding="utf-8",
    )
    log_handle = TRAIN_LOG_FILE.open("a", encoding="utf-8", errors="replace")
    try:
        _train_process = subprocess.Popen(
            _train_command,
            cwd=str(PROJECT_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_handle.close()

    return training_status(payload.python, payload.device)


@app.post("/api/train/stop")
def train_stop():
    global _train_process

    job = current_train_job()
    if not job["running"]:
        return training_status()
    assert _train_process is not None
    _train_process.terminate()
    return training_status()


def main():
    parser = argparse.ArgumentParser(description="Hyakkiyakou dataset collector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--root", default=str(DEFAULT_DATASET_ROOT))
    args = parser.parse_args()
    store.set_root(args.root)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
