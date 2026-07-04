# This Python file uses the following encoding: utf-8
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import struct
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


def save_workbench_config(updates: dict) -> dict:
    WORKBENCH_ROOT.mkdir(parents=True, exist_ok=True)
    config = load_workbench_config()
    config.update({key: value for key, value in updates.items() if value is not None})
    with LOCAL_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    WORKBENCH_CONFIG.clear()
    WORKBENCH_CONFIG.update(config)
    return config


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
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"
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
ENV_CHECK_FILE_CACHE_SECONDS = 24 * 60 * 60
ENV_CHECK_CACHE_FILE = TRAIN_RUNS_DIR / "environment_check_cache.json"
SAMPLE_DIGEST_VERSION = 2

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
_directory_picker_lock = threading.Lock()
_train_plan: dict | None = None


class Store:
    def __init__(self, root: Path = DEFAULT_DATASET_ROOT):
        self.root = root
        self.ensure()

    @property
    def classes_path(self) -> Path:
        return self.root / "classes.json"

    @property
    def training_manifest_path(self) -> Path:
        return self.root / "training_manifest.json"

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


class OasRootIn(BaseModel):
    oas_root: str = Field(min_length=1)


class DirectoryPickIn(BaseModel):
    title: str = "选择目录"
    initial: str | None = None


class ClassIn(BaseModel):
    rarity: Literal["buff", "sp", "ssr", "sr", "r", "n", "g"]
    name: str = Field(min_length=1)
    label: str | None = None
    id: int | None = None


class DeleteClassIn(BaseModel):
    label: str
    force: bool = False


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
    mode: Literal["full", "incremental"] = "full"
    epochs: int = Field(default=120, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=64, le=2048)
    batch: int = Field(default=16, ge=-1, le=256)
    device: str = "cpu"
    workers: int = Field(default=4, ge=0, le=16)
    cache: str = Field(default="ram")
    name: str = "hya_patch"
    force: bool = False
    archive_existing: bool = True


class TrainDepsIn(BaseModel):
    python: str | None = None
    mode: Literal["default", "cuda"] = "default"


class LegacyDetectIn(BaseModel):
    image: str
    source: Literal["legacy", "patch", "both"] = "legacy"
    conf_threshold: float = Field(default=0.25, ge=0.01, le=1)
    iou_threshold: float = Field(default=0.7, ge=0.01, le=1)
    patch_model_path: str | None = None
    patch_labels_path: str | None = None


class LegacyBatchDetectIn(BaseModel):
    split: Literal["train", "val"] = "train"
    images: list[str] | None = None
    source: Literal["legacy", "patch", "both"] = "legacy"
    conf_threshold: float = Field(default=0.25, ge=0.01, le=1)
    iou_threshold: float = Field(default=0.7, ge=0.01, le=1)
    patch_model_path: str | None = None
    patch_labels_path: str | None = None


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


def normalize_oas_root_path(path: str) -> Path:
    raw = path.strip().strip('"')
    if not raw:
        raise HTTPException(status_code=400, detail="OAS 根目录不能为空")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = WORKBENCH_ROOT / candidate
    return candidate.resolve()


def configured_oas_root() -> str:
    value = str(WORKBENCH_CONFIG.get("oas_root") or "").strip()
    if not value:
        return ""
    try:
        return str(normalize_oas_root_path(value))
    except HTTPException:
        return value
    except OSError:
        return value


def oas_restart_required(saved_root: Path | None = None) -> bool:
    root = saved_root
    if root is None:
        configured = configured_oas_root()
        if not configured:
            return False
        try:
            root = Path(configured).resolve()
        except OSError:
            return False
    try:
        return root.resolve() != OAS_ROOT.resolve()
    except OSError:
        return str(root) != str(OAS_ROOT)


def save_oas_root_config(oas_root: str) -> dict:
    root = normalize_oas_root_path(oas_root)
    if not is_oas_root(root):
        raise HTTPException(
            status_code=400,
            detail=f"不是有效的 OAS 根目录: {root}。需要包含 toolkit\\python.exe、module\\config\\config.py、tasks\\Hyakkiyakou",
        )
    save_workbench_config({"oas_root": str(root)})
    return {
        "saved_oas_root": str(root),
        "effective_oas_root": str(OAS_ROOT),
        "restart_required": oas_restart_required(root),
        "local_config": str(LOCAL_CONFIG_FILE),
    }


def pick_directory(title: str, initial: str | None = None) -> dict:
    if os.name != "nt":
        raise HTTPException(status_code=501, detail="当前仅支持 Windows 原生目录选择器")
    if not _directory_picker_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="已有目录选择窗口打开，请先完成或取消那个窗口")
    try:
        powershell = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        powershell_path = str(powershell) if powershell.exists() else "powershell.exe"
        title_b64 = base64.b64encode((title or "选择目录").encode("utf-8")).decode("ascii")
        initial_b64 = base64.b64encode((initial or "").encode("utf-8")).decode("ascii")
        script = r'''
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$title = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($args[0]))
$initial = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($args[1]))
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = $title
$dialog.ShowNewFolderButton = $true
if ($initial -and (Test-Path -LiteralPath $initial)) {
    $dialog.SelectedPath = (Resolve-Path -LiteralPath $initial).ProviderPath
}
$owner = New-Object System.Windows.Forms.Form
$owner.Text = $title
$owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
$owner.Size = New-Object System.Drawing.Size(1, 1)
$owner.ShowInTaskbar = $true
$owner.TopMost = $true
$owner.Opacity = 0
$owner.Show()
$owner.Activate()
$owner.BringToFront()
$result = $dialog.ShowDialog($owner)
$owner.Close()
$owner.Dispose()
if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
    Write-Output $dialog.SelectedPath
}
'''
        try:
            result = subprocess.run(
                [powershell_path, "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script, title_b64, initial_b64],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=180,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=408, detail="选择目录超时；如果没有看到弹窗，请用普通 PowerShell 启动工作台，或手动输入路径") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"无法打开目录选择器: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "目录选择器启动失败"
            raise HTTPException(status_code=500, detail=detail)
        selected = result.stdout.strip()
        if not selected:
            return {"path": "", "cancelled": True}
        return {"path": selected, "cancelled": False}
    finally:
        _directory_picker_lock.release()


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


def load_patch_classes(labels_path: Path | None = None) -> list[dict]:
    if labels_path is None:
        labels_path = PATCH_LABELS_FILE if PATCH_LABELS_FILE.exists() else store.classes_path
    if not labels_path.exists():
        return []
    with labels_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("labels", [])
    return [normalize_class_item(item) for item in raw]


def find_class_index(classes: list[dict], label: str) -> int:
    target = label.strip()
    for index, item in enumerate(classes):
        if item.get("label") == target:
            return index
    return -1


def scan_class_usage(class_id: int) -> list[dict]:
    affected: list[dict] = []
    for split in ("train", "val"):
        label_dir = store.root / "labels" / split
        if not label_dir.exists():
            continue
        for label_file in sorted(label_dir.glob("*.txt")):
            try:
                text = label_file.read_text(encoding="utf-8")
            except OSError:
                continue
            count = 0
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    if int(float(stripped.split()[0])) == class_id:
                        count += 1
                except (ValueError, IndexError):
                    continue
            if count:
                affected.append({
                    "image": f"{split}/{label_file.stem}.png",
                    "boxes": count,
                })
    return affected


def _strip_class_lines(text: str, removed_index: int) -> str:
    pattern = re.compile(rf"^{removed_index}(\s|$)", re.MULTILINE)
    lines = [line for line in text.splitlines() if not pattern.match(line)]
    return "\n".join(lines) + ("\n" if lines else "")


def _renumber_label_lines(text: str, removed_index: int) -> str:
    def _shift(match: re.Match) -> str:
        idx = int(match.group(1))
        if idx <= removed_index:
            return match.group(0)
        return f"{idx - 1}{match.group(2)}"

    return re.sub(r"^(\d+)(\s)", _shift, text, flags=re.MULTILINE)


def delete_class_with_renumber(label: str, force: bool) -> dict:
    classes = store.load_classes()
    index = find_class_index(classes, label)
    if index == -1:
        raise HTTPException(
            status_code=404,
            detail={"error": "class_not_found", "label": label},
        )
    item = classes[index]
    if int(item.get("id", 0)) <= LEGACY_MAX_ID or item.get("label") in LEGACY_MAP.get(item.get("rarity", ""), {}):
        raise HTTPException(
            status_code=400,
            detail={"error": "legacy_class", "label": label, "message": "legacy 类不可删除"},
        )

    affected = scan_class_usage(index)
    total_boxes = sum(entry["boxes"] for entry in affected)
    if affected and not force:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "class_in_use",
                "label": label,
                "affected": affected,
                "total_boxes": total_boxes,
            },
        )

    renumbered_files = 0
    for split in ("train", "val"):
        label_dir = store.root / "labels" / split
        if not label_dir.exists():
            continue
        for label_file in label_dir.glob("*.txt"):
            try:
                text = label_file.read_text(encoding="utf-8")
            except OSError:
                continue
            stripped = _strip_class_lines(text, index)
            new_text = _renumber_label_lines(stripped, index)
            if new_text != text:
                label_file.write_text(new_text, encoding="utf-8")
                renumbered_files += 1

    classes.pop(index)
    store.save_classes(classes)
    export_dataset()

    return {
        "label": label,
        "removed": True,
        "affected_images": len(affected),
        "stripped_boxes": total_boxes,
        "renumbered_files": renumbered_files,
        "exported": True,
        "classes": classes,
    }


def patch_predict_model_file() -> Path | None:
    if PATCH_PT_MODEL_FILE.exists():
        return PATCH_PT_MODEL_FILE
    if PATCH_MODEL_FILE.exists():
        return PATCH_MODEL_FILE
    return None


def get_patch_model(model_path: Path | None = None):
    global _patch_model, _patch_model_mtime, _patch_model_path

    if model_path is None:
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


def detect_patch_image(
    path: Path,
    conf_threshold: float,
    iou_threshold: float,
    patch_model_path: Path | None = None,
    patch_labels_path: Path | None = None,
) -> list[dict]:
    image_bgr = cv2.imread(str(path))
    if image_bgr is None:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {path.name}")

    classes = load_patch_classes(patch_labels_path)
    with _patch_lock:
        model = get_patch_model(patch_model_path)
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


def detect_reference_image(
    path: Path,
    source: str,
    conf_threshold: float,
    iou_threshold: float,
    patch_model_path: Path | None = None,
    patch_labels_path: Path | None = None,
) -> tuple[list[dict], list[str]]:
    if source == "legacy":
        return detect_legacy_image(path, conf_threshold, iou_threshold), []
    if source == "patch":
        return detect_patch_image(path, conf_threshold, iou_threshold, patch_model_path, patch_labels_path), []
    if source == "both":
        detections = []
        warnings = []
        for label, detector in (
            ("OAS 原模型", lambda p, c, i: detect_legacy_image(p, c, i)),
            ("训练模型", lambda p, c, i: detect_patch_image(p, c, i, patch_model_path, patch_labels_path)),
        ):
            try:
                detections.extend(detector(path, conf_threshold, iou_threshold))
            except HTTPException as exc:
                warnings.append(f"{label}失败: {exc.detail}")
        detections.sort(key=lambda item: item["conf"], reverse=True)
        return detections, warnings
    raise HTTPException(status_code=400, detail=f"Unknown detect source: {source}")


def timestamp_id() -> str:
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"


def empty_training_manifest() -> dict:
    return {
        "version": 1,
        "digest_version": SAMPLE_DIGEST_VERSION,
        "runs": [],
        "samples": {},
    }


def load_training_manifest() -> dict:
    path = store.training_manifest_path
    if not path.exists():
        return empty_training_manifest()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return empty_training_manifest()
    if not isinstance(data, dict):
        return empty_training_manifest()
    data.setdefault("version", 1)
    data.setdefault("runs", [])
    data.setdefault("samples", {})
    if migrate_training_manifest_digests(data):
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def save_training_manifest(manifest: dict):
    store.training_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with store.training_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def sample_digest_for_files(image_file: Path, label_file: Path) -> str:
    digest = hashlib.sha256()
    image_stat = image_file.stat()
    digest.update(f"{image_stat.st_size}:{image_stat.st_mtime_ns}".encode("ascii"))
    digest.update(b"\0labels\0")
    if label_file.exists():
        digest.update(label_file.read_bytes())
    return digest.hexdigest()


def legacy_sample_digest_for_files(image_file: Path, label_file: Path) -> str:
    digest = hashlib.sha256()
    digest.update(image_file.read_bytes())
    digest.update(b"\0labels\0")
    if label_file.exists():
        digest.update(label_file.read_bytes())
    return digest.hexdigest()


def migrate_training_manifest_digests(manifest: dict) -> bool:
    if int(manifest.get("digest_version") or 1) >= SAMPLE_DIGEST_VERSION:
        return False

    for image_rel, record in manifest.get("samples", {}).items():
        parts = Path(image_rel).parts
        if len(parts) != 2 or parts[0] not in ("train", "val") or not isinstance(record, dict):
            continue
        image_file = store.root / "images" / parts[0] / parts[1]
        label_file = store.root / "labels" / parts[0] / f"{image_file.stem}.txt"
        if not image_file.exists() or not label_file.exists():
            continue
        old_digest = record.get("digest")
        if old_digest and legacy_sample_digest_for_files(image_file, label_file) == old_digest:
            record["digest"] = sample_digest_for_files(image_file, label_file)

    manifest["digest_version"] = SAMPLE_DIGEST_VERSION
    return True


def sample_training_state(
    image_rel: str,
    image_file: Path | None = None,
    label_file: Path | None = None,
    manifest: dict | None = None,
) -> dict:
    if image_file is None or label_file is None:
        image_file, split = image_path(image_rel)
        label_file = store.root / "labels" / split / f"{image_file.stem}.txt"
    if not label_file.exists() or not label_file.read_text(encoding="utf-8").strip():
        return {"trained": False, "digest": "", "last_run": "", "trained_at": ""}

    digest = sample_digest_for_files(image_file, label_file)
    manifest = manifest or load_training_manifest()
    record = manifest.get("samples", {}).get(image_rel) or {}
    trained = record.get("digest") == digest
    return {
        "trained": trained,
        "digest": digest,
        "last_run": record.get("last_run", "") if trained else "",
        "trained_at": record.get("trained_at", "") if trained else "",
    }


def labeled_image_items(split: str, manifest: dict | None = None) -> list[dict]:
    image_dir = store.root / "images" / split
    label_dir = store.root / "labels" / split
    manifest = manifest or load_training_manifest()
    items = []
    for image_file in sorted(image_dir.glob("*.png"), key=lambda p: p.stat().st_mtime):
        label_file = label_dir / f"{image_file.stem}.txt"
        if not label_file.exists():
            continue
        lines = [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        rel = f"{split}/{image_file.name}"
        train_state = sample_training_state(rel, image_file, label_file, manifest)
        items.append({
            "image": rel,
            "path": image_file,
            "label_path": label_file,
            "boxes": len(lines),
            "digest": train_state["digest"],
            "trained": train_state["trained"],
        })
    return items


def write_training_subset_yaml(run_id: str, train_items: list[dict], val_items: list[dict], classes: list[dict]) -> Path:
    subset_dir = TRAIN_RUNS_DIR / "datasets" / run_id
    subset_dir.mkdir(parents=True, exist_ok=True)
    train_txt = subset_dir / "train.txt"
    val_txt = subset_dir / "val.txt"
    train_txt.write_text(
        "\n".join(item["path"].as_posix() for item in train_items) + ("\n" if train_items else ""),
        encoding="utf-8",
    )
    val_txt.write_text(
        "\n".join(item["path"].as_posix() for item in val_items) + ("\n" if val_items else ""),
        encoding="utf-8",
    )
    names = "\n".join(f"  {index}: {item['label']}" for index, item in enumerate(classes))
    data_yaml = (
        f"path: {store.root.as_posix()}\n"
        f"train: {train_txt.as_posix()}\n"
        f"val: {val_txt.as_posix()}\n"
        "names:\n"
        f"{names}\n"
    )
    data_yaml_path = subset_dir / "data.yaml"
    data_yaml_path.write_text(data_yaml, encoding="utf-8")
    return data_yaml_path


def prepare_training_dataset(mode: str, run_id: str) -> dict:
    classes = store.load_classes()
    manifest = load_training_manifest()
    train_items = labeled_image_items("train", manifest)
    val_items = labeled_image_items("val", manifest)
    selected_train = [item for item in train_items if not item["trained"]] if mode == "incremental" else train_items
    if not selected_train:
        detail = "没有可用于训练的已标注 train 图片"
        if mode == "incremental":
            detail = "没有新的已标注 train 图片；所有当前 train 标注都已经进入过模型训练"
        raise HTTPException(status_code=400, detail=detail)
    data_yaml_path = write_training_subset_yaml(run_id, selected_train, val_items, classes)
    return {
        "data_yaml": data_yaml_path,
        "classes": classes,
        "train_items": selected_train,
        "val_items": val_items,
        "all_train_items": train_items,
    }


def archive_current_model(reason: str = "before_train") -> dict | None:
    artifacts = [
        (PATCH_PT_MODEL_FILE, "best.pt"),
        (PATCH_PT_MODEL_FILE.with_name("last.pt"), "last.pt"),
        (PATCH_PT_MODEL_FILE.with_suffix(".onnx"), "best.onnx"),
        (PATCH_MODEL_FILE, PATCH_MODEL_FILE.name),
        (PATCH_LABELS_FILE, PATCH_LABELS_FILE.name),
        (store.root / "data.yaml", "data.yaml"),
        (store.classes_path, "classes.json"),
    ]
    existing = [(source, name) for source, name in artifacts if source.exists()]
    if not existing:
        return None

    archive_id = timestamp_id()
    target_dir = MODEL_VERSIONS_DIR / archive_id
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source, name in existing:
        target = target_dir / name
        shutil.copy2(source, target)
        copied.append({
            "source": str(source),
            "archive": str(target),
        })
    manifest = {
        "id": archive_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "files": copied,
    }
    with (target_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {
        "id": archive_id,
        "path": str(target_dir),
        "files": copied,
    }


def legacy_classes_for_picker() -> list[dict]:
    items: list[dict] = []
    for rarity in ("buff", "n", "g", "r", "sr", "ssr", "sp"):
        mapping = LEGACY_MAP.get(rarity) or {}
        for label, name in mapping.items():
            items.append({"id": legacy_id_for_label(label), "label": label, "name": name, "rarity": rarity})
    return items


def class_usage_stats(classes: list[dict]) -> dict[str, dict]:
    usage = {
        item["label"]: {"boxes": 0, "images": 0}
        for item in classes
        if item.get("label")
    }
    if not classes:
        return usage
    index_to_label = {index: item["label"] for index, item in enumerate(classes)}
    for split in ("train", "val"):
        label_dir = store.root / "labels" / split
        if not label_dir.exists():
            continue
        for label_file in label_dir.glob("*.txt"):
            labels_in_image: set[str] = set()
            try:
                lines = label_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                fields = line.split()
                if not fields:
                    continue
                try:
                    class_index = int(float(fields[0]))
                except ValueError:
                    continue
                label = index_to_label.get(class_index)
                if not label:
                    continue
                entry = usage.setdefault(label, {"boxes": 0, "images": 0})
                entry["boxes"] += 1
                labels_in_image.add(label)
            for label in labels_in_image:
                usage.setdefault(label, {"boxes": 0, "images": 0})["images"] += 1
    return usage


def latest_model_archive() -> dict | None:
    if not MODEL_VERSIONS_DIR.exists():
        return None
    archives = sorted([path for path in MODEL_VERSIONS_DIR.iterdir() if path.is_dir()])
    if not archives:
        return None
    latest = archives[-1]
    manifest_path = latest / "manifest.json"
    created_at = ""
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                created_at = (json.load(f) or {}).get("created_at", "")
        except Exception:
            created_at = ""
    return {
        "id": latest.name,
        "path": str(latest),
        "created_at": created_at,
    }


def list_model_archives() -> list[dict]:
    if not MODEL_VERSIONS_DIR.exists():
        return []
    archives: list[dict] = []
    for path in sorted(MODEL_VERSIONS_DIR.iterdir(), key=lambda p: p.name):
        if not path.is_dir():
            continue
        manifest_path = path / "manifest.json"
        created_at = ""
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    created_at = (json.load(f) or {}).get("created_at", "")
            except Exception:
                created_at = ""
        best_pt = path / "best.pt"
        onnx = path / "hya_patch_fp32.onnx"
        labels = path / "hya_patch_labels.json"
        archives.append({
            "id": path.name,
            "path": str(path),
            "created_at": created_at,
            "best_pt_path": str(best_pt) if best_pt.exists() else None,
            "onnx_path": str(onnx) if onnx.exists() else None,
            "labels_path": str(labels) if labels.exists() else None,
        })
    return archives


def _resolve_patch_paths(
    model_path_str: str | None,
    labels_path_str: str | None,
) -> tuple[Path | None, Path | None]:
    if not model_path_str and not labels_path_str:
        return None, None
    model_path = _resolve_local_model_file(model_path_str, "模型文件", {".pt", ".onnx"}) if model_path_str else None
    labels_path = _resolve_local_model_file(labels_path_str, "标签文件", {".json"}) if labels_path_str else None
    if model_path and not labels_path:
        candidate = model_path.parent / "hya_patch_labels.json"
        if candidate.exists():
            labels_path = candidate
    return model_path, labels_path


def _resolve_local_model_file(path_str: str, label: str, suffixes: set[str]) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"{label}不存在: {path}")
    resolved = path.resolve()
    allowed_roots = [MODELS_DIR.resolve(), TRAIN_RUNS_DIR.resolve()]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise HTTPException(
            status_code=400,
            detail=f"{label}必须位于工作台 models 或 runs 目录内: {resolved}",
        )
    if resolved.suffix.lower() not in suffixes:
        suffix_text = "、".join(sorted(suffixes))
        raise HTTPException(status_code=400, detail=f"{label}类型不支持: {resolved.name}，只允许 {suffix_text}")
    return resolved


def patch_model_status() -> dict:
    predict_model = patch_predict_model_file()
    return {
        "path": str(PATCH_MODEL_FILE),
        "exists": PATCH_MODEL_FILE.exists(),
        "mtime": PATCH_MODEL_FILE.stat().st_mtime if PATCH_MODEL_FILE.exists() else None,
        "pt_path": str(PATCH_PT_MODEL_FILE),
        "pt_exists": PATCH_PT_MODEL_FILE.exists(),
        "pt_mtime": PATCH_PT_MODEL_FILE.stat().st_mtime if PATCH_PT_MODEL_FILE.exists() else None,
        "predict_path": str(predict_model) if predict_model else None,
        "predict_exists": predict_model is not None,
        "latest_archive": latest_model_archive(),
        "archives": list_model_archives(),
    }


def finalize_train_plan(exit_code: int | None):
    global _train_plan

    if _train_plan is None or _train_plan.get("finalized"):
        return
    plan = _train_plan
    finished_at = datetime.now().isoformat(timespec="seconds")
    status = "finished" if exit_code == 0 else "failed"
    manifest = load_training_manifest()
    run_record = {
        "id": plan["id"],
        "status": status,
        "exit_code": exit_code,
        "mode": plan["mode"],
        "started_at": plan["started_at"],
        "finished_at": finished_at,
        "base_model": plan["base_model"],
        "data_yaml": plan["data_yaml"],
        "train_images": plan["train_images"],
        "val_images": plan["val_images"],
        "archive": plan.get("archive"),
    }
    manifest.setdefault("runs", []).append(run_record)
    if exit_code == 0:
        for image_rel, digest in plan["sample_digests"].items():
            manifest.setdefault("samples", {})[image_rel] = {
                "digest": digest,
                "last_run": plan["id"],
                "trained_at": finished_at,
            }
    save_training_manifest(manifest)
    plan["finalized"] = True
    plan["status"] = status
    plan["exit_code"] = exit_code
    plan["finished_at"] = finished_at


def split_stats(split: str) -> dict:
    image_dir = store.root / "images" / split
    images = sorted(image_dir.glob("*.png"))
    items = labeled_image_items(split)
    labeled_images = len(items)
    boxes = sum(item["boxes"] for item in items)
    untrained_items = [item for item in items if not item["trained"]]
    return {
        "images": len(images),
        "labeled_images": labeled_images,
        "boxes": boxes,
        "trained_labeled_images": labeled_images - len(untrained_items),
        "untrained_images": len(images) - (labeled_images - len(untrained_items)),
        "untrained_labeled_images": len(untrained_items),
        "untrained_boxes": sum(item["boxes"] for item in untrained_items),
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


def environment_cache_key(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return path


def _safe_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def environment_cache_fingerprint(path: str, modules: list[str]) -> str:
    try:
        python_path = Path(path).resolve()
    except OSError:
        python_path = Path(path)
    parts = [
        str(python_path),
        str(_safe_mtime_ns(python_path)),
        str(_safe_mtime_ns(VENV_TRAIN_PYTHON)),
        str(_safe_mtime_ns(OAS_TRAIN_PYTHON)),
    ]
    site_roots = []
    for root in (VENV_SITE_PACKAGES, OAS_ROOT / "toolkit" / "Lib" / "site-packages"):
        try:
            resolved_root = root.resolve()
        except OSError:
            resolved_root = root
        if resolved_root not in site_roots:
            site_roots.append(resolved_root)
    for root in site_roots:
        parts.append(f"site:{root}:{_safe_mtime_ns(root)}")
        for module in modules:
            candidates = [root / module]
            try:
                candidates.extend(root.glob(f"{module}*.dist-info"))
                candidates.extend(root.glob(f"{module.replace('_', '-')}*.dist-info"))
            except OSError:
                pass
            seen: set[Path] = set()
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                except OSError:
                    resolved = candidate
                if resolved in seen:
                    continue
                seen.add(resolved)
                parts.append(f"{resolved}:{_safe_mtime_ns(resolved)}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _load_environment_cache_file() -> dict:
    if not ENV_CHECK_CACHE_FILE.exists():
        return {"version": 1, "records": {}}
    try:
        with ENV_CHECK_CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"version": 1, "records": {}}
    if not isinstance(data, dict):
        return {"version": 1, "records": {}}
    records = data.get("records")
    if not isinstance(records, dict):
        data["records"] = {}
    return data


def load_environment_file_cache(cache_key: str, fingerprint: str) -> dict | None:
    data = _load_environment_cache_file()
    record = data.get("records", {}).get(cache_key)
    if not isinstance(record, dict):
        return None
    checked_at = float(record.get("checked_at") or 0)
    if record.get("fingerprint") != fingerprint:
        return None
    if time.time() - checked_at > ENV_CHECK_FILE_CACHE_SECONDS:
        return None
    environment = record.get("environment")
    if not isinstance(environment, dict):
        return None
    return mark_environment_cached(environment, "file")


def save_environment_file_cache(cache_key: str, fingerprint: str, environment: dict):
    TRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_environment_cache_file()
    records = data.setdefault("records", {})
    checked_at = float(environment.get("checked_at") or time.time())
    clean_environment = dict(environment)
    clean_environment["cached"] = False
    clean_environment["cache_source"] = "fresh"
    records[cache_key] = {
        "fingerprint": fingerprint,
        "checked_at": checked_at,
        "environment": clean_environment,
    }
    tmp_path = ENV_CHECK_CACHE_FILE.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(ENV_CHECK_CACHE_FILE)


def clear_environment_cache(path: str | None = None):
    if path:
        _environment_cache.pop(environment_cache_key(path), None)
    else:
        _environment_cache.clear()
    try:
        ENV_CHECK_CACHE_FILE.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def mark_environment_cached(environment: dict, source: str) -> dict:
    data = dict(environment)
    data["cached"] = True
    data["cache_source"] = source
    return data


def pending_training_environment(path: str, reason: str) -> dict:
    return {
        "python": path,
        "exists": executable_exists(path),
        "ok": False,
        "modules": {module: False for module in ["ultralytics", "torch", "onnx"]},
        "torch": empty_torch_info(),
        "gpu": {
            "hardware": nvidia_gpus(),
            "usable": False,
            "reason": reason,
        },
        "error": reason,
        "cached": False,
        "cache_source": "pending",
        "checked_at": time.time(),
        "fingerprint": "",
    }


def check_training_environment(path: str, force: bool = False, cache_result: bool = True) -> dict:
    modules = ["ultralytics", "torch", "onnx"]
    cache_key = environment_cache_key(path)
    fingerprint = environment_cache_fingerprint(path, modules)
    if not force:
        cached = _environment_cache.get(cache_key)
        if cached and time.time() - cached[0] < ENV_CHECK_CACHE_SECONDS:
            data = cached[1]
            if data.get("fingerprint") == fingerprint:
                return mark_environment_cached(data, "memory")
        file_cached = load_environment_file_cache(cache_key, fingerprint)
        if file_cached:
            _environment_cache[cache_key] = (time.time(), file_cached)
            return file_cached

    def finish(data: dict) -> dict:
        finished = dict(data)
        finished["fingerprint"] = fingerprint
        finished["cached"] = False
        finished["cache_source"] = "fresh"
        finished["checked_at"] = time.time()
        if cache_result:
            _environment_cache[cache_key] = (time.time(), finished)
            save_environment_file_cache(cache_key, fingerprint, finished)
        return finished

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
    if exit_code is not None:
        finalize_train_plan(exit_code)
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


def training_status(path: str | None = None, device: str = "cpu", refresh_env: bool = False) -> dict:
    python_path = clean_python_path(path)
    job = current_train_job()
    install = current_install_job()
    if install["running"]:
        environment = pending_training_environment(python_path, "依赖安装中，安装完成后会重新检查")
    else:
        environment = check_training_environment(python_path, force=refresh_env)
    return {
        "dataset": dataset_training_stats(),
        "environment": environment,
        "default_python": str(DEFAULT_TRAIN_PYTHON),
        "oas_python": str(OAS_TRAIN_PYTHON),
        "venv_python": str(VENV_TRAIN_PYTHON),
        "commands": training_commands(python_path, device),
        "job": job,
        "install": install,
        "log": train_log_tail(),
        "install_log": install_log_tail(),
        "model": patch_model_status(),
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


def image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
            width, height = struct.unpack(">II", header[16:24])
            return width, height
    except OSError:
        return None

    image = cv2.imread(str(path))
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def read_annotations_from_files(path: Path, label_path: Path, classes: list[dict]) -> list[dict]:
    if not label_path.exists():
        return []
    dimensions = image_dimensions(path)
    if dimensions is None:
        return []
    width, height = dimensions
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


def read_annotations(image_rel: str, classes: list[dict]) -> list[dict]:
    path, split = image_path(image_rel)
    label_path = store.root / "labels" / split / f"{path.stem}.txt"
    return read_annotations_from_files(path, label_path, classes)


def frame_item(path: Path, split: str, classes: list[dict], manifest: dict | None = None) -> dict:
    rel = f"{split}/{path.name}"
    label_file = store.root / "labels" / split / f"{path.stem}.txt"
    train_state = sample_training_state(rel, path, label_file, manifest)
    return {
        "image": rel,
        "name": path.name,
        "split": split,
        "mtime": path.stat().st_mtime,
        "boxes": read_annotations_from_files(path, label_file, classes),
        "trained": train_state["trained"],
        "trained_at": train_state["trained_at"],
        "last_train_run": train_state["last_run"],
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
        "legacy_classes": legacy_classes_for_picker(),
        "class_usage": class_usage_stats(classes),
        "frames": {
            "train": len(list((store.root / "images" / "train").glob("*.png"))),
            "val": len(list((store.root / "images" / "val").glob("*.png"))),
        },
        "export": {
            "patch_labels": str(PATCH_LABELS_FILE),
            "data_yaml": str(store.root / "data.yaml"),
        },
        "patch_model": patch_model_status(),
        "legacy_model": {
            "available": LegacyTracker is not None,
            "error": LEGACY_IMPORT_ERROR,
        },
        "oas": {
            "root": str(OAS_ROOT),
            "exists": OAS_ROOT.exists(),
            "configured_root": configured_oas_root(),
            "restart_required": oas_restart_required(),
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


@app.post("/api/oas-root")
def oas_root_settings(payload: OasRootIn):
    return save_oas_root_config(payload.oas_root)


@app.post("/api/pick-directory")
def pick_directory_api(payload: DirectoryPickIn):
    return pick_directory(payload.title, payload.initial)


@app.get("/api/frames")
def frames(split: Literal["train", "val"] = "train"):
    classes = store.load_classes()
    manifest = load_training_manifest()
    files = sorted((store.root / "images" / split).glob("*.png"), key=lambda p: p.stat().st_mtime)
    return {"frames": [frame_item(path, split, classes, manifest) for path in files]}


@app.post("/api/frames/move")
def move_frames(payload: MoveFramesIn):
    moved = []
    manifest = load_training_manifest()
    manifest_changed = False
    for image_rel in payload.images:
        item = move_frame_item(image_rel, payload.target)
        if item is not None:
            moved.append(item)
            record = manifest.get("samples", {}).pop(item["from"], None)
            if record is not None:
                manifest.setdefault("samples", {})[item["to"]] = record
                manifest_changed = True
    if manifest_changed:
        save_training_manifest(manifest)
    return {
        "target": payload.target,
        "moved": moved,
    }


@app.post("/api/frames/delete")
def delete_frames(payload: DeleteFramesIn):
    deleted = []
    manifest = load_training_manifest()
    manifest_changed = False
    for image_rel in payload.images:
        deleted.append(delete_frame_item(image_rel))
        if manifest.get("samples", {}).pop(image_rel, None) is not None:
            manifest_changed = True
    if manifest_changed:
        save_training_manifest(manifest)
    return {
        "deleted": deleted,
    }


@app.get("/api/image")
def image(image: str = Query(...)):
    path, _split = image_path(image)
    return FileResponse(path)


@app.post("/api/legacy-detect")
def legacy_detect(payload: LegacyDetectIn):
    path, _split = image_path(payload.image)
    model_path, labels_path = _resolve_patch_paths(payload.patch_model_path, payload.patch_labels_path)
    detections, warnings = detect_reference_image(
        path,
        payload.source,
        payload.conf_threshold,
        payload.iou_threshold,
        model_path,
        labels_path,
    )
    return {
        "image": payload.image,
        "source": payload.source,
        "patch_model_path": str(model_path) if model_path else None,
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

    model_path, labels_path = _resolve_patch_paths(payload.patch_model_path, payload.patch_labels_path)
    predictions = {}
    warnings = []
    total = 0
    for image_rel, path in image_items:
        detections, image_warnings = detect_reference_image(
            path,
            payload.source,
            payload.conf_threshold,
            payload.iou_threshold,
            model_path,
            labels_path,
        )
        predictions[image_rel] = detections
        warnings.extend(f"{image_rel}: {warning}" for warning in image_warnings)
        total += len(detections)
    return {
        "split": payload.split,
        "source": payload.source,
        "patch_model_path": str(model_path) if model_path else None,
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


@app.post("/api/classes/delete")
def delete_class(payload: DeleteClassIn):
    result = delete_class_with_renumber(payload.label, payload.force)
    return result


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
def train_status(
    python: str | None = Query(None),
    device: str = Query("cpu"),
    refresh_env: bool = Query(False),
):
    return training_status(python, device, refresh_env)


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

    clear_environment_cache(python_path)
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
    global _train_process, _train_command, _train_started_at, _train_plan

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

    run_id = timestamp_id()
    base_model = payload.model.strip() if payload.model else "yolov8n.pt"
    if payload.mode == "incremental":
        if not PATCH_PT_MODEL_FILE.exists():
            raise HTTPException(status_code=400, detail=f"增量训练需要先有当前 best.pt: {PATCH_PT_MODEL_FILE}")
        if base_model == "yolov8n.pt":
            base_model = str(PATCH_PT_MODEL_FILE)
    prepared = prepare_training_dataset(payload.mode, run_id)
    train_items = prepared["train_items"]
    val_items = prepared["val_items"]
    archive = archive_current_model("before_train") if payload.archive_existing else None

    TRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _train_command = [
        environment["python"],
        str(TRAIN_SCRIPT),
        "--data", str(prepared["data_yaml"]),
        "--model", base_model,
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
    _train_plan = {
        "id": run_id,
        "mode": payload.mode,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": base_model,
        "data_yaml": str(prepared["data_yaml"]),
        "train_images": [item["image"] for item in train_items],
        "val_images": [item["image"] for item in val_items],
        "sample_digests": {item["image"]: item["digest"] for item in train_items},
        "archive": archive,
        "finalized": False,
    }
    TRAIN_LOG_FILE.write_text(
        "Hyakkiyakou patch training\n"
        f"dataset: {store.root}\n"
        f"run: {run_id}\n"
        f"mode: {payload.mode}\n"
        f"train images: {len(train_items)}\n"
        f"val images: {len(val_items)}\n"
        f"archive: {archive['path'] if archive else 'none'}\n"
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
