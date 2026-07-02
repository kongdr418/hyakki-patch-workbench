# This Python file uses the following encoding: utf-8
import time

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from oashya.tracker import Tracker as LegacyTracker

from tasks.Hyakkiyakou.detector.labels import DEFAULT_EXTRA_LABELS, extra_classify


HYAKKI_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PATCH_MODEL = HYAKKI_DIR / "models" / "hya_patch_fp32.onnx"


@dataclass
class _LetterboxMeta:
    scale: float
    pad_x: float
    pad_y: float
    input_w: int
    input_h: int
    image_w: int
    image_h: int


class _OnnxPatchDetector:
    def __init__(self, args: dict | None = None):
        args = args or {}
        self.model_path = Path(args.get("patch_model_path") or DEFAULT_PATCH_MODEL)
        self.labels_path = Path(args.get("patch_labels_path") or DEFAULT_EXTRA_LABELS)
        self.conf_threshold = float(args.get("patch_conf_threshold", args.get("conf_threshold", 0.6)))
        self.iou_threshold = float(args.get("patch_iou_threshold", args.get("iou_threshold", 0.7)))
        self.class_ids = [item["id"] for item in extra_classify()]
        self._previous: list[dict] = []
        self._next_id = 10000

        if not self.model_path.exists() or not self.labels_path.exists() or not self.class_ids:
            self.session = None
            return

        import onnxruntime as ort

        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_h = int(input_shape[2] if isinstance(input_shape[2], int) else 640)
        self.input_w = int(input_shape[3] if isinstance(input_shape[3], int) else 640)

    @property
    def enabled(self) -> bool:
        return self.session is not None

    def clear_tracks(self):
        self._previous.clear()

    def __call__(self, image, response: list | None = None) -> list[tuple]:
        if not self.enabled:
            return []
        tensor, meta = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        detections = self._postprocess(outputs[0], meta)
        return self._track(detections)

    def _preprocess(self, image) -> tuple[np.ndarray, _LetterboxMeta]:
        image_h, image_w = image.shape[:2]
        scale = min(self.input_w / image_w, self.input_h / image_h)
        resized_w = int(round(image_w * scale))
        resized_h = int(round(image_h * scale))
        pad_x = (self.input_w - resized_w) / 2
        pad_y = (self.input_h - resized_h) / 2

        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        top = int(round(pad_y - 0.1))
        left = int(round(pad_x - 0.1))
        canvas[top:top + resized_h, left:left + resized_w] = resized
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]

        return tensor, _LetterboxMeta(
            scale=scale,
            pad_x=left,
            pad_y=top,
            input_w=self.input_w,
            input_h=self.input_h,
            image_w=image_w,
            image_h=image_h,
        )

    def _postprocess(self, output, meta: _LetterboxMeta) -> list[tuple]:
        pred = np.squeeze(output)
        if pred.ndim != 2:
            return []
        if self._looks_like_end2end(pred):
            return self._postprocess_end2end(pred, meta)
        # YOLOv8 often exports as (classes + 4, anchors); YOLOv5 is usually (anchors, classes + 5).
        if pred.shape[0] < pred.shape[1] and pred.shape[0] <= len(self.class_ids) + 5:
            pred = pred.T
        if self._looks_like_end2end(pred):
            return self._postprocess_end2end(pred, meta)

        detections = []
        min_fields = 4 + len(self.class_ids)
        if pred.shape[1] < min_fields:
            return []
        has_objectness = pred.shape[1] >= len(self.class_ids) + 5
        class_offset = 5 if has_objectness else 4
        for row in pred:
            objectness = float(row[4]) if has_objectness else 1.0
            scores = row[class_offset:class_offset + len(self.class_ids)]
            if scores.size == 0:
                continue
            local_class = int(np.argmax(scores))
            conf = objectness * float(scores[local_class])
            if conf < self.conf_threshold:
                continue
            cx, cy, w, h = [float(v) for v in row[:4]]
            cx = (cx - meta.pad_x) / meta.scale
            cy = (cy - meta.pad_y) / meta.scale
            w = w / meta.scale
            h = h / meta.scale
            if w <= 0 or h <= 0:
                continue
            cx = min(max(cx, 0), meta.image_w - 1)
            cy = min(max(cy, 0), meta.image_h - 1)
            w = min(w, meta.image_w)
            h = min(h, meta.image_h)
            detections.append((self.class_ids[local_class], conf, cx, cy, w, h))
        return self._nms(detections)

    def _looks_like_end2end(self, pred: np.ndarray) -> bool:
        if pred.shape[1] != 6 or pred.shape[0] == 0:
            return False
        conf = pred[:, 4]
        class_ids = pred[:, 5]
        if np.nanmax(conf) > 1.0001 or np.nanmin(conf) < -0.0001:
            return False
        if np.nanmin(class_ids) < 0 or np.nanmax(class_ids) >= max(1, len(self.class_ids)):
            return False
        integerish = np.all(np.abs(class_ids - np.round(class_ids)) < 1e-3)
        xyxy_like = np.mean((pred[:, 2] > pred[:, 0]) & (pred[:, 3] > pred[:, 1])) > 0.8
        return bool(integerish and xyxy_like)

    def _postprocess_end2end(self, pred: np.ndarray, meta: _LetterboxMeta) -> list[tuple]:
        detections = []
        for x1, y1, x2, y2, conf, local_class in pred:
            conf = float(conf)
            if conf < self.conf_threshold:
                continue
            local_class = int(round(float(local_class)))
            if local_class < 0 or local_class >= len(self.class_ids):
                continue
            x1 = (float(x1) - meta.pad_x) / meta.scale
            y1 = (float(y1) - meta.pad_y) / meta.scale
            x2 = (float(x2) - meta.pad_x) / meta.scale
            y2 = (float(y2) - meta.pad_y) / meta.scale
            x1 = min(max(x1, 0), meta.image_w - 1)
            y1 = min(max(y1, 0), meta.image_h - 1)
            x2 = min(max(x2, 0), meta.image_w - 1)
            y2 = min(max(y2, 0), meta.image_h - 1)
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            detections.append((self.class_ids[local_class], conf, (x1 + x2) / 2, (y1 + y2) / 2, w, h))
        return self._nms(detections)

    def _nms(self, detections: list[tuple]) -> list[tuple]:
        if not detections:
            return []
        boxes = []
        for _class, conf, cx, cy, w, h in detections:
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray([det[1] for det in detections], dtype=np.float32)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = self._iou(boxes[i], boxes[rest])
            order = rest[ious <= self.iou_threshold]
        return [detections[i] for i in keep]

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
        area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
        return intersection / np.maximum(area_a + area_b - intersection, 1e-6)

    def _track(self, detections: list[tuple]) -> list[tuple]:
        now = time.time()
        tracks = []
        next_previous = []
        for class_id, conf, cx, cy, w, h in detections:
            match = self._match_previous(class_id, cx, cy)
            if match is None:
                track_id = self._next_id
                self._next_id += 1
                velocity = 0.0
            else:
                track_id = match["id"]
                dt_ms = max((now - match["time"]) * 1000, 1.0)
                velocity = (cx - match["cx"]) / dt_ms
            tracks.append((track_id, class_id, conf, int(cx), int(cy), int(w), int(h), float(velocity)))
            next_previous.append({"id": track_id, "class": class_id, "cx": cx, "cy": cy, "time": now})
        self._previous = next_previous
        return tracks

    def _match_previous(self, class_id: int, cx: float, cy: float) -> dict | None:
        best = None
        best_distance = 160 ** 2
        for item in self._previous:
            if item["class"] != class_id:
                continue
            distance = (cx - item["cx"]) ** 2 + (cy - item["cy"]) ** 2
            if distance < best_distance:
                best_distance = distance
                best = item
        return best


class Tracker:
    def __init__(self, args: dict = {}, logger=None):
        self.legacy = LegacyTracker(args=args, logger=logger)
        self.patch = _OnnxPatchDetector(args=args)

    def __getattr__(self, item):
        return getattr(self.legacy, item)

    def __call__(self, image, response: list):
        tracks = list(self.legacy(image=image, response=response))
        tracks.extend(self.patch(image=image, response=response))
        return tracks

    def clear_tracks(self):
        self.legacy.clear_tracks()
        self.patch.clear_tracks()
