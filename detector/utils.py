# This Python file uses the following encoding: utf-8
import cv2
import numpy as np

from tasks.Hyakkiyakou.detector.labels import id2label


def _class_color(class_id: int):
    return (
        int((class_id * 37 + 61) % 255),
        int((class_id * 17 + 149) % 255),
        int((class_id * 97 + 23) % 255),
    )


def draw_tracks(image, tracks):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for _id, _class, _conf, _cx, _cy, _w, _h, _v in tracks:
        x1 = int(_cx - _w / 2)
        y1 = int(_cy - _h / 2)
        x2 = int(_cx + _w / 2)
        y2 = int(_cy + _h / 2)
        text_label = f"{id2label(_class)}({_conf:.2f})"
        text_id = f"ID: {_id}"

        color = _class_color(_class)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        (text_width, text_height), baseline = cv2.getTextSize(
            text_label,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=1,
        )
        bottom_left_corner = (x1, y1)
        top_right_corner = (x1 + text_width, y1 + text_height + baseline)
        cv2.rectangle(image, bottom_left_corner, top_right_corner, color, cv2.FILLED)
        cv2.putText(
            image,
            text_label,
            (x1, y1 + text_height),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )
        cv2.putText(
            image,
            text_id,
            (x1, y1 - text_height + baseline),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
        )
    text_oas = "Built with OAS"
    text_open = "Open Source"
    cv2.putText(
        image,
        text_open,
        (40, 70),
        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        fontScale=1.5,
        color=(0, 255, 0),
        thickness=2,
    )
    cv2.putText(
        image,
        text_oas,
        (800, 650),
        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        fontScale=2,
        color=(0, 255, 0),
        thickness=2,
    )
    return image


def image_hash_color_preview(class_ids: list[int]) -> np.ndarray:
    canvas = np.zeros((40, max(1, len(class_ids)) * 40, 3), dtype=np.uint8)
    for index, class_id in enumerate(class_ids):
        canvas[:, index * 40:(index + 1) * 40] = _class_color(class_id)
    return canvas
