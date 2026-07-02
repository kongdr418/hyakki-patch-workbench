# This Python file uses the following encoding: utf-8
import json

from pathlib import Path

from oashya.labels import CLASSIFY as LEGACY_CLASSIFY
from oashya.labels import CLASSINDEX


HYAKKI_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EXTRA_LABELS = HYAKKI_DIR / "models" / "hya_patch_labels.json"

RARITY_TO_WEIGHT_INDEX = {
    "sp": 0,
    "ssr": 1,
    "sr": 2,
    "r": 3,
    "n": 4,
    "g": 5,
}
RARITY_TO_SCORE = {
    "g": 0,
    "n": 1,
    "r": 2,
    "sr": 3,
    "ssr": 4,
    "sp": 5,
}


def _legacy_rarity(class_id: int) -> str | None:
    if CLASSINDEX.MIN_BUFF <= class_id <= CLASSINDEX.MAX_BUFF:
        return "buff"
    if CLASSINDEX.MIN_N <= class_id <= CLASSINDEX.MAX_N:
        return "n"
    if CLASSINDEX.MIN_G <= class_id <= CLASSINDEX.MAX_G:
        return "g"
    if CLASSINDEX.MIN_R <= class_id <= CLASSINDEX.MAX_R:
        return "r"
    if CLASSINDEX.MIN_SR <= class_id <= CLASSINDEX.MAX_SR:
        return "sr"
    if CLASSINDEX.MIN_SSR <= class_id <= CLASSINDEX.MAX_SSR:
        return "ssr"
    if CLASSINDEX.MIN_SP <= class_id <= CLASSINDEX.MAX_SP:
        return "sp"
    return None


def _load_extra_labels(path: Path = DEFAULT_EXTRA_LABELS) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("labels", [])
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list or an object with a labels list")

    next_id = max(item["id"] for item in LEGACY_CLASSIFY) + 1
    result = []
    seen_ids = {item["id"] for item in LEGACY_CLASSIFY}
    seen_labels = {item["class"] for item in LEGACY_CLASSIFY}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid label item: {item!r}")
        label = item.get("label") or item.get("class")
        if not label:
            raise ValueError(f"Missing label/class in: {item!r}")
        class_id = int(item.get("id", item.get("class_id", next_id)))
        while class_id in seen_ids:
            class_id += 1
        name = item.get("name", label)
        rarity = item.get("rarity") or str(label).split("_", 1)[0]
        rarity = str(rarity).lower()
        if label in seen_labels:
            raise ValueError(f"Duplicated label: {label}")
        result.append({
            "id": class_id,
            "class": label,
            "name": name,
            "rarity": rarity,
        })
        seen_ids.add(class_id)
        seen_labels.add(label)
        next_id = max(next_id, class_id + 1)
    return result


def _build_classify() -> list[dict]:
    legacy = []
    for item in LEGACY_CLASSIFY:
        class_id = item["id"]
        legacy.append({
            "id": class_id,
            "class": item["class"],
            "name": item["name"],
            "rarity": _legacy_rarity(class_id),
        })
    return legacy + _load_extra_labels()


CLASSIFY = _build_classify()
ID_TO_CLASS = {item["id"]: item for item in CLASSIFY}
LABEL_TO_CLASS = {item["class"]: item for item in CLASSIFY}


def extra_classify() -> list[dict]:
    return [item for item in CLASSIFY if item["id"] > CLASSINDEX.MAX_SP]


def id2label(class_id: int) -> str:
    item = ID_TO_CLASS.get(class_id)
    if item is None:
        return f"unknown_{class_id}"
    return item["class"]


def id2name(class_id: int) -> str:
    item = ID_TO_CLASS.get(class_id)
    if item is None:
        return f"未知({class_id})"
    return item["name"]


def label2id(label: str) -> int:
    item = LABEL_TO_CLASS.get(label)
    if item is None:
        raise Exception(f"Unknown Hyakkiyakou label: {label}")
    return item["id"]


def class_rarity(class_id: int) -> str | None:
    item = ID_TO_CLASS.get(class_id)
    if item is not None:
        return item.get("rarity")
    return _legacy_rarity(class_id)


def rarity_weight_index(class_id: int) -> int | None:
    return RARITY_TO_WEIGHT_INDEX.get(class_rarity(class_id))


def rarity_score(class_id: int) -> int:
    return RARITY_TO_SCORE.get(class_rarity(class_id), -1)


def is_rare_ssr_sp(class_id: int) -> bool:
    return class_rarity(class_id) in {"ssr", "sp"}
