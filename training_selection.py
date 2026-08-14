from __future__ import annotations

import hashlib
import heapq

from collections import Counter


def _item_class_counts(item: dict) -> dict[int, int]:
    return {
        int(class_id): int(count)
        for class_id, count in item.get("class_counts", {}).items()
        if int(count) > 0
    }


def _stable_rank(item: dict) -> int:
    image = str(item.get("image", ""))
    return int(hashlib.sha256(image.encode("utf-8")).hexdigest()[:16], 16)


def select_training_items_by_class(
    items: list[dict],
    max_boxes_per_class: int,
) -> tuple[list[dict], dict]:
    """Select a compact, deterministic image subset with per-class box targets."""
    candidates = list(items)
    total_counts: Counter[int] = Counter()
    for item in candidates:
        total_counts.update(_item_class_counts(item))

    target_counts = {
        class_id: min(count, max_boxes_per_class) if max_boxes_per_class > 0 else count
        for class_id, count in total_counts.items()
    }
    if max_boxes_per_class <= 0:
        selected = candidates
        selected_counts = total_counts.copy()
    else:
        prepared = []
        for item in candidates:
            counts = _item_class_counts(item)
            prepared.append((item, counts, _stable_rank(item)))

        def score_item(counts: dict[int, int], stable_rank: int, selected_counts: Counter[int]):
            accepted_counts = {
                class_id: min(count, max(0, target_counts[class_id] - selected_counts[class_id]))
                for class_id, count in counts.items()
            }
            accepted_counts = {
                class_id: count
                for class_id, count in accepted_counts.items()
                if count > 0
            }
            useful_boxes = sum(accepted_counts.values())
            if useful_boxes <= 0:
                return None, accepted_counts
            normalized_benefit = sum(
                count / target_counts[class_id]
                for class_id, count in accepted_counts.items()
            )
            score = (
                normalized_benefit,
                len(accepted_counts),
                useful_boxes,
                -(sum(counts.values()) - useful_boxes),
                stable_rank,
            )
            return tuple(-value for value in score), accepted_counts

        selected_counts: Counter[int] = Counter()
        selected = []
        queue = []
        for index, (_item, counts, stable_rank) in enumerate(prepared):
            priority, _accepted_counts = score_item(counts, stable_rank, selected_counts)
            if priority is not None:
                heapq.heappush(queue, (priority, index))

        while queue:
            previous_priority, index = heapq.heappop(queue)
            item, counts, stable_rank = prepared[index]
            priority, accepted_counts = score_item(counts, stable_rank, selected_counts)
            if priority is None:
                continue
            if priority != previous_priority:
                heapq.heappush(queue, (priority, index))
                continue
            selected_item = dict(item)
            selected_item["training_class_counts"] = accepted_counts
            selected.append(selected_item)
            selected_counts.update(accepted_counts)

    stats = {
        "max_boxes_per_class": max_boxes_per_class,
        "candidate_images": len(candidates),
        "selected_images": len(selected),
        "candidate_boxes": sum(total_counts.values()),
        "selected_boxes": sum(selected_counts.values()),
        "target_boxes": sum(target_counts.values()),
        "candidate_class_counts": dict(total_counts),
        "selected_class_counts": dict(selected_counts),
    }
    return selected, stats
