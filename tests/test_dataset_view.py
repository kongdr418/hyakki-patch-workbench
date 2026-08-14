import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

import collector_app

from collector_app import _frame_summary, _split_stats_from_frames


def frame(name: str, labels: list[str], trained: bool) -> dict:
    return {
        "image": f"train/{name}.png",
        "name": f"{name}.png",
        "split": "train",
        "mtime": 123.0,
        "boxes": [{"label": label, "x": 1, "y": 2, "w": 3, "h": 4} for label in labels],
        "trained": trained,
        "trained_at": "",
        "last_train_run": "",
    }


class DatasetViewTests(unittest.TestCase):
    def test_frame_summary_omits_full_box_coordinates(self):
        summary = _frame_summary(frame("sample", ["ssr_001", "r_001"], True))

        self.assertEqual(summary["box_count"], 2)
        self.assertEqual(summary["box_labels"], ["ssr_001", "r_001"])
        self.assertNotIn("boxes", summary)
        self.assertNotIn("name", summary)

    def test_split_stats_counts_untrained_and_unlabeled_images(self):
        frames = [
            frame("trained", ["ssr_001", "r_001"], True),
            frame("changed", ["sr_001"], False),
            frame("empty", [], False),
        ]

        stats = _split_stats_from_frames(frames)

        self.assertEqual(stats["images"], 3)
        self.assertEqual(stats["labeled_images"], 2)
        self.assertEqual(stats["boxes"], 3)
        self.assertEqual(stats["trained_labeled_images"], 1)
        self.assertEqual(stats["untrained_images"], 2)
        self.assertEqual(stats["untrained_labeled_images"], 1)
        self.assertEqual(stats["untrained_boxes"], 1)

    def test_cache_file_round_trip_and_invalidation(self):
        original_path = collector_app.DATASET_VIEW_CACHE_FILE
        original_cache = collector_app._dataset_view_cache
        try:
            with TemporaryDirectory() as temp_dir:
                cache_path = Path(temp_dir) / "dataset-view.json"
                collector_app.DATASET_VIEW_CACHE_FILE = cache_path
                fingerprint = {"root": "test", "paths": []}
                view = {"fingerprint": fingerprint, "frames": {"train": [], "val": []}}

                collector_app._save_dataset_view_file(fingerprint, view)
                self.assertEqual(collector_app._load_dataset_view_file(fingerprint), view)

                cache_path.write_text("[]", encoding="utf-8")
                self.assertIsNone(collector_app._load_dataset_view_file(fingerprint))

                collector_app._dataset_view_cache = view
                collector_app.invalidate_dataset_view_cache()
                self.assertIsNone(collector_app._dataset_view_cache)
                self.assertFalse(cache_path.exists())
        finally:
            collector_app.DATASET_VIEW_CACHE_FILE = original_path
            collector_app._dataset_view_cache = original_cache


if __name__ == "__main__":
    unittest.main()
