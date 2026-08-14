import unittest

from training_selection import select_training_items_by_class


def item(name: str, **class_counts: int) -> dict:
    return {
        "image": f"train/{name}.png",
        "class_counts": {int(class_id): count for class_id, count in class_counts.items()},
    }


class TrainingSelectionTests(unittest.TestCase):
    def test_keeps_all_rare_boxes_and_caps_common_class(self):
        items = [item(f"common-{index}", **{"0": 1}) for index in range(50)]
        items += [item(f"rare-{index}", **{"1": 1}) for index in range(4)]

        selected, stats = select_training_items_by_class(items, 30)

        self.assertEqual(stats["selected_class_counts"][0], 30)
        self.assertEqual(stats["selected_class_counts"][1], 4)
        self.assertEqual(len(selected), 34)

    def test_prefers_images_that_cover_multiple_needed_classes(self):
        items = [item("combined", **{"0": 1, "1": 1})]
        items += [item(f"class-0-{index}", **{"0": 1}) for index in range(10)]
        items += [item(f"class-1-{index}", **{"1": 1}) for index in range(10)]

        selected, stats = select_training_items_by_class(items, 3)

        self.assertIn("train/combined.png", [entry["image"] for entry in selected])
        self.assertEqual(stats["selected_class_counts"], {0: 3, 1: 3})
        self.assertEqual(stats["selected_images"], 5)

    def test_drops_overrepresented_boxes_from_mixed_images(self):
        items = [item(f"rare-{index}", **{"0": 5, str(index + 1): 1}) for index in range(10)]

        selected, stats = select_training_items_by_class(items, 3)

        self.assertEqual(len(selected), 10)
        self.assertEqual(stats["selected_class_counts"][0], 3)
        self.assertTrue(all(stats["selected_class_counts"][class_id] == 1 for class_id in range(1, 11)))
        self.assertEqual(sum(entry["training_class_counts"].get(0, 0) for entry in selected), 3)

    def test_zero_disables_the_limit(self):
        items = [item(f"frame-{index}", **{"0": 2}) for index in range(5)]

        selected, stats = select_training_items_by_class(items, 0)

        self.assertEqual(selected, items)
        self.assertEqual(stats["selected_boxes"], 10)

    def test_selection_is_deterministic(self):
        items = [item(f"frame-{index}", **{"0": 1}) for index in range(20)]

        first, _ = select_training_items_by_class(items, 5)
        second, _ = select_training_items_by_class(items, 5)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
