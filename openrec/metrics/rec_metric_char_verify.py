"""
RecCharVerifyMetric (v4):
  Metric for CharWiseVerify pipeline.
  - Text branch: acc, norm_edit_dis (main indicators)
  - IDS branch: ids_acc, ids_norm_edit_dis (auxiliary)
  
  Same structure as RecTextIDSMetric but adapted for the v4 pipeline.
"""

from .rec_metric import RecMetric


class RecCharVerifyMetric(object):

    def __init__(
        self,
        main_indicator="acc",
        is_filter=False,
        ignore_space=False,
        stream=False,
        with_ratio=False,
        max_len=25,
        max_len_ids=None,
        max_ratio=4,
        **kwargs,
    ):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space

        self.max_len_text = max_len
        self.max_len_ids = max_len_ids if max_len_ids is not None else max_len

        self.text_metric = RecMetric(
            main_indicator=main_indicator,
            is_filter=is_filter,
            ignore_space=ignore_space,
            stream=stream,
            with_ratio=with_ratio,
            max_len=self.max_len_text,
            max_ratio=max_ratio,
        )
        self.ids_metric = RecMetric(
            main_indicator=main_indicator,
            is_filter=is_filter,
            ignore_space=False,  # IDS spaces are structural separators
            stream=stream,
            with_ratio=with_ratio,
            max_len=self.max_len_ids,
            max_ratio=max_ratio,
        )

    def __call__(self, pred_label, batch=None, training=False, *args, **kwargs):
        # pred_label[0] = (text_preds, text_labels)
        # pred_label[1] = (ids_preds, ids_labels)
        text_metric = self.text_metric(pred_label[0], batch, training=training)
        ids_metric = self.ids_metric(pred_label[1], batch, training=training)

        text_metric["ids_acc"] = ids_metric["acc"]
        text_metric["ids_norm_edit_dis"] = ids_metric["norm_edit_dis"]
        return text_metric

    def get_metric(self):
        text_metric = self.text_metric.get_metric()
        ids_metric = self.ids_metric.get_metric()

        text_metric["ids_acc"] = ids_metric["acc"]
        text_metric["ids_norm_edit_dis"] = ids_metric["norm_edit_dis"]
        return text_metric


# python -m openrec.metrics.rec_metric_char_verify
if __name__ == "__main__":
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Project root: {project_root}")
    print("=" * 60)
    print("Testing RecCharVerifyMetric")
    print("=" * 60)

    try:
        metric = RecCharVerifyMetric(
            main_indicator="acc",
            ignore_space=True,
            max_len=15,
            max_len_ids=15,
        )
        print("[OK] Metric initialized.")

        # Simulate predictions
        # Text branch: 1 correct, 1 wrong
        text_preds = [("hello", 0.99), ("world", 0.8)]
        text_labels = [("hello", None), ("word", None)]

        # IDS branch: 2 correct
        ids_preds = [("⿰亻尔 ⿱女子", 0.99), ("⿰亻尔 ⿱女子", 0.95)]
        ids_labels = [("⿰亻尔 ⿱女子", None), ("⿰亻尔 ⿱女子", None)]

        pred_label_input = [
            (text_preds, text_labels),
            (ids_preds, ids_labels),
        ]

        print("\nCalling metric...")
        batch_metric = metric(pred_label_input)
        print(f"  Batch metric: {batch_metric}")

        final_metric = metric.get_metric()
        print(f"  Final metric: {final_metric}")

        eps = 1e-6
        assert abs(final_metric["acc"] - 0.5) < eps, f"Expected text acc=0.5, got {final_metric['acc']}"
        assert abs(final_metric["ids_acc"] - 1.0) < eps, f"Expected ids acc=1.0, got {final_metric['ids_acc']}"
        print(f"  [OK] text acc = {final_metric['acc']:.2f} (expected 0.50)")
        print(f"  [OK] ids_acc = {final_metric['ids_acc']:.2f} (expected 1.00)")

        # Test accumulation
        metric2 = RecCharVerifyMetric(main_indicator="acc", max_len=15, max_len_ids=15)

        text_preds_2 = [("abc", 0.9), ("def", 0.9), ("ghi", 0.9)]
        text_labels_2 = [("abc", None), ("def", None), ("xyz", None)]
        ids_preds_2 = [("a", 0.9), ("b", 0.9), ("c", 0.9)]
        ids_labels_2 = [("a", None), ("b", None), ("d", None)]

        metric2([(text_preds_2[:2], text_labels_2[:2]), (ids_preds_2[:2], ids_labels_2[:2])])
        metric2([(text_preds_2[2:], text_labels_2[2:]), (ids_preds_2[2:], ids_labels_2[2:])])

        final2 = metric2.get_metric()
        print(f"\n  Accumulated metric: {final2}")
        # text: 2/3 correct, ids: 2/3 correct
        expected_acc = 2.0 / 3.0
        assert abs(final2["acc"] - expected_acc) < eps, f"Expected {expected_acc}, got {final2['acc']}"
        print(f"  [OK] Accumulation works: acc={final2['acc']:.4f}, ids_acc={final2['ids_acc']:.4f}")

        print("\n" + "=" * 60)
        print("[PASS] All RecCharVerifyMetric tests passed!")
        print("=" * 60)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FAIL] {e}")
