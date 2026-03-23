import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from viewer import gt_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None)
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root) if args.data_root else None
    status = gt_labels.preload_labels(data_root)
    print("STATUS", status)

    if not status.get("ok"):
        raise SystemExit(2)

    embedded_source = status.get("source")
    if not embedded_source:
        raise SystemExit(3)

    keys = list(gt_labels._GT_CACHE.keys())
    if not keys:
        raise SystemExit(4)

    first_key = keys[0]
    gt = gt_labels.get_gt_for_case(first_key, data_root)
    if not gt:
        raise SystemExit(5)

    print("FIRST_KEY", first_key)
    print("FIRST_GT", gt[0])


if __name__ == "__main__":
    main()

