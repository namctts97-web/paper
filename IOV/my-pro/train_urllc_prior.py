from __future__ import annotations

import sys

from train_prior import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--task", "urllc",
        "--data", "data/urllc_expert_dataset.csv",
        *sys.argv[1:],
    ]
    main()
