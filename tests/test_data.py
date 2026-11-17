import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_paths(csv_path):
    with open(csv_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    header = lines[0].split(",")
    num_samples, num_steps = int(header[0]), int(header[1])
    rows = []
    for line in lines[1:]:
        vals = [float(x) for x in line.split(",")[2:]]
        rows.append(vals)
    arr = np.array(rows)
    assert arr.shape == (num_samples, num_steps), f"shape mismatch in {csv_path}: {arr.shape}"
    return arr


def test_chronological_purge_and_non_overlapping_test():
    train_path = os.path.join(DATA_DIR, "GSPC_train_paths.csv")
    val_path = os.path.join(DATA_DIR, "GSPC_val_paths.csv")
    test_path = os.path.join(DATA_DIR, "GSPC_test_paths.csv")

    assert os.path.exists(train_path), "missing train csv"
    assert os.path.exists(val_path), "missing val csv (no held-out validation split)"
    assert os.path.exists(test_path), "missing test csv"

    train = load_paths(train_path)
    val = load_paths(val_path)
    test = load_paths(test_path)

    window_size = train.shape[1]
    assert val.shape[1] == window_size
    assert test.shape[1] == window_size

    # verify zero window overlap between train -> val and val -> test
    train_tail = set(np.round(train[-window_size:, :].ravel(), 4))
    val_head = set(np.round(val[:window_size, :].ravel(), 4))
    val_tail = set(np.round(val[-window_size:, :].ravel(), 4))
    test_head = set(np.round(test[:window_size, :].ravel(), 4))

    assert len(train_tail & val_head) == 0, "train/val overlap detected"
    assert len(val_tail & test_head) == 0, "val/test overlap detected"

    # verify test windows are strictly non-overlapping for any partial offset
    for i in range(len(test) - 1):
        for offset in range(1, window_size - 2):
            if np.allclose(test[i, offset:], test[i + 1, : window_size - offset]):
                raise AssertionError(f"test windows {i} and {i+1} overlap at offset {offset}")

    print(f"PASS: train={len(train)}, val={len(val)}, test={len(test)} non-overlapping horizons")


if __name__ == "__main__":
    test_chronological_purge_and_non_overlapping_test()
