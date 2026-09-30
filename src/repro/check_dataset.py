"""Steps 2-3: verify source files, Fold1-5 .npy inputs and pretrained checkpoints (read-only)."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "third_party", "ghoddoosian")   # vendored upstream files, opened via relative paths
os.chdir(ROOT)
problems = []

print("== core source files ==")
for name in ["Training.py", "Preprocessing.py", "blink_video.py", "HMSLSTM_cell.py",
             "MultiHMSLSTM.py", "HMSLSTM_Main.py"]:
    exists = os.path.isfile(name)
    print("%-20s %s" % (name, "present" if exists else "MISSING"))
    if not exists:
        problems.append(name)

print("\n== npy inputs (expect X:(N,30,4), y:(N,1)) ==")
for fold in range(1, 6):
    row = {}
    for prefix in ["Blinks", "BlinksTest", "Labels", "LabelsTest"]:
        fn = "%s_30_Fold%d.npy" % (prefix, fold)
        if not os.path.isfile(fn):
            problems.append(fn)
            print("Fold%d %-10s MISSING" % (fold, prefix))
            continue
        arr = np.load(fn)
        row[prefix] = arr
        print("Fold%d %-10s shape=%-16s dtype=%s" % (fold, prefix, arr.shape, arr.dtype))
    for xk, yk in [("Blinks", "Labels"), ("BlinksTest", "LabelsTest")]:
        if xk in row and yk in row:
            x, y = row[xk], row[yk]
            good = x.ndim == 3 and x.shape[1:] == (30, 4) and y.shape == (x.shape[0], 1)
            if not good:
                problems.append("Fold%d %s/%s shape" % (fold, xk, yk))
            print("        %s/%s consistent: %s | label values: %s" % (xk, yk, good, sorted(set(y.ravel().tolist()))))

print("\n== pretrained checkpoints my_model1..5 ==")
for fold in range(1, 6):
    parts = ["my_model%d.meta" % fold, "my_model%d.index" % fold,
             "my_model%d.data-00000-of-00001" % fold]
    status = ["%s=%s" % (p.split(".", 1)[1], "ok" if os.path.isfile(p) else "MISSING") for p in parts]
    missing = [p for p in parts if not os.path.isfile(p)]
    problems.extend(missing)
    print("my_model%d  %s" % (fold, "  ".join(status)))
# the 'checkpoint' state file is not part of the provided files (restore uses the explicit prefix)
print("'checkpoint' state file: %s (Training.py restores by explicit prefix, so it is not required)"
      % ("present" if os.path.isfile("checkpoint") else "absent"))

# checkpoint variable listing for each fold (reads the .index/.data only)
import tensorflow as tf
for fold in range(1, 6):
    reader = tf.train.NewCheckpointReader("my_model%d" % fold)
    shapes = reader.get_variable_to_shape_map()
    n_params = sum(int(np.prod(s)) for k, s in shapes.items() if "Adam" not in k and "power" not in k)
    print("my_model%d restorable variables: %d (non-optimizer scalar params ~%d)" % (fold, len(shapes), n_params))

print("\nDATASET_OK" if not problems else "\nPROBLEMS: %s" % problems)
sys.exit(0 if not problems else 1)
