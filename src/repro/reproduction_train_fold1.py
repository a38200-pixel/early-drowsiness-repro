"""Step 7: retrain Fold1 with the original Training.py settings, writing ONLY to
experiments/00_reproduction/fold1_training/.

Training.py is executed in memory. In-memory changes (nothing on disk is edited):
  * for i in range(5)                -> for i in range(1)             (Fold1 only)
  * ii=i                             -> ii=i+1                        (save as my_model1 inside the new dir)
  * variable_path='./'               -> the fold1_training directory  (checkpoint save location)
  * open('./x%d.npy' ...) etc.       -> written into the fold1_training directory
Hyper-parameters, data normalisation, model and loss are untouched. No random seed is set
(the original code sets none), so every run differs slightly, as in the original.

Usage:  python reproduction_train_fold1.py [num_epochs]     (default 80 = original setting)
"""
import csv
import glob
import hashlib
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "third_party", "ghoddoosian")   # vendored upstream code; it opens .npy/my_model* by relative path
MANIFEST = os.path.join(REPO, "configs", "protected_manifest.sha256")
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True                             # keep third_party free of __pycache__
OUT = os.path.join(REPO, "experiments", "00_reproduction", "fold1_training")
NUM_EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
os.makedirs(OUT, exist_ok=True)
if glob.glob(os.path.join(OUT, "my_model1*")):
    sys.exit("refusing to run: %s already contains my_model1* (move it first, never overwrite)" % OUT)
OUT_PREFIX = OUT.replace("\\", "/") + "/"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest():
    """{absolute path: expected sha256} for the 44 vendored files in configs/protected_manifest.sha256."""
    with open(MANIFEST, "r") as f:
        rows = [l.rstrip("\r\n").split("  ", 1) for l in f if l.strip()]
    return {os.path.normpath(os.path.join(REPO, rel)): digest for digest, rel in rows}


def protected_files():
    return sorted(manifest())


class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


before = {f: sha256(f) for f in protected_files()}
assert len(before) == 44 and all(before[f] == d for f, d in manifest().items()), \
    "vendored files differ from configs/protected_manifest.sha256"

with open(os.path.join(ROOT, "Training.py"), "r") as f:
    lines = f.read().splitlines()
loop_start = next(i for i, l in enumerate(lines) if l.startswith("load=False"))
head = "\n".join(lines[:loop_start]) + "\n"
tail = "\n".join(lines[loop_start:]) + "\n"

assert head.count("variable_path='./'") == 1
head = head.replace("variable_path='./'", "variable_path=%r" % OUT_PREFIX)
for old, new in [("for i in range(5):", "for i in range(1):"), ("ii=i    #", "ii=i+1    #"),
                 ("num_epochs=80", "num_epochs=%d" % NUM_EPOCHS)]:
    assert tail.count(old) == 1, old
    tail = tail.replace(old, new)
assert tail.count("open('./") == 6
tail = tail.replace("open('./", "open(%r + '" % OUT_PREFIX)

log = open(os.path.join(OUT, "training_stdout.log"), "w")
sys.stdout = Tee(sys.__stdout__, log)

ns = {"__name__": "training_inmemory"}
exec(compile(head, "Training.py[defs]", "exec"), ns)

per_epoch = {"bsa_acc": [], "va": [], "bsre": [], "vre": []}
partial_path = os.path.join(OUT, "training_history_partial.csv")
with open(partial_path, "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "test_bsa_acc", "va", "bsre", "vre", "elapsed_s"])
t0 = time.time()

orig = {k: ns[k] for k in ["regression_per_blink", "regression_per_video", "vote_accuracy_per_video", "Train",
                           "calc_accuracy_per_batch"]}
n_test = np.load(os.path.join(ROOT, "LabelsTest_30_Fold1.npy")).shape[0]


def wrap_acc(Y, predicts):
    a = orig["calc_accuracy_per_batch"](Y, predicts)
    if len(Y) == n_test:                       # only the full-test-set call is the test BSA
        per_epoch["bsa_acc"].append(float(a))
    return a


def wrap_va(*a, **k):
    v = orig["vote_accuracy_per_video"](*a, **k)
    per_epoch["va"].append(float(v))
    return v


def wrap_bsre(*a, **k):
    v = orig["regression_per_blink"](*a, **k)
    per_epoch["bsre"].append(float(v))
    return v


def wrap_vre(*a, **k):
    v = orig["regression_per_video"](*a, **k)
    per_epoch["vre"].append(float(v))
    n = len(per_epoch["vre"])
    with open(partial_path, "a", newline="") as f:
        csv.writer(f).writerow([n, per_epoch["bsa_acc"][n - 1], per_epoch["va"][n - 1],
                                per_epoch["bsre"][n - 1], v, "%.0f" % (time.time() - t0)])
    return v


final = {}


def wrap_train(*a, **k):
    out = orig["Train"](*a, **k)
    final["out"] = out
    return out


ns.update(calc_accuracy_per_batch=wrap_acc, vote_accuracy_per_video=wrap_va,
          regression_per_blink=wrap_bsre, regression_per_video=wrap_vre, Train=wrap_train)
exec(compile(tail, "Training.py[loop,fold1]", "exec"), ns)

x, y, yy, y_test, yy_test, y_v = final["out"]
with open(os.path.join(OUT, "training_history.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss_sum", "train_moving_acc", "test_bsa_loss", "test_bsa_acc", "va", "bsre", "vre"])
    for e in range(NUM_EPOCHS):
        w.writerow([e + 1, "%.6f" % y[e], "%.6f" % yy[e], "%.6f" % y_test[e], "%.6f" % yy_test[e],
                    "%.6f" % y_v[e], "%.6f" % per_epoch["bsre"][e], "%.6f" % per_epoch["vre"][e]])

after = {f: sha256(f) for f in protected_files()}
changed = [os.path.basename(f) for f in before if before[f] != after.get(f)]
added = [n for n in os.listdir(ROOT) if n != "__pycache__" and os.path.join(ROOT, n) not in before]
print("\nprotected files checked: %d | changed: %s | added: %s" % (len(before), changed or "none", added or "none"))
print("total wall time: %.1f min" % ((time.time() - t0) / 60.0))
print("final epoch: BSA loss %.6f | BSA acc %.6f | VA %.6f | BSRE %.6f | VRE %.6f"
      % (y_test[-1], yy_test[-1], y_v[-1], per_epoch["bsre"][-1], per_epoch["vre"][-1]))
best = int(np.argmax(y_v))
print("best  epoch by VA: %d (VA %.6f, BSA acc %.6f)" % (best + 1, y_v[best], yy_test[best]))
print("TRAIN_DONE" if not changed and not added else "PROTECTED_FILES_CHANGED")
