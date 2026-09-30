"""Steps 4-5: evaluate the provided pretrained my_model1..5 without modifying Training.py.

Training.py is read as text and executed in memory. Only the module-level switch
``load=False`` is flipped to ``load=True`` (restore + one evaluation epoch, no saving).
The original metric functions are wrapped so that BSRE / VRE return values can be recorded.

Usage:  python reproduction_eval.py [num_folds]      (default 5; use 1 for a Fold1-only check)
"""
import contextlib
import csv
import hashlib
import glob
import io
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "third_party", "ghoddoosian")   # vendored upstream code; it opens .npy/my_model* by relative path
MANIFEST = os.path.join(REPO, "configs", "protected_manifest.sha256")
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True                             # keep third_party free of __pycache__
# regression re-run goes to its own folder so the archived original results are never overwritten
ART = os.path.join(REPO, "experiments", "00_reproduction", "regression_eval")
os.makedirs(ART, exist_ok=True)
NUM_FOLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# Known Fold1 values from the earlier manual run (REPRODUCTION_LOG.md)
FOLD1_REFERENCE = {"bsa_loss": 15.711236, "bsa_acc": 0.4838455476753349, "va": 0.6363636363636364,
                   "bsre": 3.185322, "vre": 2.419539}


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


before = {f: sha256(f) for f in protected_files()}
assert len(before) == 44 and all(before[f] == d for f, d in manifest().items()), \
    "vendored files differ from configs/protected_manifest.sha256"

# ---- load Training.py in memory and split definitions from the fold loop -------------
with open(os.path.join(ROOT, "Training.py"), "r") as f:
    src = f.read()
lines = src.splitlines()
loop_start = next(i for i, l in enumerate(lines) if l.startswith("load=False"))
head = "\n".join(lines[:loop_start]) + "\n"
tail = "\n".join(lines[loop_start:]) + "\n"
tail = tail.replace("load=False", "load=True", 1)                       # restore + evaluate only
tail = tail.replace("for i in range(5):", "for i in range(%d):" % NUM_FOLDS, 1)

ns = {"__name__": "training_inmemory"}
stdout_log = io.StringIO()
records = []          # one dict per fold, filled by the wrappers
current = {}

with contextlib.redirect_stdout(stdout_log):
    exec(compile(head, "Training.py[defs]", "exec"), ns)

orig_blink, orig_video, orig_train = ns["regression_per_blink"], ns["regression_per_video"], ns["Train"]


def wrap_blink(*a, **k):
    v = orig_blink(*a, **k)
    current["bsre"] = float(v)
    return v


def wrap_video(*a, **k):
    v = orig_video(*a, **k)
    current["vre"] = float(v)
    return v


def wrap_train(*a, **k):
    current.clear()
    out = orig_train(*a, **k)
    x, y, yy, y_test, yy_test, y_v = out
    current.update(bsa_loss=float(y_test[0]), bsa_acc=float(yy_test[0]), va=float(y_v[0]))
    records.append(dict(current))
    return out


ns["regression_per_blink"], ns["regression_per_video"], ns["Train"] = wrap_blink, wrap_video, wrap_train

with contextlib.redirect_stdout(stdout_log):
    exec(compile(tail, "Training.py[loop,load=True]", "exec"), ns)

with open(os.path.join(ART, "pretrained_eval_stdout.log"), "w") as f:
    f.write(stdout_log.getvalue())

# ---- integrity check: nothing protected may have changed -----------------------------
after = {f: sha256(f) for f in protected_files()}
changed = [os.path.basename(f) for f in before if before[f] != after.get(f)]
added = [n for n in os.listdir(ROOT) if n != "__pycache__" and os.path.join(ROOT, n) not in before]
print("protected files checked: %d | changed: %s | added: %s" % (len(before), changed or "none", added or "none"))
assert not changed and not added, "protected files were modified!"

# ---- results table --------------------------------------------------------------------
cols = ["bsa_loss", "bsa_acc", "va", "bsre", "vre"]
header = ["Fold", "BSA_loss", "BSA_accuracy", "VA", "BSRE", "VRE"]
print("\n%-6s %12s %14s %10s %10s %10s" % tuple(header))
for i, r in enumerate(records):
    print("%-6d %12.6f %14.6f %10.6f %10.6f %10.6f" % ((i + 1,) + tuple(r[c] for c in cols)))
arr = np.array([[r[c] for c in cols] for r in records])
mean, std = arr.mean(axis=0), arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(len(cols))
print("%-6s %12.6f %14.6f %10.6f %10.6f %10.6f" % (("mean",) + tuple(mean)))
print("%-6s %12.6f %14.6f %10.6f %10.6f %10.6f" % (("std",) + tuple(std)))

if NUM_FOLDS == 5:
    with open(os.path.join(ART, "pretrained_5fold_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, r in enumerate(records):
            w.writerow([i + 1] + ["%.6f" % r[c] for c in cols])
        w.writerow(["mean"] + ["%.6f" % v for v in mean])
        w.writerow(["std(ddof=1)"] + ["%.6f" % v for v in std])
    with open(os.path.join(ART, "pretrained_5fold_summary.txt"), "w") as f:
        f.write("Pretrained my_model1..5 evaluation (Training.py executed in memory with load=True)\n")
        f.write("Std is the sample standard deviation (ddof=1) across the 5 folds.\n\n")
        f.write("%-12s %12s %14s %10s %10s %10s\n" % tuple(header))
        for i, r in enumerate(records):
            f.write("%-12d %12.6f %14.6f %10.6f %10.6f %10.6f\n" % ((i + 1,) + tuple(r[c] for c in cols)))
        f.write("%-12s %12.6f %14.6f %10.6f %10.6f %10.6f\n" % (("mean",) + tuple(mean)))
        f.write("%-12s %12.6f %14.6f %10.6f %10.6f %10.6f\n" % (("std",) + tuple(std)))
        f.write("\nprotected files changed: none (%d hashed before/after)\n" % len(before))
    print("\nwrote pretrained_5fold_results.csv / pretrained_5fold_summary.txt")

r1 = records[0]
print("\nFold1 vs earlier reference:")
ok = True
for c in cols:
    d = abs(r1[c] - FOLD1_REFERENCE[c])
    ok = ok and d < 1e-4
    print("  %-9s now %.6f ref %.6f |diff| %.2e" % (c, r1[c], FOLD1_REFERENCE[c], d))
print("FOLD1_MATCHES_REFERENCE" if ok else "FOLD1_DIFFERS_FROM_REFERENCE")
