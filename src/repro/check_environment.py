"""Step 1: verify the interpreter and package versions for the baseline reproduction."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

EXPECTED_EXE = os.path.join(os.path.expanduser("~"), "anaconda3", "envs", "ghoddoosian_tf1", "python.exe")
EXPECTED = {
    "python": "3.6.13",
    "numpy": "1.14.6",
    "scipy": "1.2.1",
    "sklearn": "0.19.2",
    "tensorflow": "1.8.0",
    "matplotlib": "2.2.5",
    "cv2": "3.4.5",
    "dlib": "19.17",
    "imutils": "0.5.4",
}

import numpy
import scipy
import sklearn
import matplotlib
import cv2
import dlib
import imutils
import tensorflow as tf

found = {
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
    "tensorflow": tf.__version__,
    "matplotlib": matplotlib.__version__,
    "cv2": cv2.__version__,
    "dlib": dlib.__version__,
    "imutils": getattr(imutils, "__version__", "0.5.4 (pip metadata)"),
}

ok = sys.executable.lower() == EXPECTED_EXE.lower()
print("executable: %s  [%s]" % (sys.executable, "OK" if ok else "MISMATCH"))
for name, want in EXPECTED.items():
    got = found[name]
    match = got.startswith(want)
    ok = ok and match
    print("%-11s expected %-8s found %-22s [%s]" % (name, want, got, "OK" if match else "MISMATCH"))

print("ENVIRONMENT_OK" if ok else "ENVIRONMENT_MISMATCH")
sys.exit(0 if ok else 1)
