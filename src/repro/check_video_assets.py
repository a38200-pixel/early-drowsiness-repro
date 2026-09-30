"""Step 11: check that the dlib landmark model and the blink SVM load (read-only, no video is processed)."""
import os
import pickle
import sys
import warnings

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "third_party", "ghoddoosian")   # the .sav is opened via a relative path
os.chdir(ROOT)
DLIB_PATH = os.environ.get("DLIB_LANDMARK_PATH")           # dlib .dat is not part of the upstream repo (see data/README.md)
if not DLIB_PATH or not os.path.isfile(DLIB_PATH):
    sys.exit("set the environment variable DLIB_LANDMARK_PATH to shape_predictor_68_face_landmarks.dat (see data/README.md)")
warnings.simplefilter("always")
import numpy as np
import dlib

ok = True
for name in [DLIB_PATH, "Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav"]:
    exists = os.path.isfile(name)
    print("%-55s exists=%s size=%s" % (name, exists, os.path.getsize(name) if exists else "-"))
    ok = ok and exists

predictor = dlib.shape_predictor(DLIB_PATH)
detector = dlib.get_frontal_face_detector()
print("dlib shape_predictor loaded: %s | frontal face detector: %s" % (type(predictor).__name__, type(detector).__name__))

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    with open("Trained_SVM_C=1000_gamma=0.1_for 7kNegSample.sav", "rb") as f:
        svm = pickle.load(f)
    for x in w:
        print("SVM load warning:", str(x.message)[:140])
n_feat = svm.support_vectors_.shape[1]
print("SVM: %s C=%s gamma=%s kernel=%s n_features=%d classes=%s" % (
    type(svm).__name__, svm.C, svm.gamma, svm.kernel, n_feat, svm.classes_))
# blink_video.py feeds EAR_series (13 values) -> reshape(1,-1); check the width and prediction
pred = svm.predict(np.full((1, 13), 0.3))
print("dummy open-eye EAR series (13 x 0.30) -> IF_Closed_Eyes = %s" % pred)
pred = svm.predict(np.full((1, 13), 0.10))
print("dummy closed-eye EAR series (13 x 0.10) -> IF_Closed_Eyes = %s" % pred)
print("VIDEO_ASSETS_OK" if ok and n_feat == 13 else "VIDEO_ASSETS_PROBLEM")
