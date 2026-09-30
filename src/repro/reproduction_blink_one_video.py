"""Step 11: run blink_video.py on ONE video without editing blink_video.py.

blink_video.py hard-codes the two I/O variables at the bottom of the file:
    output_file = 'alert.txt'   # line 579 (opened in append mode 'ab')
    path = ''                   # line 580 (input video)
This wrapper reads the file as text, replaces only those two lines in memory and executes it.
The original opens Tk / OpenCV windows; press any key in the "Frame" window to stop early.

Usage:  python reproduction_blink_one_video.py <video_path> [output_txt]
        default output_txt: experiments/00_reproduction/blink_test/<video_name>_blinks.txt
Env:    DLIB_LANDMARK_PATH (required), RLDD_DATA_DIR (required if <video_path> is relative)
Output columns per blink (5): TOTAL_BLINKS, BLINK_FRAME_FREQ*100, amplitude, duration, velocity
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.join(REPO, "third_party", "ghoddoosian")
os.chdir(ROOT)            # the original loads the .sav via a relative path (the .dat is substituted in memory below)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True

DLIB_PATH = os.environ.get("DLIB_LANDMARK_PATH")   # dlib .dat is not part of the upstream repo (see data/README.md)
if not DLIB_PATH or not os.path.isfile(DLIB_PATH):
    sys.exit("set the environment variable DLIB_LANDMARK_PATH to shape_predictor_68_face_landmarks.dat (see data/README.md)")

if len(sys.argv) < 2:
    sys.exit(__doc__)
video = sys.argv[1]
if not os.path.isabs(video):                       # relative video paths are resolved under RLDD_DATA_DIR
    if not os.environ.get("RLDD_DATA_DIR"):
        sys.exit("set the environment variable RLDD_DATA_DIR to the UTA-RLDD video folder (or pass an absolute video path)")
    video = os.path.join(os.environ["RLDD_DATA_DIR"], video)
video = os.path.abspath(video)
if not os.path.isfile(video):
    sys.exit("video not found: %s" % video)
default_out = os.path.join(REPO, "experiments", "00_reproduction", "blink_test",
                           os.path.splitext(os.path.basename(video))[0] + "_blinks.txt")
out = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else default_out
if os.path.exists(out):
    sys.exit("refusing to append to an existing file: %s" % out)
os.makedirs(os.path.dirname(out), exist_ok=True)

with open(os.path.join(ROOT, "blink_video.py"), "r") as f:
    src = f.read()
for old, new in [("output_file = 'alert.txt'", "output_file = %r" % out), ("path = ''", "path = %r" % video),
                 ("'shape_predictor_68_face_landmarks.dat'", repr(DLIB_PATH))]:
    assert src.count(old) == 1, "expected exactly one occurrence of %r" % old
    src = src.replace(old, new)

print("video : %s\noutput: %s" % (video, out))
exec(compile(src, "blink_video.py[in-memory]", "exec"), {"__name__": "blink_video_inmemory"})
print("finished; blinks written: %s" % (sum(1 for _ in open(out)) if os.path.exists(out) else 0))
