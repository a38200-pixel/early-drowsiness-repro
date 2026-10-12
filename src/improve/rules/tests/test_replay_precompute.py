# replay_precompute 시험: 합성 배열과 가짜 검출기, 랜드마크만 쓴다(웹캠, 영상 파일, 모델 파일 없음).
import contextlib
import inspect
import io
import os
import tempfile
import unittest

from improve.hmlstm import timeline as TL
from improve.rules import ear_math
from improve.rules import live_demo
from improve.rules import replay_precompute as RP


def _modules():
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise unittest.SkipTest("cv2 또는 numpy 를 가져올 수 없어 건너뜁니다: %s" % error)
    return cv2, np


class _FakeImutils(object):
    def __init__(self, np, calls):
        self.np, self.calls = np, calls

    def resize(self, frame, width=None):
        self.calls.append(("resize", width))
        return self.np.zeros((253, width, 3), dtype=self.np.uint8)


class _FakeShape(object):
    pass


def _landmarks(np):
    shape = np.zeros((68, 2), dtype=np.int64)
    for start, end in (ear_math.LEFT_EYE_INDEX, ear_math.RIGHT_EYE_INDEX):
        base = np.array([[0, 5], [3, 2], [6, 2], [9, 5], [6, 8], [3, 8]], dtype=np.int64)
        shape[start:end] = base + np.array([10 * start, 10])
    return shape


class ProcessFrameTest(unittest.TestCase):
    def _run(self, detections):
        cv2, np = _modules()
        calls = []

        def adjust_gamma(image, gamma=1.0):
            calls.append(("gamma", gamma))
            return image

        def detector(gray, upsample):
            calls.append(("detect", upsample))
            return detections

        shape = _landmarks(np)

        def predictor(gray, rect):
            calls.append(("predict", rect))
            return _FakeShape()

        eye_indexes = (tuple(ear_math.LEFT_EYE_INDEX), tuple(ear_math.RIGHT_EYE_INDEX))
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = RP.process_frame(cv2, np, _FakeImutils(np, calls), adjust_gamma, detector, predictor, lambda _s: shape, eye_indexes, frame)
        return result, calls, shape, np

    def test_constants_and_step_order_follow_live_demo(self):
        result, calls, _shape, _np = self._run(["first", "second"])
        self.assertEqual(calls, [("resize", live_demo.PREPROCESS_WIDTH), ("gamma", live_demo.PREPROCESS_GAMMA), ("detect", 0), ("predict", "first")])
        self.assertTrue(result["face"])

    def test_source_order_matches_live_demo_run(self):
        tokens = ("imutils.resize(", "cv2.cvtColor(", "adjust_gamma(gray", "detector(gray", "predictor(gray", "rects[0]", "ear_from_landmarks(")
        live = inspect.getsource(live_demo.run)
        mine = inspect.getsource(RP.process_frame)
        order = lambda text: sorted(tokens, key=lambda token: text.index(token))
        self.assertEqual(order(live), order(mine))
        self.assertEqual(order(mine), list(tokens))
        self.assertIn("width=PREPROCESS_WIDTH", live)
        self.assertIn("width=LD.PREPROCESS_WIDTH", mine)
        self.assertIn("gamma=PREPROCESS_GAMMA", live)
        self.assertIn("gamma=LD.PREPROCESS_GAMMA", mine)
        self.assertIn("detector(gray, 0)", live)
        self.assertEqual(RP.UPSAMPLE, 0)
        self.assertIn("detector(gray, UPSAMPLE)", mine)

    def test_ear_and_scaled_eye_polygons(self):
        result, _calls, shape, np = self._run(["rect"])
        landmarks = [(int(x), int(y)) for x, y in shape]
        self.assertAlmostEqual(result["ear"], ear_math.ear_from_landmarks(landmarks)[2])
        scale = 640 / 450.0
        left_start, left_end = ear_math.LEFT_EYE_INDEX
        self.assertEqual(result["left"], (shape[left_start:left_end] * scale).astype(np.int32).tolist())
        self.assertEqual(len(result["right"]), RP.EYE_POINTS)

    def test_no_face_gives_nulls(self):
        result, calls, _shape, _np = self._run([])
        self.assertEqual(result, {"face": False, "ear": None, "left": None, "right": None})
        self.assertNotIn("predict", [name for name, _value in calls])


class DocumentTest(unittest.TestCase):
    def _document(self):
        face = {"face": True, "ear": 0.3, "left": [[1, 2]] * 6, "right": [[3, 4]] * 6}
        none = {"face": False, "ear": None, "left": None, "right": None}
        rotation = {"requested": "auto", "metadata_angle": 270, "applied_angle": 270, "double_rotation_guard": False}
        return RP.build_document("a.mp4", 29.97, [face, none], rotation, (1080, 1920))

    def test_valid_document_passes_and_records_the_preprocess_constants(self):
        document = RP.validate_document(self._document())
        self.assertEqual(document["preprocess"], {"width": live_demo.PREPROCESS_WIDTH, "gamma": live_demo.PREPROCESS_GAMMA, "upsample": 0})
        self.assertEqual(document["frames_read"], 2)

    def test_bad_documents_are_rejected(self):
        broken = []
        document = self._document(); document["frames"][0]["ear"] = None; broken.append(document)
        document = self._document(); document["frames"][0]["left"] = [[1, 2]] * 5; broken.append(document)
        document = self._document(); document["frames"][1]["ear"] = 0.2; broken.append(document)
        document = self._document(); document["frames_read"] = 5; broken.append(document)
        document = self._document(); document["rotation"]["applied_angle"] = 45; broken.append(document)
        document = self._document(); document["video_name"] = "dir/a.mp4"; broken.append(document)
        document = self._document(); document["fps"] = 0; broken.append(document)
        for document in broken:
            with self.assertRaises(RP.PrecomputeError):
                RP.validate_document(document)

    def test_output_goes_only_to_a_new_path_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "rules.json")
            TL.write_json_document(path, self._document(), RP.validate_document)
            with self.assertRaises(TL.TimelineError):
                TL.write_json_document(path, self._document(), RP.validate_document)          # 덮어쓰기 거부
        inside = os.path.join(os.path.dirname(os.path.abspath(RP.__file__)), "never_written.json")
        with self.assertRaises(TL.TimelineError):
            TL.write_json_document(inside, self._document(), RP.validate_document)            # 저장소 안 거부
        self.assertFalse(os.path.exists(inside))


class ArgumentTest(unittest.TestCase):
    def test_parse_args(self):
        arguments = RP.parse_args(["--video", "v.mp4", "--predictor", "p.dat", "--out", "o.json"])
        self.assertEqual(arguments.rotate, "0")
        arguments = RP.parse_args(["--video", "v.mp4", "--predictor", "p.dat", "--out", "o.json", "--rotate", "auto"])
        self.assertEqual(arguments.rotate, "auto")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                RP.parse_args(["--video", "v.mp4", "--predictor", "p.dat", "--out", "o.json", "--rotate", "45"])

    def test_no_image_saving_calls(self):
        with open(RP.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for token in ("imwrite", "imsave", "VideoWriter", "ImageGrab", "save_frame"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
