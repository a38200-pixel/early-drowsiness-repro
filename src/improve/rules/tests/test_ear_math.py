# C1 순수 함수부 시험(합성 랜드마크). 저자 EAR 함수와의 일치는 ast 로 함수 정의만 꺼내 독립 네임스페이스에서 실행해 비교한다.
# 저자 파일 전체를 import 하거나 실행하지 않는다. 의존성(scipy)이나 파일이 없으면 그 시험만 건너뛴다.
import ast
import os
import unittest

from improve.rules import ear_math

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
AUTHOR_FILE = os.path.join(REPO_ROOT, "third_party", "ghoddoosian", "blink_video.py")


def make_eye(vertical, width=3.0):
    """눈 6점: eye[0]=(0,0), eye[3]=(width,0), 위쪽 (1,v) (2,v), 아래쪽 (2,-v) (1,-v)."""
    return [(0.0, 0.0), (1.0, vertical), (2.0, vertical), (width, 0.0), (2.0, -vertical), (1.0, -vertical)]


class EyeAspectRatioTest(unittest.TestCase):
    def test_formula(self):
        # 손계산: A = |eye[1]-eye[5]| = 0.4, B = |eye[2]-eye[4]| = 0.4, C = |eye[0]-eye[3]| = 3 -> (0.4+0.4)/(2*3) = 0.8/6
        self.assertAlmostEqual(ear_math.eye_aspect_ratio(make_eye(0.2, 3.0)), 0.8 / 6.0, places=12)

    def test_cap_at_045(self):
        # 손계산: A = B = 4, C = 3 -> 8/6 = 1.333 이므로 0.45 로 제한(저자 89-90줄)
        self.assertEqual(ear_math.eye_aspect_ratio(make_eye(2.0, 3.0)), 0.45)

    def test_small_width_gives_03(self):
        # C = 0.05 < 0.1 이면 세로 길이와 상관없이 0.3(저자 84-85줄)
        self.assertEqual(ear_math.eye_aspect_ratio(make_eye(0.4, 0.05)), 0.3)
        self.assertEqual(ear_math.eye_aspect_ratio(make_eye(0.0, 0.0)), 0.3)

    def test_width_just_below_and_at_limit(self):
        # C = 0.0999 < 0.1 -> 0.3. C = 0.125 >= 0.1 이고 세로 0 이면 (0+0)/(2*0.125) = 0
        self.assertEqual(ear_math.eye_aspect_ratio(make_eye(0.0, 0.0999)), 0.3)
        self.assertEqual(ear_math.eye_aspect_ratio(make_eye(0.0, 0.125)), 0.0)

    def test_needs_six_points(self):
        with self.assertRaises(ValueError):
            ear_math.eye_aspect_ratio(make_eye(0.2)[:5])


class MeanAndLandmarksTest(unittest.TestCase):
    def test_mean_of_both_eyes(self):
        self.assertAlmostEqual(ear_math.mean_ear(0.2, 0.3), 0.25, places=12)

    def test_index_constants_are_reported_values(self):
        self.assertEqual(ear_math.LEFT_EYE_INDEX, (42, 48))
        self.assertEqual(ear_math.RIGHT_EYE_INDEX, (36, 42))

    def test_eye_slices(self):
        landmarks = [(float(i), 0.0) for i in range(68)]
        left, right = ear_math.eyes_from_landmarks(landmarks)
        self.assertEqual([p[0] for p in left], [42.0, 43.0, 44.0, 45.0, 46.0, 47.0])
        self.assertEqual([p[0] for p in right], [36.0, 37.0, 38.0, 39.0, 40.0, 41.0])

    def test_ear_from_landmarks(self):
        landmarks = [(float(i), 0.0) for i in range(68)]
        landmarks[42:48] = make_eye(0.2, 3.0)    # 왼쪽 EAR = 0.8/6
        landmarks[36:42] = make_eye(0.3, 3.0)    # 오른쪽 EAR = (0.6+0.6)/6 = 0.2
        left, right, mean = ear_math.ear_from_landmarks(landmarks)
        self.assertAlmostEqual(left, 0.8 / 6.0, places=12)
        self.assertAlmostEqual(right, 0.2, places=12)
        self.assertAlmostEqual(mean, (0.8 / 6.0 + 0.2) / 2.0, places=12)

    def test_too_few_landmarks(self):
        with self.assertRaises(ValueError):
            ear_math.eyes_from_landmarks([(0.0, 0.0)] * 47)


class AuthorFunctionAgreementTest(unittest.TestCase):
    """저자 blink_video.py 의 eye_aspect_ratio 정의만 꺼내 같은 입력에서 값이 같은지 본다."""

    @classmethod
    def setUpClass(cls):
        try:
            from scipy.spatial import distance as dist
        except ImportError as error:
            raise unittest.SkipTest("scipy 를 import 할 수 없어 저자 함수 일치 시험을 건너뜁니다: %s" % error)
        if not os.path.exists(AUTHOR_FILE):
            raise unittest.SkipTest("저자 파일이 없어 저자 함수 일치 시험을 건너뜁니다.")
        with open(AUTHOR_FILE, "r", encoding="utf-8", errors="replace") as handle:
            source = handle.read()
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            raise unittest.SkipTest("저자 파일을 파싱하지 못해 건너뜁니다: %s" % error)
        nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "eye_aspect_ratio"]
        if not nodes:
            raise unittest.SkipTest("저자 파일에서 eye_aspect_ratio 정의를 찾지 못해 건너뜁니다.")
        module = ast.Module(body=[nodes[0]])
        ast.fix_missing_locations(module)
        namespace = {"dist": dist}
        exec(compile(module, "<저자 eye_aspect_ratio 정의만>", "exec"), namespace)
        cls.author_function = staticmethod(namespace["eye_aspect_ratio"])

    def test_same_values_on_synthetic_eyes(self):
        for vertical in (0.0, 0.2, 0.5, 2.0):
            for width in (0.0, 0.05, 0.0999, 0.125, 3.0, 40.0):
                with self.subTest(vertical=vertical, width=width):
                    eye = make_eye(vertical, width)
                    self.assertAlmostEqual(ear_math.eye_aspect_ratio(eye), self.author_function(eye), places=12)


if __name__ == "__main__":
    unittest.main()
