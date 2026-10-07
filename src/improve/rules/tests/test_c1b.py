# C1b 시험: (a) ear_dump 의 순수 함수와 쓰기→읽기 왕복, (b) 복제본(blink_video_dump.py)이 원본에서 허용된 변경(CLI, GUI-REMOVED, DUMP)만
# 했는지 줄 단위로 비교, (c) 정적 점검, (d) import 시험. 합성 데이터와 소스 텍스트만 쓴다. 영상, 웹캠, dlib 모델(.dat), SVM(.sav)을 열지 않는다.
# 시험으로 확인할 수 없는 것(실행 단계 검증, 미검증): 덤프 행 수 = 영상 프레임 수, blink 기록이 D0 기록(80행)과 같은지, source 비율.
# 시험 출력 파일은 임시 폴더에만 쓴다.
import ast
import difflib
import hashlib
import importlib
import os
import re
import tempfile
import unittest

from improve.rules import ear_dump, replay

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", "..", ".."))
ORIGINAL = os.path.join(REPO_ROOT, "third_party", "ghoddoosian", "blink_video.py")
CLONE = os.path.join(PACKAGE_DIR, "blink_video_dump.py")
MANIFEST = os.path.join(REPO_ROOT, "configs", "protected_manifest.sha256")
MARKER = re.compile(r"^\s*# \[(CLI|GUI-REMOVED|DUMP)\]")

# 원본에는 있지만 복제본에 없는 줄의 허용 목록: (첫 줄, 끝 줄, 첫 줄에 들어 있어야 하는 문자열, 종류, 사유). 줄 번호는 원본 기준이다.
ALLOWED_REMOVED = [
    (15, 18, "matplotlib.pyplot", "GUI-REMOVED", "그래프와 창을 만드는 가져오기 4줄"),
    (313, 313, "shape_predictor", "CLI", "랜드마크 모델 경로 하드코딩을 --predictor 인자로 교체"),
    (315, 315, "pickle.load", "CLI", "SVM 경로 하드코딩을 --svm 인자로 교체"),
    (334, 343, "tk.Tk()", "GUI-REMOVED", "창과 그래프 생성 10줄"),
    (473, 474, "line.set_ydata", "GUI-REMOVED", "그래프 갱신 2줄(검출 분기)"),
    (476, 476, "cv2.imshow", "GUI-REMOVED", "화면 표시 1줄(검출 분기, 큐에서 꺼내는 475줄은 남김)"),
    (480, 484, "cv2.waitKey", "GUI-REMOVED", "키 입력 확인과 키로 끝내기(검출 분기, 빈 줄 포함)"),
    (555, 556, "line.set_ydata", "GUI-REMOVED", "그래프 갱신 2줄(추적 분기)"),
    (558, 558, "cv2.imshow", "GUI-REMOVED", "화면 표시 1줄(추적 분기, 큐에서 꺼내는 557줄은 남김)"),
    (562, 566, "cv2.waitKey", "GUI-REMOVED", "키 입력 확인과 키로 끝내기(추적 분기, 빈 줄 포함)"),
    (570, 570, "destroyAllWindows", "GUI-REMOVED", "창 닫기 1줄"),
    (579, 581, "output_file = 'alert.txt'", "CLI", "하드코딩된 출력·입력 경로와 최상위 실행 호출을 __main__ 인자 처리로 교체"),
]
ALLOWED_KINDS = ("CLI", "GUI-REMOVED", "DUMP")


def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read().replace("\r\n", "\n").split("\n")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def function_dumps(path):
    """파일에 있는 모든 함수 정의(중첩 포함)의 ast.dump 를 {이름: 덤프}로 돌려준다."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        tree = ast.parse(handle.read())
    return {node.name: ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


class EarDumpFormatTest(unittest.TestCase):
    def test_detected_row(self):
        # 30프레임째, fps 30 -> 30/30 = 1.0초
        self.assertEqual(ear_dump.format_row(30, 30.0, 0.2, 0.25, "detected"), ("1.000000", "0.200000", "0.250000", "detected"))

    def test_tracked_row_and_rounding(self):
        # 1/23.976 = 0.0417083.. -> 소수 6자리 0.041708
        self.assertEqual(ear_dump.format_row(1, 23.976, 0.1, 0.12, "tracked"), ("0.041708", "0.100000", "0.120000", "tracked"))

    def test_none_row_has_empty_ear_cells(self):
        self.assertEqual(ear_dump.format_row(0, 30.0, 0.3, 0.3, "none"), ("0.000000", "", "", "none"))
        # 59/30 = 1.96666.. -> 1.966667. none 이면 EAR 값이 None 이어도 된다.
        self.assertEqual(ear_dump.format_row(59, 30.0, None, None, "none"), ("1.966667", "", "", "none"))

    def test_constants(self):
        self.assertEqual(ear_dump.DUMP_COLUMNS, ("timestamp_s", "ear_left", "ear_right", "source"))
        self.assertEqual(ear_dump.SOURCES, ("detected", "tracked", "none"))

    def test_invalid_arguments(self):
        bad_calls = [
            (0, 30.0, 0.2, 0.2, "camera"),
            (0, 0, 0.2, 0.2, "detected"),
            (0, -1.0, 0.2, 0.2, "detected"),
            (0, float("nan"), 0.2, 0.2, "detected"),
            (0, float("inf"), 0.2, 0.2, "detected"),
            (0, None, 0.2, 0.2, "detected"),
            (0, True, 0.2, 0.2, "detected"),
            (-1, 30.0, 0.2, 0.2, "detected"),
            (1.5, 30.0, 0.2, 0.2, "detected"),
            (True, 30.0, 0.2, 0.2, "detected"),
            (0, 30.0, None, 0.2, "detected"),
            (0, 30.0, 0.2, None, "tracked"),
            (0, 30.0, float("nan"), 0.2, "detected"),
        ]
        for args in bad_calls:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    ear_dump.format_row(*args)


class DumpWriterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = os.path.join(self.temp.name, "ear_dump.csv")

    def write_sample_rows(self):
        with ear_dump.DumpWriter(self.path) as writer:
            writer.write_row(0, 30.0, 0.2, 0.3, "detected")
            writer.write_row(1, 30.0, 0.1, 0.1, "detected")
            writer.write_row(2, 30.0, 0.4, 0.2, "tracked")
            writer.write_row(3, 30.0, None, None, "none")
            writer.write_row(4, 30.0, None, None, "none")
            writer.write_row(5, 30.0, 0.25, 0.35, "detected")
            return writer.rows_written

    def test_round_trip_with_replay_reader(self):
        self.assertEqual(self.write_sample_rows(), 6)
        samples = replay.read_ear_csv(self.path)
        # 기대값: 시각 = 프레임/30, EAR = (왼쪽 + 오른쪽)/2, none 행의 EAR 은 없음(None)
        expected = [(0.0, 0.25, "detected"), (1 / 30.0, 0.10, "detected"), (2 / 30.0, 0.30, "tracked"),
                    (3 / 30.0, None, "none"), (4 / 30.0, None, "none"), (5 / 30.0, 0.30, "detected")]
        self.assertEqual(len(samples), len(expected))
        for got, want in zip(samples, expected):
            self.assertAlmostEqual(got[0], want[0], delta=1e-6)
            if want[1] is None:
                self.assertIsNone(got[1])
            else:
                self.assertAlmostEqual(got[1], want[1], delta=1e-6)
            self.assertEqual(got[2], want[2])

    def test_file_text(self):
        self.write_sample_rows()
        with open(self.path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        self.assertEqual(lines[0], "timestamp_s,ear_left,ear_right,source")
        self.assertEqual(len(lines), 7)                      # 헤더 1 + 프레임 6
        self.assertEqual(lines[1], "0.000000,0.200000,0.300000,detected")
        self.assertEqual(lines[4], "0.100000,,,none")        # 프레임 3 = 3/30 = 0.1초, EAR 칸 비어 있음
        self.assertEqual(lines[5], "0.133333,,,none")

    def test_flushed_after_exit_even_on_error(self):
        with self.assertRaises(RuntimeError):
            with ear_dump.DumpWriter(self.path) as writer:
                writer.write_row(0, 30.0, 0.2, 0.3, "detected")
                raise RuntimeError("시험용 오류")
        with open(self.path, encoding="utf-8") as handle:
            self.assertEqual(len(handle.read().splitlines()), 2)

    def test_existing_file_is_not_overwritten(self):
        self.write_sample_rows()
        with self.assertRaises(FileExistsError):
            with ear_dump.DumpWriter(self.path):
                pass

    def test_path_is_required(self):
        with self.assertRaises(TypeError):
            ear_dump.DumpWriter()
        with self.assertRaises(ValueError):
            ear_dump.DumpWriter("")

    def test_write_outside_with_block(self):
        writer = ear_dump.DumpWriter(self.path)
        with self.assertRaises(RuntimeError):
            writer.write_row(0, 30.0, 0.2, 0.3, "detected")

    def test_invalid_row_does_not_count(self):
        with ear_dump.DumpWriter(self.path) as writer:
            with self.assertRaises(ValueError):
                writer.write_row(0, 30.0, 0.2, 0.3, "camera")
            self.assertEqual(writer.rows_written, 0)


class FidelityTest(unittest.TestCase):
    """복제본은 원본에서 CLI, GUI-REMOVED, DUMP 세 종류의 변경만 해야 한다."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ORIGINAL) or not os.path.exists(CLONE):
            raise unittest.SkipTest("원본 또는 복제본 파일이 없어 충실성 시험을 건너뜁니다.")
        cls.start_sha = sha256_of(ORIGINAL)
        cls.original = read_lines(ORIGINAL)
        cls.clone = read_lines(CLONE)
        matcher = difflib.SequenceMatcher(None, cls.original, cls.clone, autojunk=False)
        cls.opcodes = [op for op in matcher.get_opcodes() if op[0] != "equal"]

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "start_sha"):
            assert sha256_of(ORIGINAL) == cls.start_sha, "시험 중 원본 blink_video.py 가 바뀌었습니다."

    def test_original_sha256_is_unchanged_and_in_manifest(self):
        self.assertEqual(sha256_of(ORIGINAL), self.start_sha)
        if not os.path.exists(MANIFEST):
            self.skipTest("보호 목록이 없어 해시 대조를 건너뜁니다.")
        with open(MANIFEST, encoding="utf-8") as handle:
            entries = dict((line.split(None, 1)[1].strip().lstrip("*"), line.split(None, 1)[0]) for line in handle if line.strip())
        self.assertEqual(entries["third_party/ghoddoosian/blink_video.py"], self.start_sha)

    def test_removed_lines_are_exactly_the_allowed_list(self):
        allowed = {}
        for first, last, text, kind, reason in ALLOWED_REMOVED:
            self.assertIn(kind, ALLOWED_KINDS)
            self.assertTrue(reason)
            self.assertIn(text, self.original[first - 1], "허용 목록의 %d줄 내용이 원본과 다릅니다." % first)
            for number in range(first, last + 1):
                if self.original[number - 1].strip():
                    allowed[number] = kind
        removed = set()
        for tag, i1, i2, _j1, _j2 in self.opcodes:
            if tag in ("delete", "replace"):
                for index in range(i1, i2):
                    if self.original[index].strip():
                        removed.add(index + 1)
        self.assertEqual(sorted(removed - set(allowed)), [], "허용 목록에 없는 줄이 복제본에서 빠졌습니다.")
        self.assertEqual(sorted(set(allowed) - removed), [], "허용 목록에 있지만 복제본에 남아 있는 줄이 있습니다.")

    def test_clone_only_lines_are_inside_marked_blocks(self):
        kinds = []
        for tag, _i1, _i2, j1, j2 in self.opcodes:
            if tag not in ("insert", "replace"):
                continue
            block = [line for line in self.clone[j1:j2]]
            nonblank = [line for line in block if line.strip()]
            if not nonblank:
                continue
            self.assertTrue(MARKER.match(nonblank[0]), "표시 없는 추가 블록: %r" % nonblank[0])
            kinds.append(MARKER.match(nonblank[0]).group(1))
        self.assertTrue(kinds)
        for kind in ("CLI", "GUI-REMOVED", "DUMP"):
            self.assertIn(kind, kinds)

    def test_every_marker_line_starts_a_block(self):
        marker_lines = [i for i, line in enumerate(self.clone) if MARKER.match(line)]
        inserted = set()
        for tag, _i1, _i2, j1, j2 in self.opcodes:
            if tag in ("insert", "replace"):
                inserted.update(range(j1, j2))
        for index in marker_lines:
            self.assertIn(index, inserted, "표시 줄이 변경 블록 밖에 있습니다: %r" % self.clone[index])

    def test_core_functions_are_identical_to_original(self):
        original = function_dumps(ORIGINAL)
        clone = function_dumps(CLONE)
        for name in ("adjust_gamma", "eye_aspect_ratio", "mouth_aspect_ratio", "EMERGENCY", "Linear_Interpolate",
                     "Ultimate_Blink_Check", "Blink_Tracker"):
            with self.subTest(function=name):
                self.assertIn(name, original)
                self.assertEqual(original[name], clone.get(name))


class StaticChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(ORIGINAL) or not os.path.exists(CLONE):
            raise unittest.SkipTest("원본 또는 복제본 파일이 없어 정적 점검을 건너뜁니다.")
        with open(ORIGINAL, encoding="utf-8", errors="replace") as handle:
            cls.original_text = handle.read()
        with open(CLONE, encoding="utf-8", errors="replace") as handle:
            cls.clone_text = handle.read()
        cls.tree = ast.parse(cls.clone_text)

    def test_no_gui_or_image_calls(self):
        for token in ("imwrite", "imshow", "waitKey", "tkinter", "matplotlib", "destroyAllWindows"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.clone_text)

    def test_queue_get_calls_are_kept(self):
        pattern = re.compile(r"Q\.get\(\)")
        self.assertEqual(len(pattern.findall(self.clone_text)), len(pattern.findall(self.original_text)))
        self.assertEqual(len(pattern.findall(self.original_text)), 4)

    def test_emergency_constants_unchanged(self):
        self.assertIn("if ear < 0.21:", self.clone_text)
        self.assertIn("if COUNTER >= 50:", self.clone_text)

    def test_queue_size_unchanged(self):
        self.assertIn("Q = Queue(maxsize=7)", self.clone_text)

    def test_four_required_arguments(self):
        # 기존 시험 1건 수정(사용자 승인 2026-10-07, 결정 I-19): 필수 인자는 여전히 정확히 4개여야 하고, 선택 인자는 --rotate 하나만 허용한다.
        names = {}
        keywords = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                name = node.args[0].s
                required = [kw.value.value for kw in node.keywords if kw.arg == "required"]
                names[name] = required
                keywords[name] = dict((kw.arg, kw.value) for kw in node.keywords)
        self.assertEqual(sorted(names), ["--out-dir", "--predictor", "--rotate", "--svm", "--video"])
        required_names = sorted(name for name, required in names.items() if required == [True])
        self.assertEqual(required_names, ["--out-dir", "--predictor", "--svm", "--video"])
        # --rotate 는 필수가 아니고(required 키워드 없음), 기본값은 "0", 선택지는 {0, 90, 180, 270, auto}이다.
        self.assertEqual(names["--rotate"], [])
        self.assertEqual(keywords["--rotate"]["default"].s, "0")
        self.assertEqual([item.s for item in keywords["--rotate"]["choices"].elts], ["0", "90", "180", "270", "auto"])

    def test_main_guard_and_no_top_level_execution(self):
        allowed_types = (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.If)
        for statement in self.tree.body:
            self.assertIsInstance(statement, allowed_types, "최상위에 실행 코드가 있습니다: %r" % (statement,))
        guards = [s for s in self.tree.body if isinstance(s, ast.If)]
        self.assertEqual(len(guards), 1)
        test = guards[0].test
        self.assertIsInstance(test, ast.Compare)
        self.assertEqual(test.left.id, "__name__")
        self.assertEqual(test.comparators[0].s, "__main__")

    def test_one_dump_row_per_branch(self):
        self.assertEqual(self.clone_text.count('write_row(dump_index, fps, leftEAR, rightEAR, "detected")'), 1)
        self.assertEqual(self.clone_text.count('write_row(dump_index, fps, leftEAR, rightEAR, "tracked")'), 1)
        self.assertEqual(self.clone_text.count('write_row(dump_index, fps, None, None, "none")'), 1)
        self.assertEqual(self.clone_text.count("cv2.CAP_PROP_FPS"), 1)

    def test_no_default_paths_in_clone(self):
        for token in ("shape_predictor_68_face_landmarks.dat", "Trained_SVM_C=1000", "alert.txt"):
            with self.subTest(token=token):
                self.assertNotIn(token, self.clone_text)


class ImportTest(unittest.TestCase):
    def test_import_does_not_run_detection(self):
        try:
            import cv2  # noqa: F401
            import dlib  # noqa: F401
            import imutils  # noqa: F401
        except ImportError as error:
            self.skipTest("dlib, cv2, imutils 중 import 할 수 없는 것이 있어 건너뜁니다: %s" % error)
        module = importlib.import_module("improve.rules.blink_video_dump")
        self.assertTrue(callable(module.blink_detector))
        # 실행되었다면 생겼을 이름이 없어야 한다(최상위 실행이 __main__ 보호 안에 있음).
        for name in ("stream", "output_file", "path", "predictor_path", "svm_path", "dump_writer"):
            self.assertFalse(hasattr(module, name), name)


if __name__ == "__main__":
    unittest.main()
