# C3 판정 로그와 사용자 메모 시험(임시 폴더). 프레임·이미지 저장 기능이 없는지, 출력 폴더가 필수 인자인지 본다.
import inspect
import os
import tempfile
import unittest

from improve.rules import session_log

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 프레임·이미지를 저장하는 기능을 가리키는 토큰은 패키지의 모든 파일에서 금지한다.
FORBIDDEN_TOKENS = ("imwrite", "imsave", "VideoWriter", "ImageGrab", "save_frame", "snapshot")
# 영상을 읽는 C1b 추출기 복제본만 cv2 를 쓸 수 있다(영상 읽기와 전처리용이며 저장 호출은 위 토큰으로 막는다). 다른 파일은 금지한다.
IMAGE_LIBRARY_TOKENS = ("cv2", "PIL")
IMAGE_LIBRARY_ALLOWED_FILES = ("blink_video_dump.py", "live_demo.py", "replay_precompute.py", "replay_viewer.py")


class SessionLogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = self.temp.name

    def test_decision_round_trip(self):
        out_dir = os.path.join(self.root, "log")
        log = session_log.SessionLog(out_dir, "run_01")
        first = session_log.make_decision_record(0.0, 0.31, True, "detected", "open", False, 0.21)
        second = session_log.make_decision_record(3.0, 0.1, True, "detected", "closed", True, 0.21)
        log.log_decision(first)
        log.log_decision(second)
        records = session_log.read_jsonl(os.path.join(out_dir, "decisions_run_01.jsonl"))
        self.assertEqual(records, [first, second])
        self.assertEqual(set(records[0]), set(session_log.DECISION_FIELDS))

    def test_memo_round_trip_with_korean_text_and_label(self):
        out_dir = os.path.join(self.root, "log")
        log = session_log.SessionLog(out_dir, "run-02")
        log.add_memo(12.5, "눈을 오래 감았다", "감김")
        log.add_memo(40, "라벨 없음")
        records = session_log.read_jsonl(os.path.join(out_dir, "memos_run-02.jsonl"))
        self.assertEqual(records, [{"timestamp_s": 12.5, "text": "눈을 오래 감았다", "label": "감김"},
                                   {"timestamp_s": 40.0, "text": "라벨 없음", "label": None}])

    def test_out_dir_is_required(self):
        with self.assertRaises(TypeError):
            session_log.SessionLog()
        with self.assertRaises(TypeError):
            session_log.SessionLog(run_id="run1")
        with self.assertRaises(ValueError):
            session_log.SessionLog("", "run1")
        self.assertIs(inspect.signature(session_log.SessionLog.__init__).parameters["out_dir"].default,
                      inspect.Parameter.empty)

    def test_run_id_rejects_names_and_paths(self):
        for bad in ("video.mp4", "../x", "a/b", "a\\b", "", "한글", "x" * 65, None):
            with self.subTest(run_id=bad):
                with self.assertRaises(ValueError):
                    session_log.SessionLog(os.path.join(self.root, "log"), bad)

    def test_existing_log_is_not_appended(self):
        out_dir = os.path.join(self.root, "log")
        session_log.SessionLog(out_dir, "run1")
        with self.assertRaises(FileExistsError):
            session_log.SessionLog(out_dir, "run1")

    def test_bad_inputs(self):
        log = session_log.SessionLog(os.path.join(self.root, "log"), "run1")
        with self.assertRaises(ValueError):
            log.log_decision({"timestamp_s": 0.0})
        with self.assertRaises(ValueError):
            log.add_memo(float("nan"), "text")
        with self.assertRaises(ValueError):
            log.add_memo(1.0, "   ")
        with self.assertRaises(ValueError):
            session_log.make_decision_record(float("inf"), 0.3, True, "detected", "open", False, 0.21)

    def test_files_only_in_out_dir(self):
        out_dir = os.path.join(self.root, "log")
        log = session_log.SessionLog(out_dir, "run1")
        log.add_memo(1.0, "메모")
        self.assertEqual(sorted(os.listdir(out_dir)), ["decisions_run1.jsonl", "memos_run1.jsonl"])
        self.assertEqual(os.listdir(self.root), ["log"])


class NoFrameStorageTest(unittest.TestCase):
    def test_no_frame_or_image_saving_in_session_log(self):
        names = [name.lower() for name, _value in inspect.getmembers(session_log)]
        for token in ("frame", "image", "snapshot"):
            self.assertFalse([name for name in names if token in name], "공개 이름에 %s 가 들어 있습니다." % token)

    def test_no_image_calls_anywhere_in_package(self):
        offenders = []
        for name in sorted(os.listdir(PACKAGE_DIR)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(PACKAGE_DIR, name), encoding="utf-8") as handle:
                source = handle.read()
            for token in FORBIDDEN_TOKENS:
                if token in source:
                    offenders.append((name, token))
            if name not in IMAGE_LIBRARY_ALLOWED_FILES:
                for token in IMAGE_LIBRARY_TOKENS:
                    if token in source:
                        offenders.append((name, token))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
