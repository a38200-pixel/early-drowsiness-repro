# 영상 회전 처리(video_rotation.py) 시험. 합성 MP4 박스 바이트와 작은 배열만 쓴다. 영상 파일과 cv2 를 열지 않는다.
# 기대값은 정의에서 먼저 정했고 코드 결과에 맞추지 않았다.
#   행렬 (a, b, c, d)는 점 (x, y)를 (a*x + c*y, b*x + d*y)로 옮기고 y 는 아래 방향이다.
#   표시를 위해 저장 프레임에 시계 방향으로 적용할 각도: (1,0,0,1) -> 0, (0,1,-1,0) -> 90, (-1,0,0,-1) -> 180, (0,-1,1,0) -> 270.
#   검산(270): (x, y) -> (y, -x). 저장 프레임의 위쪽 가장자리(y=0)는 표시에서 왼쪽 가장자리가 되고 이는 시계 방향 270도(= 반시계 90도)이다.
#   작은 배열 [[1,2,3],[4,5,6]] 의 시계 방향 회전 결과(손으로 계산):
#     90  -> [[4,1],[5,2],[6,3]]   (첫 열을 아래에서 위로 읽어 첫 행이 된다)
#     180 -> [[6,5,4],[3,2,1]]
#     270 -> [[3,6],[2,5],[1,4]]   (위쪽 행이 왼쪽 열이 되고, 오른쪽 위 3 이 왼쪽 위로 간다)
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

from improve.rules import video_rotation as R

MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(R.__file__)), "video_rotation.py")

IDENTITY = (1, 0, 0, 1)
ROTATE_90 = (0, 1, -1, 0)
ROTATE_180 = (-1, 0, 0, -1)
ROTATE_270 = (0, -1, 1, 0)


def fixed(value):
    return int(round(value * 65536))


def box(box_type, payload, mode="normal"):
    if mode == "large":
        return struct.pack(">I4sQ", 1, box_type, 16 + len(payload)) + payload
    if mode == "size0":
        return struct.pack(">I4s", 0, box_type) + payload
    return struct.pack(">I4s", 8 + len(payload), box_type) + payload


def tkhd_payload(version, matrix, width=1920.0, height=1080.0):
    a, b, c, d = matrix
    # version 0: 버전/플래그 4 + 시각 4 x 2 + 트랙 번호 4 + 예약 4 + 길이 4 = 24, 이어서 예약 8, layer 2, 그룹 2, 볼륨 2, 예약 2 -> 행렬 시작 40
    # version 1: 버전/플래그 4 + 시각 8 x 2 + 트랙 번호 4 + 예약 4 + 길이 8 = 36, 이어서 16 -> 행렬 시작 52
    head = bytes([version]) + b"\x00\x00\x00" + b"\x00" * (20 if version == 0 else 32) + b"\x00" * 16
    body = struct.pack(">9i", fixed(a), fixed(b), 0, fixed(c), fixed(d), 0, 0, 0, 0x40000000)
    return head + body + struct.pack(">II", fixed(width), fixed(height))


def hdlr_payload(handler):
    return b"\x00" * 8 + handler + b"\x00" * 12 + b"name\x00"


def trak(handler, matrix, version=0, width=1920.0, height=1080.0, mode="normal"):
    mdia = box(b"mdia", box(b"mdhd", b"\x00" * 24) + box(b"hdlr", hdlr_payload(handler)))
    return box(b"trak", box(b"tkhd", tkhd_payload(version, matrix, width, height)) + mdia, mode)


def moov(*traks, mode="normal"):
    return box(b"moov", box(b"mvhd", b"\x00" * 100) + b"".join(traks), mode)


FTYP = box(b"ftyp", b"isom\x00\x00\x00\x00isom")
MDAT = box(b"mdat", b"\x11" * 64)


class RotationFileTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self._count = 0

    def write(self, data):
        self._count += 1
        path = os.path.join(self._directory.name, "sample%d.mp4" % self._count)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def read(self, data):
        return R.read_rotation(self.write(data))

    def assert_ok(self, result, angle):
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["angle"], angle)


class AngleTableTest(RotationFileTestCase):
    def test_four_angles(self):
        for matrix, angle in ((IDENTITY, 0), (ROTATE_90, 90), (ROTATE_180, 180), (ROTATE_270, 270)):
            with self.subTest(matrix=matrix):
                result = self.read(FTYP + moov(trak(b"vide", matrix)) + MDAT)
                self.assert_ok(result, angle)
                for got, expected in zip(result["matrix"], matrix):
                    self.assertAlmostEqual(got, expected, places=6)
                self.assertEqual((result["tkhd_width"], result["tkhd_height"]), (1920.0, 1080.0))

    def test_result_keys(self):
        result = self.read(FTYP + moov(trak(b"vide", ROTATE_270)) + MDAT)
        self.assertEqual(sorted(result), ["angle", "matrix", "status", "tkhd_height", "tkhd_width"])

    def test_small_deviation_inside_tolerance(self):
        result = self.read(FTYP + moov(trak(b"vide", (0.001, -1.0005, 0.9995, -0.001))) + MDAT)
        self.assert_ok(result, 270)

    def test_tkhd_size_values(self):
        result = self.read(FTYP + moov(trak(b"vide", ROTATE_270, width=640.0, height=360.0)) + MDAT)
        self.assertEqual((result["tkhd_width"], result["tkhd_height"]), (640.0, 360.0))


class UnsupportedMatrixTest(RotationFileTestCase):
    def assert_unsupported(self, matrix):
        result = self.read(FTYP + moov(trak(b"vide", matrix)) + MDAT)
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["angle"])

    def test_mirror_matrices(self):
        for matrix in ((-1, 0, 0, 1), (1, 0, 0, -1), (0, 1, 1, 0), (0, -1, -1, 0)):
            with self.subTest(matrix=matrix):
                self.assert_unsupported(matrix)

    def test_off_axis_matrix(self):
        self.assert_unsupported((0.7071, 0.7071, -0.7071, 0.7071))

    def test_scaled_matrix(self):
        self.assert_unsupported((2, 0, 0, 2))

    def test_deviation_outside_tolerance(self):
        self.assert_unsupported((0.0, -1.0, 1.05, 0.0))


class BoxLayoutTest(RotationFileTestCase):
    def test_tkhd_version_1(self):
        for matrix, angle in ((IDENTITY, 0), (ROTATE_90, 90), (ROTATE_270, 270)):
            with self.subTest(matrix=matrix):
                self.assert_ok(self.read(FTYP + moov(trak(b"vide", matrix, version=1)) + MDAT), angle)

    def test_version_1_and_0_give_same_matrix(self):
        first = self.read(FTYP + moov(trak(b"vide", ROTATE_270, version=0)) + MDAT)
        second = self.read(FTYP + moov(trak(b"vide", ROTATE_270, version=1)) + MDAT)
        self.assertEqual(first, second)

    def test_unknown_tkhd_version_is_unsupported(self):
        result = self.read(FTYP + moov(trak(b"vide", ROTATE_270, version=2)) + MDAT)
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["angle"])

    def test_64bit_sizes(self):
        self.assert_ok(self.read(FTYP + box(b"mdat", b"\x11" * 32, mode="large") + moov(trak(b"vide", ROTATE_90), mode="large")), 90)
        self.assert_ok(self.read(FTYP + moov(trak(b"vide", ROTATE_270, mode="large"))), 270)

    def test_size_zero_box_extends_to_end_of_file(self):
        # moov 가 마지막 박스이고 크기 0(파일 끝까지)
        self.assert_ok(self.read(FTYP + MDAT + moov(trak(b"vide", ROTATE_270), mode="size0")), 270)

    def test_moov_before_and_after_mdat(self):
        self.assert_ok(self.read(FTYP + moov(trak(b"vide", ROTATE_270)) + MDAT), 270)
        self.assert_ok(self.read(FTYP + MDAT + moov(trak(b"vide", ROTATE_270))), 270)

    def test_audio_track_first_is_ignored(self):
        audio = trak(b"soun", ROTATE_90)       # 오디오 트랙의 행렬은 쓰지 않는다
        self.assert_ok(self.read(FTYP + moov(audio, trak(b"vide", ROTATE_270)) + MDAT), 270)
        self.assert_ok(self.read(FTYP + moov(audio, trak(b"vide", IDENTITY)) + MDAT), 0)

    def test_first_video_track_is_used(self):
        self.assert_ok(self.read(FTYP + moov(trak(b"vide", ROTATE_90), trak(b"vide", ROTATE_270)) + MDAT), 90)


class NoTrackAndTruncationTest(RotationFileTestCase):
    def test_no_video_track(self):
        result = self.read(FTYP + moov(trak(b"soun", IDENTITY)) + MDAT)
        self.assertEqual(result["status"], "no_video_track")
        self.assertIsNone(result["angle"])

    def test_no_moov(self):
        self.assertEqual(self.read(FTYP + MDAT)["status"], "no_video_track")

    def test_empty_file(self):
        self.assertEqual(self.read(b"")["status"], "no_video_track")

    def test_truncated_inside_moov_is_not_ok(self):
        data = FTYP + moov(trak(b"vide", ROTATE_270)) + MDAT
        moov_start = len(FTYP)
        for cut in (moov_start + 4, moov_start + 40, len(FTYP) + len(moov(trak(b"vide", ROTATE_270))) - 10):
            with self.subTest(cut=cut):
                result = self.read(data[:cut])
                self.assertNotEqual(result["status"], "ok")
                self.assertIsNone(result["angle"])

    def test_truncated_mdat_after_moov_is_still_readable(self):
        # moov 를 읽은 뒤에는 mdat 를 보지 않으므로 mdat 가 잘려도 각도를 읽는다
        data = FTYP + moov(trak(b"vide", ROTATE_270)) + MDAT
        self.assert_ok(self.read(data[:-30]), 270)

    def test_truncated_mdat_before_moov_is_unsupported(self):
        data = FTYP + MDAT + moov(trak(b"vide", ROTATE_270))
        result = self.read(data[:len(FTYP) + 20])
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["angle"])

    def test_box_size_smaller_than_header_is_unsupported(self):
        bad = struct.pack(">I4s", 4, b"free") + b"\x00" * 8
        self.assertEqual(self.read(FTYP + bad)["status"], "unsupported")

    def test_missing_file_raises(self):
        with self.assertRaises(OSError):
            R.read_rotation(os.path.join(self._directory.name, "none.mp4"))


class ApplyRotationTest(unittest.TestCase):
    def setUp(self):
        self.frame = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

    def test_pixel_positions_clockwise(self):
        expected = {
            90: [[4, 1], [5, 2], [6, 3]],
            180: [[6, 5, 4], [3, 2, 1]],
            270: [[3, 6], [2, 5], [1, 4]],
        }
        for angle, rows in expected.items():
            with self.subTest(angle=angle):
                self.assertEqual(R.apply_rotation(self.frame, angle).tolist(), rows)

    def test_angle_zero_returns_same_object(self):
        self.assertIs(R.apply_rotation(self.frame, 0), self.frame)

    def test_90_and_270_swap_height_and_width(self):
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        self.assertEqual(R.apply_rotation(color, 90).shape, (3, 2, 3))
        self.assertEqual(R.apply_rotation(color, 270).shape, (3, 2, 3))
        self.assertEqual(R.apply_rotation(color, 180).shape, (2, 3, 3))
        self.assertEqual(R.apply_rotation(color, 0).shape, (2, 3, 3))

    def test_channels_follow_pixels(self):
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        color[0, 0] = (10, 20, 30)               # 왼쪽 위
        self.assertEqual(R.apply_rotation(color, 90)[0, 1].tolist(), [10, 20, 30])     # 시계 방향 90: 오른쪽 위
        self.assertEqual(R.apply_rotation(color, 180)[1, 2].tolist(), [10, 20, 30])    # 오른쪽 아래
        self.assertEqual(R.apply_rotation(color, 270)[2, 0].tolist(), [10, 20, 30])    # 왼쪽 아래

    def test_four_quarter_turns_restore_the_frame(self):
        result = self.frame
        for _ in range(4):
            result = R.apply_rotation(result, 90)
        self.assertEqual(result.tolist(), self.frame.tolist())
        self.assertEqual(R.apply_rotation(R.apply_rotation(self.frame, 90), 270).tolist(), self.frame.tolist())

    def test_result_is_contiguous(self):
        non_contiguous = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::-1]
        self.assertFalse(non_contiguous.flags["C_CONTIGUOUS"])
        for angle in (90, 180, 270):
            with self.subTest(angle=angle):
                self.assertTrue(R.apply_rotation(self.frame, angle).flags["C_CONTIGUOUS"])
                self.assertTrue(R.apply_rotation(non_contiguous, angle).flags["C_CONTIGUOUS"])

    def test_dtype_preserved_and_input_not_modified(self):
        before = self.frame.copy()
        for angle in (90, 180, 270):
            self.assertEqual(R.apply_rotation(self.frame, angle).dtype, np.uint8)
        self.assertEqual(self.frame.tolist(), before.tolist())

    def test_invalid_angles(self):
        for angle in (1, 45, -90, 360, 450, None, "90", True):
            with self.subTest(angle=angle):
                with self.assertRaises(ValueError):
                    R.apply_rotation(self.frame, angle)


class PlanRotationTest(RotationFileTestCase):
    def test_auto_reads_metadata_angle(self):
        for matrix, angle in ((IDENTITY, 0), (ROTATE_90, 90), (ROTATE_270, 270)):
            with self.subTest(matrix=matrix):
                path = self.write(FTYP + moov(trak(b"vide", matrix)) + MDAT)
                self.assertEqual(R.plan_rotation("auto", path), (angle, angle))

    def test_auto_stops_for_180(self):
        path = self.write(FTYP + moov(trak(b"vide", ROTATE_180)) + MDAT)
        with self.assertRaises(R.RotationError):
            R.plan_rotation("auto", path)

    def test_auto_stops_when_metadata_is_not_ok(self):
        for data in (FTYP + moov(trak(b"vide", (-1, 0, 0, 1))) + MDAT,      # 반사
                     FTYP + moov(trak(b"soun", IDENTITY)) + MDAT,          # 비디오 트랙 없음
                     FTYP + MDAT):                                         # moov 없음
            with self.subTest(size=len(data)):
                with self.assertRaises(R.RotationError):
                    R.plan_rotation("auto", self.write(data))

    def test_numeric_request_does_not_read_the_file(self):
        missing = os.path.join(self._directory.name, "none.mp4")
        for requested, angle in (("0", 0), ("90", 90), ("180", 180), ("270", 270), (90, 90)):
            with self.subTest(requested=requested):
                self.assertEqual(R.plan_rotation(requested, missing), (angle, None))

    def test_unknown_request(self):
        for requested in ("45", "", "AUTO", None):
            with self.subTest(requested=requested):
                with self.assertRaises(R.RotationError):
                    R.plan_rotation(requested, "unused")


class GuardTest(unittest.TestCase):
    def test_normal_frame_keeps_metadata_angle(self):
        # 저장 모양(높이 1080, 너비 1920)이 CAP 속성(너비 1920, 높이 1080)과 같으면 회전하지 않은 프레임이다
        self.assertEqual(R.guard_applied_angle("auto", 270, (1080, 1920, 3), 1920.0, 1080.0), (270, False))
        self.assertEqual(R.guard_applied_angle("auto", 90, (1080, 1920, 3), 1920.0, 1080.0), (90, False))

    def test_already_rotated_frame_is_not_rotated_again(self):
        # 프레임이 (높이 1920, 너비 1080)로 이미 뒤집힌 모양이면 적용 각도 0 과 보호 작동
        self.assertEqual(R.guard_applied_angle("auto", 270, (1920, 1080, 3), 1920.0, 1080.0), (0, True))
        self.assertEqual(R.guard_applied_angle("auto", 90, (1920, 1080, 3), 1920.0, 1080.0), (0, True))

    def test_guard_only_for_auto_and_quarter_turns(self):
        self.assertEqual(R.guard_applied_angle("270", 270, (1920, 1080, 3), 1920.0, 1080.0), (270, False))
        self.assertEqual(R.guard_applied_angle("auto", 0, (1920, 1080, 3), 1920.0, 1080.0), (0, False))
        self.assertEqual(R.guard_applied_angle("0", 0, (1920, 1080, 3), 1920.0, 1080.0), (0, False))

    def test_no_guard_without_usable_properties(self):
        self.assertEqual(R.guard_applied_angle("auto", 270, (1920, 1080, 3), 0.0, 0.0), (270, False))
        self.assertEqual(R.guard_applied_angle("auto", 270, (1080, 1080, 3), 1080.0, 1080.0), (270, False))


class RotationLineTest(unittest.TestCase):
    def test_auto_line(self):
        line = R.rotation_line("auto", 270, 270, (1080, 1920, 3), (1920, 1080, 3), False)
        self.assertEqual(line, "requested=auto metadata_angle=270 applied_angle=270 shape_before=1080x1920x3 "
                               "shape_after=1920x1080x3 double_rotation_guard=no\n")

    def test_default_line_and_guard_line(self):
        self.assertEqual(R.rotation_line("0", None, 0, (1080, 1920, 3), (1080, 1920, 3), False),
                         "requested=0 metadata_angle=not_read applied_angle=0 shape_before=1080x1920x3 "
                         "shape_after=1080x1920x3 double_rotation_guard=no\n")
        self.assertEqual(R.rotation_line("auto", 90, 0, (1920, 1080, 3), (1920, 1080, 3), True),
                         "requested=auto metadata_angle=90 applied_angle=0 shape_before=1920x1080x3 "
                         "shape_after=1920x1080x3 double_rotation_guard=yes\n")

    def test_single_line(self):
        line = R.rotation_line("auto", 270, 270, (1, 2, 3), (2, 1, 3), False)
        self.assertEqual(line.count("\n"), 1)
        self.assertTrue(line.endswith("\n"))


class CloneStaticTest(unittest.TestCase):
    """복제본에 회전이 프레임을 읽은 직후, 리사이즈 앞에서 한 번만 적용되는지 소스 텍스트로 점검한다(cv2 를 가져오지 않는다)."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(os.path.dirname(MODULE_PATH), "blink_video_dump.py")
        with open(path, encoding="utf-8") as handle:
            cls.text = handle.read()

    def test_rotation_comes_before_resize(self):
        resize = "frame = imutils.resize(frame, width=450)"
        self.assertEqual(self.text.count(resize), 1)
        self.assertEqual(self.text.count("video_rotation.apply_rotation("), 2)    # 첫 프레임 분기와 이후 프레임 분기
        self.assertLess(self.text.index("video_rotation.apply_rotation("), self.text.index(resize))
        self.assertLess(self.text.index("(grabbed, frame) = stream.read()"), self.text.index("video_rotation.apply_rotation("))

    def test_timestamp_and_dump_calls_unchanged(self):
        self.assertEqual(self.text.count("dump_writer.write_row(dump_index, fps, "), 3)
        self.assertEqual(self.text.count("cv2.CAP_PROP_FPS"), 1)

    def test_rotation_file_is_written_once_in_exclusive_mode(self):
        self.assertEqual(self.text.count('"rotation.txt"'), 3)    # 첫 프레임 쓰기, 시작 전 존재 확인(두 번 사용)
        self.assertIn('"x") as rotation_file', self.text)


class ExistingRotationAgreementTest(unittest.TestCase):
    """재현 트랙의 기존 회전 정의(src/repro/orientation_common.py, 읽기만)와 같은 화소 변환인지 대조한다.

    기존 정의: 방향 이름 "cw90" = 시계 방향 90도, "ccw90" = 반시계 방향 90도(= 시계 방향 270도), "180" = 180도.
    따라서 기대 대응은 apply_rotation 90 = cw90, 270 = ccw90, 180 = 180 이다(정의에서 먼저 정했고 코드 결과에 맞추지 않았다).
    """
    NAME_BY_ANGLE = {90: "cw90", 180: "180", 270: "ccw90"}

    @classmethod
    def setUpClass(cls):
        repro_dir = os.path.abspath(os.path.join(os.path.dirname(MODULE_PATH), "..", "..", "repro"))
        if not os.path.exists(os.path.join(repro_dir, "orientation_common.py")):
            raise unittest.SkipTest("src/repro/orientation_common.py 가 없어 대조를 건너뜁니다.")
        sys.path.insert(0, repro_dir)
        cls._repro_dir = repro_dir
        try:
            import orientation_common
        except ImportError as error:
            sys.path.remove(repro_dir)
            raise unittest.SkipTest("orientation_common 를 가져올 수 없어 대조를 건너뜁니다: %s" % error)
        cls.common = orientation_common

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "_repro_dir") and cls._repro_dir in sys.path:
            sys.path.remove(cls._repro_dir)

    def test_names_follow_the_clockwise_definition(self):
        expressions = {name: self.common.rotation_expression(name) for name in ("cw90", "ccw90", "180")}
        self.assertEqual(expressions["cw90"], "cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)")
        self.assertEqual(expressions["ccw90"], "cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)")
        self.assertEqual(expressions["180"], "cv2.rotate(frame, cv2.ROTATE_180)")
        self.assertEqual(self.common.rotation_expression("none"), "frame")

    def test_same_pixel_transform_as_existing_rotate_frame(self):
        try:
            import cv2  # noqa: F401
        except ImportError as error:
            self.skipTest("cv2 를 가져올 수 없어 건너뜁니다: %s" % error)
        small = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        color = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        hand = {90: [[4, 1], [5, 2], [6, 3]], 180: [[6, 5, 4], [3, 2, 1]], 270: [[3, 6], [2, 5], [1, 4]]}
        for angle, name in self.NAME_BY_ANGLE.items():
            with self.subTest(angle=angle, name=name):
                for frame in (small, color):
                    existing = self.common.rotate_frame(frame, name)
                    new = R.apply_rotation(frame, angle)
                    self.assertEqual(existing.shape, new.shape)
                    self.assertTrue(np.array_equal(existing, new))
                # 기존 함수의 결과가 손계산 기대값과도 같다(두 구현이 같은 방향으로 틀리지 않았음을 확인)
                self.assertEqual(self.common.rotate_frame(small, name).tolist(), hand[angle])

    def test_zero_is_the_same_object_in_both(self):
        small = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        self.assertIs(R.apply_rotation(small, 0), small)
        self.assertIs(self.common.rotate_frame(small, "0"), small)


class ModuleSourceTest(unittest.TestCase):
    def test_no_cv2_or_pil(self):
        with open(MODULE_PATH, encoding="utf-8") as handle:
            lines = [line for line in handle if line.lstrip().startswith(("import ", "from "))]
        text = "".join(lines)
        for token in ("cv2", "PIL", "imageio", "av"):
            with self.subTest(token=token):
                self.assertNotIn("import " + token, text)
                self.assertNotIn("from " + token + " ", text)


if __name__ == "__main__":
    unittest.main()
