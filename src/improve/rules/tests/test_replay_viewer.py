# replay_viewer 시험: 합성 문서, 가짜 프레임 소스, 가짜 글꼴 그리기만 쓴다(웹캠, 영상 파일, 창, 글꼴 파일 없음).
import contextlib
import io
import unittest

from improve.hmlstm import gauge as GAUGE
from improve.rules import live_demo
from improve.rules import replay_viewer as RV


def _modules():
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise unittest.SkipTest("cv2 또는 numpy 를 가져올 수 없어 건너뜁니다: %s" % error)
    return cv2, np


def _rules_document(ears, fps=10.0, angle=270):
    """ears: 프레임마다 EAR 또는 None(얼굴 없음)."""
    polygon = [[1, 1], [2, 1], [3, 1], [3, 2], [2, 2], [1, 2]]
    frames = [{"face": True, "ear": value, "left": polygon, "right": polygon} if value is not None
              else {"face": False, "ear": None, "left": None, "right": None} for value in ears]
    return {"schema_version": 1, "kind": "replay_rules", "video_name": "a.mp4", "fps": fps, "frames_read": len(frames),
            "rotation": {"requested": "auto", "metadata_angle": angle, "applied_angle": angle, "double_rotation_guard": False},
            "frame_size": [640, 480], "preprocess": {"width": 450, "gamma": 1.5, "upsample": 0}, "frames": frames}


def _ears():
    return [0.30] * 20 + [0.10] * 100 + [None] * 5 + [0.30] * 15 + [0.10] * 40 + [0.30] * 20     # 알람이 두 번 나는 열


def _timeline(windows, blink_frames, fps=10.0, frames_read=200):
    section = {"video_name": "a.mp4", "fps": fps, "frames_read": frames_read, "windows": windows, "blink_frame_indices": blink_frames}
    return {"baseline": section, "target": section}


def _window(k, frame_index, mean, minimum, maximum):
    return {"k": k, "end_row": k, "frame_index": frame_index, "time_sec": frame_index / 10.0, "mean": mean, "min": minimum, "max": maximum,
            "folds": [mean] * 5}


class FakeSource(object):
    def __init__(self, count, fail_at=None):
        self.count, self.fail_at = count, fail_at
        self.position = 0
        self.reads, self.grabs, self.restarts = [], 0, 0

    def read(self):
        if self.position >= self.count or self.position == self.fail_at:
            return False, None
        self.reads.append(self.position)
        self.position += 1
        return True, "frame%d" % (self.position - 1)

    def grab(self):
        if self.position >= self.count:
            return False
        self.grabs += 1
        self.position += 1
        return True

    def restart(self):
        self.restarts += 1
        self.position = 0


class ReplayEngineTest(unittest.TestCase):
    def _direct(self, ears, fps, thresholds=(0.15,)):
        rules = live_demo.LiveRules(thresholds)
        recent = live_demo.RecentMinimum()
        mins = []
        for position, value in enumerate(ears):
            timestamp = position / fps
            if value is None:
                rules.update(timestamp, None, "none")
                mins.append(None)
            else:
                recent.add(timestamp, value)
                rules.update(timestamp, value, "detected")
                mins.append(recent.minimum(timestamp))
        return rules, mins

    def test_replay_matches_feeding_live_rules_directly(self):
        ears = _ears()
        document = _rules_document(ears)
        direct, mins = self._direct(ears, 10.0)
        engine = RV.ReplayEngine(document, (0.15,))
        states = [engine.advance_to(position) for position in range(len(ears))]
        self.assertEqual(engine.rules.alarm_count(0.15), direct.alarm_count(0.15))
        self.assertGreaterEqual(direct.alarm_count(0.15), 1)
        self.assertEqual([a.alarm_time_s for a in engine.rules.machines[0.15].alarms], [a.alarm_time_s for a in direct.machines[0.15].alarms])
        self.assertEqual([state.ear_min for state in states], mins)
        self.assertEqual([state.ear for state in states], ears)
        self.assertAlmostEqual(states[7].time_s, 0.7)

    def test_skipping_screen_frames_gives_the_same_alarms(self):
        ears = _ears()
        direct, _mins = self._direct(ears, 10.0)
        engine = RV.ReplayEngine(_rules_document(ears), (0.15,))
        for position in (3, 40, 41, 90, 160, len(ears) - 1):
            engine.advance_to(position)
        self.assertEqual([a.alarm_time_s for a in engine.rules.machines[0.15].alarms], [a.alarm_time_s for a in direct.machines[0.15].alarms])

    def test_going_back_is_refused_and_reset_starts_over(self):
        engine = RV.ReplayEngine(_rules_document(_ears()), (0.15,))
        engine.advance_to(50)
        with self.assertRaises(ValueError):
            engine.advance_to(10)
        with self.assertRaises(ValueError):
            engine.advance_to(10 ** 6)
        self.assertIs(engine.advance_to(50), engine.last)
        engine.reset()
        self.assertEqual(engine.cursor, 0)
        self.assertEqual(engine.rules.alarm_count(0.15), 0)
        engine.advance_to(0)


class PlayerTest(unittest.TestCase):
    def _player(self, count=50, fail_at=None, fps=10.0):
        engine = RV.ReplayEngine(_rules_document([0.30] * count, fps), (0.15,))
        source = FakeSource(count, fail_at)
        player = RV.Player(source, engine, RV.PlaybackClock(fps), count)
        player.start(100.0)
        return player, source, engine

    def test_first_frame_then_skip_when_slow(self):
        player, source, _engine = self._player()
        first = player.step(100.0)
        self.assertEqual((first.kind, first.index, first.frame, first.skipped), ("frame", 0, "frame0", 0))
        self.assertEqual(player.step(100.0).kind, "idle")                    # 같은 시각에는 새 프레임이 없다
        late = player.step(100.55)                                           # 0.55초 뒤 = 프레임 5
        self.assertEqual((late.kind, late.index, late.frame, late.skipped), ("frame", 5, "frame5", 4))
        self.assertEqual(source.grabs, 4)                                    # 1에서 4번은 디코딩만 건너뜀
        self.assertEqual(late.state.index, 5)
        self.assertEqual(player.engine.cursor, 6)                            # 건너뛴 프레임의 EAR 도 모두 규칙에 들어갔다

    def test_pause_freezes_the_video_time_and_resume_continues(self):
        player, source, _engine = self._player()
        player.step(100.0)
        self.assertEqual(player.handle_key(RV.KEY_SPACE, 100.25), "pause")
        self.assertTrue(player.clock.paused)
        self.assertEqual(player.step(130.0).kind, "idle")                    # 멈춘 동안은 시간이 가지 않는다
        self.assertEqual(player.wait_ms(130.0), RV.MAX_WAIT_MS)
        self.assertEqual(player.handle_key(RV.KEY_SPACE, 130.0), "resume")
        step = player.step(130.1)                                            # 재생 시간 0.25초 + 0.1초 = 0.35초 -> 프레임 3
        self.assertEqual((step.kind, step.index), ("frame", 3))
        self.assertEqual(source.reads, [0, 3])

    def test_restart_resets_source_engine_and_clock(self):
        player, source, engine = self._player()
        player.step(100.0)
        player.step(101.0)
        self.assertEqual(player.handle_key(ord("r"), 105.0), "restart")
        self.assertEqual(source.restarts, 1)
        self.assertEqual((player.shown, player.decoded, engine.cursor), (-1, 0, 0))
        again = player.step(105.0)
        self.assertEqual((again.kind, again.index), ("frame", 0))
        self.assertEqual(player.handle_key(ord("R"), 106.0), "restart")

    def test_quit_and_unknown_keys(self):
        player, _source, _engine = self._player()
        self.assertEqual(player.handle_key(ord("q"), 100.0), "quit")
        self.assertIsNone(player.handle_key(255, 100.0))
        self.assertIsNone(player.handle_key(ord("x"), 100.0))

    def test_end_of_video_and_clamped_target(self):
        player, _source, _engine = self._player(count=5)
        step = player.step(110.0)                                            # 시계는 프레임 100 이지만 영상은 5프레임
        self.assertEqual((step.kind, step.index), ("frame", 4))
        self.assertEqual(player.step(111.0).kind, "idle")
        broken, _source2, _engine2 = self._player(count=50, fail_at=5)
        self.assertEqual(broken.step(100.5).kind, "end")
        self.assertEqual(broken.step(101.0).kind, "idle")

    def test_wait_is_between_one_and_the_maximum(self):
        player, _source, _engine = self._player(fps=10.0)
        player.step(100.0)
        self.assertEqual(player.wait_ms(100.0), RV.MAX_WAIT_MS)              # 다음 프레임까지 0.1초 > 최대값
        self.assertIn(player.wait_ms(100.08), (19, 20))                      # 다음 프레임까지 약 0.02초
        self.assertEqual(player.wait_ms(100.5), 1)                           # 이미 늦었다


class GaugeTest(unittest.TestCase):
    def _recorder(self):
        recorded = []
        return recorded, lambda strip, items: recorded.append([item[0] for item in items])

    def test_state_before_and_after_the_first_window(self):
        document = _timeline([_window(1, 100, 5.0, 4.0, 6.0)], list(range(70, 101)))
        before, after = RV.gauge_state(document, 99), RV.gauge_state(document, 100)
        self.assertEqual(before["stage"], GAUGE.STAGE_COLLECTING)
        self.assertEqual(before["blinks"], 30)
        self.assertEqual(after["stage"], GAUGE.STAGE_VALUE)
        self.assertEqual((after["mean"], after["min"], after["max"]), (5.0, 4.0, 6.0))
        self.assertEqual(RV.gauge_state(document, 0)["blinks"], 0)

    def test_strip_text_is_the_gauge_module_output_and_size_is_fixed(self):
        cv2, np = _modules()
        document = _timeline([_window(1, 100, 5.0, 4.0, 6.0)], list(range(70, 101)))
        recorded, painter = self._recorder()
        for index in (50, 99, 100, 150):
            state = RV.gauge_state(document, index)
            strip = RV.draw_gauge_strip(cv2, np, 640, state, painter)
            self.assertEqual(strip.shape, (RV.GAUGE_H, 640, 3))
            self.assertEqual("\n".join(recorded[-1]), GAUGE.format_gauge_text(state))
            self.assertEqual(recorded[-1][0], GAUGE.LABEL)
        self.assertTrue(recorded[0][1].startswith("수집 중"))
        self.assertEqual(recorded[2][1], "5.0 [4.0-6.0]")
        self.assertIn("창 1", recorded[3][2])

    def test_no_baseline_text_and_cool_background(self):
        cv2, np = _modules()
        document = _timeline([], [])
        recorded, painter = self._recorder()
        state = GAUGE.display_state(document["target"], 5, baseline_missing=True)
        strip = RV.draw_gauge_strip(cv2, np, 640, state, painter)
        self.assertEqual(recorded[-1][1], "기준 없음")
        self.assertEqual(tuple(int(v) for v in strip[1, 5]), RV.GAUGE_BG)
        blue, green, red = RV.GAUGE_BG
        self.assertGreater(blue, red)                                        # 알람의 붉은 계열과 다른 차가운 색

    def test_strip_cache_draws_once_for_the_same_state(self):
        cv2, np = _modules()
        document = _timeline([_window(1, 100, 5.0, 4.0, 6.0)], list(range(70, 101)))
        recorded, painter = self._recorder()
        gauge = RV.GaugeStrip(cv2, np, 640, painter)
        gauge.render(RV.gauge_state(document, 100))
        gauge.render(RV.gauge_state(document, 100))
        self.assertEqual(len(recorded), 1)

    def test_no_zone_names_are_drawn(self):
        cv2, np = _modules()
        document = _timeline([_window(1, 100, 5.0, 4.0, 6.0)], list(range(70, 101)))
        recorded, painter = self._recorder()
        RV.draw_gauge_strip(cv2, np, 640, RV.gauge_state(document, 100), painter)
        text = " ".join(recorded[-1])
        for word in ("깨어", "졸음", "경고", "위험", "주의"):
            self.assertNotIn(word, text)


class ComposeTest(unittest.TestCase):
    def test_screen_is_the_rule_screen_plus_the_gauge_strip(self):
        cv2, np = _modules()
        document = _timeline([_window(1, 10, 5.0, 4.0, 6.0)], list(range(0, 31)))
        engine = RV.ReplayEngine(_rules_document(_ears()), (0.15,))
        state = engine.advance_to(5)
        strip = RV.GaugeStrip(cv2, np, RV.DISPLAY_WIDTH, lambda strip, items: None).render(RV.gauge_state(document, 5))
        for height, width in ((480, 640), (1920, 1080), (1080, 1920)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            image = RV.compose(cv2, np, frame, 5, engine, state, strip)
            expected = live_demo.ROW_MAIN_H + live_demo.STRIP_H + RV.DEFAULT_VIDEO_HEIGHT + RV.TIME_H + RV.GAUGE_H   # 영상 칸은 고정 높이, clean 이라 키 안내 줄이 없다
            self.assertEqual(image.shape, (expected, RV.DISPLAY_WIDTH, 3))
            self.assertTrue((image[-RV.GAUGE_H:] == strip).all())

    def test_alarm_screen_is_the_unchanged_draw_ui_clean_output(self):
        cv2, np = _modules()
        engine = RV.ReplayEngine(_rules_document(_ears()), (0.15,))
        state = engine.advance_to(119)                                       # 첫 알람 뒤
        self.assertEqual(engine.rules.alarm_count(0.15), 1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        strip = np.zeros((RV.GAUGE_H, RV.DISPLAY_WIDTH, 3), dtype=np.uint8)
        image = RV.compose(cv2, np, frame, 119, engine, state, strip)
        scale = RV.DISPLAY_WIDTH / 640.0
        polygons = [(np.asarray(engine.frames[119][key], dtype=np.float64) * scale).astype(np.int32) for key in ("left", "right")]
        expected = live_demo.draw_ui(cv2, np, cv2.resize(frame, (RV.DISPLAY_WIDTH, 480)), polygons, engine.rules, engine.latch,
                                     state.ear, state.ear_min, "", state.time_s, None, None, True)
        self.assertTrue((image[:expected.shape[0]] == expected).all())


class FitAndTransformTest(unittest.TestCase):
    def test_fit_geometry_for_portrait_landscape_and_exact_frames(self):
        self.assertEqual(RV.fit_geometry(1080, 1920, 640, 480), (270, 480, 185, 0))       # 세로 영상: 높이에 맞추고 좌우 검정
        self.assertEqual(RV.fit_geometry(1920, 1080, 640, 480), (640, 360, 0, 60))        # 가로 영상: 폭에 맞추고 위아래 검정
        self.assertEqual(RV.fit_geometry(640, 480, 640, 480), (640, 480, 0, 0))
        self.assertEqual(RV.fit_geometry(10, 10, 640, 480), (480, 480, 80, 0))            # 작은 영상은 칸에 맞춰 커진다
        width, height, offset_x, offset_y = RV.fit_geometry(1080, 1920, 640, 300)
        self.assertEqual((width, height, offset_y), (169, 300, 0))
        self.assertEqual(offset_x, (640 - width) // 2)

    def test_letterbox_places_the_frame_in_the_middle_with_black_bars(self):
        cv2, np = _modules()
        frame = np.full((1920, 1080, 3), 255, dtype=np.uint8)
        cell, scales, offsets = RV.letterbox(cv2, np, frame, 640, 480)
        self.assertEqual(cell.shape, (480, 640, 3))
        self.assertEqual((scales, offsets), ((0.25, 0.25), (185, 0)))
        self.assertEqual(int(cell[240, 320, 0]), 255)
        self.assertEqual(int(cell[240, 50, 0]), 0)
        self.assertEqual(int(cell[240, 600, 0]), 0)
        wide = RV.letterbox(cv2, np, np.full((1080, 1920, 3), 255, dtype=np.uint8), 640, 480)[0]
        self.assertEqual((int(wide[10, 320, 0]), int(wide[240, 320, 0]), int(wide[470, 320, 0])), (0, 255, 0))

    def test_polygon_follows_the_same_scale_and_offset(self):
        cv2, np = _modules()
        points = RV.transform_polygon(np, [[100, 200], [1080, 1920]], (0.25, 0.25), (185, 0))
        self.assertEqual(points.tolist(), [[210, 50], [455, 480]])
        self.assertEqual(points.dtype, np.int32)

    def test_compose_draws_a_portrait_frame_with_black_side_bars(self):
        cv2, np = _modules()
        engine = RV.ReplayEngine(_rules_document(_ears()), (0.15,))
        state = engine.advance_to(5)
        frame = np.full((1920, 1080, 3), 255, dtype=np.uint8)
        strip = np.zeros((RV.GAUGE_H, RV.DISPLAY_WIDTH, 3), dtype=np.uint8)
        image = RV.compose(cv2, np, frame, 5, engine, state, strip)
        row = live_demo.ROW_MAIN_H + live_demo.STRIP_H + 240
        self.assertEqual((int(image[row, 320, 0]), int(image[row, 50, 0]), int(image[row, 600, 0])), (255, 0, 0))


class TimeStripTest(unittest.TestCase):
    def test_time_format_boundaries(self):
        self.assertEqual(RV.format_time(0), "00:00.0")
        self.assertEqual(RV.format_time(59.94), "00:59.9")
        self.assertEqual(RV.format_time(59.95), "01:00.0")                   # 올림이 분으로 넘어간다
        self.assertEqual(RV.format_time(180), "03:00.0")
        self.assertEqual(RV.format_time(3725.26), "62:05.3")
        self.assertEqual(RV.format_time(-1.0), "00:00.0")

    def test_progress_fraction(self):
        self.assertEqual(RV.progress_fraction(0, 100), 0.0)
        self.assertEqual(RV.progress_fraction(99, 100), 1.0)
        self.assertEqual(RV.progress_fraction(50, 101), 0.5)
        self.assertEqual(RV.progress_fraction(0, 1), 1.0)
        self.assertEqual(RV.progress_fraction(500, 100), 1.0)

    def test_time_texts_use_frame_number_over_fps(self):
        self.assertEqual(RV.time_texts(150, 301, 30.0), ("00:05.0 / 00:10.0", "frame 150 / 300"))
        self.assertEqual(RV.time_texts(0, 5733, 30.0)[1], "frame 0 / 5732")

    def test_strip_items_bar_and_pause_marker(self):
        cv2, np = _modules()
        recorded = []
        painter = lambda strip, items: recorded.append([item[0] for item in items])
        half = RV.draw_time_strip(cv2, np, 640, 50, 101, 10.0, False, painter)
        self.assertEqual(half.shape, (RV.TIME_H, 640, 3))
        self.assertEqual(recorded[-1], ["00:05.0 / 00:10.1", "frame 50 / 100"])
        self.assertEqual(tuple(int(v) for v in half[RV.TIME_H - 2, 100]), RV.GAUGE_RANGE)      # 채워진 쪽
        self.assertEqual(tuple(int(v) for v in half[RV.TIME_H - 2, 600]), RV.GAUGE_BAR)        # 비어 있는 쪽
        start = RV.draw_time_strip(cv2, np, 640, 0, 101, 10.0, False, painter)
        self.assertEqual(tuple(int(v) for v in start[RV.TIME_H - 2, 5]), RV.GAUGE_BAR)
        RV.draw_time_strip(cv2, np, 640, 50, 101, 10.0, True, painter)
        self.assertEqual(recorded[-1][-1], RV.PAUSED_TEXT)
        self.assertNotIn(RV.PAUSED_TEXT, recorded[0])

    def test_time_strip_cache_redraws_only_when_the_state_changes(self):
        cv2, np = _modules()
        recorded = []
        strip = RV.TimeStrip(cv2, np, 640, lambda image, items: recorded.append(1))
        strip.render(10, 101, 10.0, False)
        strip.render(10, 101, 10.0, False)
        self.assertEqual(len(recorded), 1)
        strip.render(10, 101, 10.0, True)
        strip.render(11, 101, 10.0, True)
        self.assertEqual(len(recorded), 3)


class LayoutTest(unittest.TestCase):
    def test_default_total_height_fits_the_screen(self):
        cv2, np = _modules()
        engine = RV.ReplayEngine(_rules_document(_ears()), (0.15,))
        state = engine.advance_to(5)
        strip = np.zeros((RV.GAUGE_H, RV.DISPLAY_WIDTH, 3), dtype=np.uint8)
        frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
        default = RV.compose(cv2, np, frame, 5, engine, state, strip)
        self.assertEqual(RV.DEFAULT_VIDEO_HEIGHT, 480)
        self.assertEqual(default.shape, (110 + 28 + 480 + 24 + 96, 640, 3))
        self.assertLessEqual(default.shape[0], 760)
        smaller = RV.compose(cv2, np, frame, 5, engine, state, strip, 360)
        self.assertEqual(smaller.shape[0], default.shape[0] - 120)
        taller = RV.compose(cv2, np, frame, 5, engine, state, strip, 600)
        self.assertEqual(taller.shape[0], default.shape[0] + 120)

    def test_video_height_option(self):
        base = ["--video", "v.mp4", "--rules-json", "r.json", "--timeline-json", "t.json"]
        self.assertEqual(RV.parse_args(base).video_height, 480)
        self.assertEqual(RV.parse_args(base + ["--video-height", "360"]).video_height, 360)
        with contextlib.redirect_stderr(io.StringIO()):
            for bad in ("50", "abc", "5000"):
                with self.assertRaises(SystemExit):
                    RV.parse_args(base + ["--video-height", bad])


class HelperAndArgumentTest(unittest.TestCase):
    def test_resolve_angle(self):
        document = _rules_document([0.3], angle=270)
        self.assertEqual(RV.resolve_angle("auto", document), 270)
        self.assertEqual(RV.resolve_angle("90", document), 90)

    def test_check_alignment(self):
        rules = _rules_document([0.3] * 200)
        self.assertEqual(RV.check_alignment(rules, _timeline([], [], fps=10.0, frames_read=200)), [])
        notes = RV.check_alignment(rules, _timeline([], [], fps=12.0, frames_read=150))
        self.assertEqual(len(notes), 2)

    def test_parse_args_defaults_and_required(self):
        arguments = RV.parse_args(["--video", "v.mp4", "--rules-json", "r.json", "--timeline-json", "t.json"])
        self.assertEqual((arguments.rotate, arguments.thresholds), ("auto", "0.15"))
        self.assertEqual(live_demo.parse_thresholds(arguments.thresholds), (0.15,))
        arguments = RV.parse_args(["--video", "v.mp4", "--rules-json", "r.json", "--timeline-json", "t.json", "--rotate", "90",
                                   "--thresholds", "0.15,0.21"])
        self.assertEqual((arguments.rotate, arguments.thresholds), ("90", "0.15,0.21"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                RV.parse_args(["--video", "v.mp4"])
            with self.assertRaises(SystemExit):
                RV.parse_args(["--video", "v.mp4", "--rules-json", "r.json", "--timeline-json", "t.json", "--rotate", "45"])

    def test_missing_korean_font_is_an_error_not_a_silent_fallback(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("Pillow 를 가져올 수 없어 건너뜁니다")
        with self.assertRaises(RV.ViewerError):
            RV.load_font_getter(("no_such_font_file.ttf",))

    def test_no_image_saving_calls(self):
        with open(RV.__file__, encoding="utf-8") as handle:
            source = handle.read()
        for token in ("imwrite", "imsave", "VideoWriter", "ImageGrab", "save_frame"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
