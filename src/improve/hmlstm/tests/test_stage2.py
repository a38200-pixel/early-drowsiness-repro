"""stage2 시험: 합성 배열을 OS 임시 폴더(저장소 밖)에 잠깐 만들어 쓴다. 저장소의 stage2_stats.json 은 읽기만 한다."""
import json
import os
import re
import tempfile
import unittest

import numpy as np

from improve.hmlstm import stage2 as S2


def make_arrays(data_dir, seed=0):
    """폴드 1~5 의 합성 학습 배열(모양 (N, 30, 4), 0 패딩 창 포함)을 data_dir 에 저장하고 {폴드: 배열} 을 돌려준다."""
    rng = np.random.RandomState(seed)
    arrays = {}
    for fold in S2.FOLDS:
        array = rng.randn(5 + fold, 30, 4) * (fold + 1.0) + fold
        array[0, :10, :] = 0.0                                                    # 0 패딩 창
        np.save(os.path.join(data_dir, S2.ARRAY_NAME % fold), array)
        arrays[fold] = array
    return arrays


class RoundTripTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = self.temp.name
        self.arrays = make_arrays(self.dir)

    def test_statistics_use_the_whole_array_including_padding_with_ddof_zero(self):
        document, checks = S2.build_document(self.dir)
        self.assertEqual(checks, [])
        for fold in S2.FOLDS:
            entry = document["folds"][str(fold)]
            array = self.arrays[fold]
            for k in range(4):
                self.assertEqual(entry["mean"][k], float(np.mean(array[:, :, k])))
                self.assertEqual(entry["std"][k], float(np.std(array[:, :, k])))
            self.assertEqual((entry["n"], entry["shape"], entry["dtype"]), (array.shape[0], list(array.shape), "float64"))
            self.assertEqual(len(entry["sha256_12"]), 12)

    def test_json_round_trip_restores_floats_exactly(self):
        document, _ = S2.build_document(self.dir)
        path = os.path.join(self.dir, "stats.json")
        S2.write_document(document, path)
        with self.assertRaises(FileExistsError):
            S2.write_document(document, path)                                     # 이미 있는 파일은 덮어쓰지 않는다
        loaded = S2.load_document(path)
        self.assertEqual(loaded, document)
        stats = S2.load_stats(path)
        for fold in S2.FOLDS:
            entry = document["folds"][str(fold)]
            self.assertEqual(list(stats[fold][0]), entry["mean"])                 # 정확히 같은 float
            self.assertEqual(list(stats[fold][1]), entry["std"])
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertNotIn(self.dir, text)                                          # 절대경로는 쓰지 않는다
        self.assertEqual(sorted(json.loads(text)["folds"]), ["1", "2", "3", "4", "5"])

    def test_verify_matches_then_detects_a_changed_array(self):
        document, _ = S2.build_document(self.dir)
        rows = S2.verify_document(document, self.dir)
        self.assertTrue(all(r["exact"] and r["hash_match"] and r["shape_match"] and r["n_match"] for r in rows))
        self.assertEqual(max(r["max_abs_diff"] for r in rows), 0.0)
        changed = self.arrays[2].copy()
        changed[3, 5, 1] += 0.5
        np.save(os.path.join(self.dir, S2.ARRAY_NAME % 2), changed)
        rows = S2.verify_document(document, self.dir)
        by_fold = dict((r["fold"], r) for r in rows)
        self.assertFalse(by_fold[2]["exact"])
        self.assertFalse(by_fold[2]["hash_match"])
        self.assertTrue(by_fold[1]["exact"])

    def test_run_config_hash_check(self):
        root = os.path.join(self.dir, "ckpt")
        for fold in S2.FOLDS:
            folder = os.path.join(root, "b1_seed1", "seed1", "fold%d" % fold)
            os.makedirs(folder)
            if fold == 5:
                continue                                                          # 폴드 5 는 run_config.json 없음
            recorded = S2.file_sha256(os.path.join(self.dir, S2.ARRAY_NAME % fold))
            if fold == 3:
                recorded = "0" * 64                                               # 일부러 틀린 해시
            with open(os.path.join(folder, "run_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"input_sha256": {S2.ARRAY_NAME % fold: recorded}}, handle)
        _document, checks = S2.build_document(self.dir, ckpt_root=root)
        self.assertEqual(dict(checks), {1: "match", 2: "match", 3: "mismatch", 4: "match", 5: "no_run_config"})

    def test_load_document_rejects_a_bad_format(self):
        document, _ = S2.build_document(self.dir)
        path = os.path.join(self.dir, "bad.json")
        del document["folds"]["4"]["std"]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        with self.assertRaises(ValueError):
            S2.load_document(path)

    def test_standardize_uses_each_feature_and_leaves_the_input_alone(self):
        windows = np.arange(2 * 30 * 4, dtype=np.float64).reshape(2, 30, 4)
        before = windows.copy()
        out = S2.standardize(windows, [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 8.0, 16.0])
        np.testing.assert_array_equal(windows, before)
        np.testing.assert_allclose(out[:, :, 2], (before[:, :, 2] - 3.0) / 8.0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(out[:, :, 0], (before[:, :, 0] - 1.0) / 2.0, rtol=0, atol=1e-15)


class PackagedStatsTest(unittest.TestCase):
    def test_packaged_json_is_valid_and_has_no_long_hashes_or_paths(self):
        document = S2.load_document()
        stats = S2.load_stats()
        self.assertEqual(sorted(stats), [1, 2, 3, 4, 5])
        for fold in S2.FOLDS:
            means, stds = stats[fold]
            self.assertEqual((means.shape, stds.shape), ((4,), (4,)))
            self.assertTrue(np.all(np.isfinite(means)) and np.all(stds > 0))
            self.assertEqual(document["folds"][str(fold)]["shape"][1:], [30, 4])
        with open(S2.DEFAULT_PATH, encoding="utf-8") as handle:
            text = handle.read()
        self.assertIsNone(re.search(r"[0-9a-fA-F]{40}", text))
        self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", text))


if __name__ == "__main__":
    unittest.main()
