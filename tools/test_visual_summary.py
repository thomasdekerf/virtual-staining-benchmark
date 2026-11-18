"""Protect chart semantics: comparable samples, fixed scales and no fake scores."""
import importlib.util
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('visual_summary', HERE / 'build_visual_summary.py')
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class VisualSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Published snapshot checks also run in the public repository, where
        # private workspace receipts intentionally are not distributed.
        cls.data = summary.read_json(HERE.parent / 'dist/data/visual-summary.json')
        cls.workspace_receipts_available = (HERE.parents[1] / 'artifacts/her2match/training_20261118/latest_status.json').exists()

    def test_mixed_metric_directions_and_missing_values_are_explicit(self):
        self.assertEqual(summary.normalize(10, 0, 20, True), .5)
        self.assertEqual(summary.normalize(0, 0, 20, False), 1)
        self.assertEqual(summary.normalize(20, 0, 20, False), 0)
        self.assertIsNone(summary.normalize(None, 0, 20, False))
        self.assertEqual(summary.normalize(10, 10, 10, True), .5)

    def test_radar_has_identical_complete_test_coverage_and_fixed_cohort_scales(self):
        cohort = self.data['cohort']
        self.assertEqual(cohort['n'], 5179)
        self.assertEqual(len(cohort['runs']), 16)
        self.assertEqual(len(cohort['default_runs']), 3)
        self.assertEqual({m['direction'] for m in cohort['metrics']}, {'higher', 'lower'})
        for metric in cohort['metrics']:
            key = metric['key']
            self.assertEqual({r['coverage'][key] for r in cohort['runs']}, {5179})
            normalized = [r['normalized'][key] for r in cohort['runs']]
            self.assertEqual(min(normalized), 0)
            self.assertEqual(max(normalized), 1)
            # Selection changes cannot change the normalization range.
            for run in cohort['runs']:
                expected = summary.normalize(run['values'][key], metric['min'], metric['max'], metric['direction'] == 'higher')
                self.assertEqual(run['normalized'][key], expected)

    def test_new_training_is_unscored_and_does_not_inflate_historical_results(self):
        recent = self.data['activity']['her2match']
        self.assertEqual(len(recent), 11)
        self.assertTrue(all(not row['metrics_rankable'] for row in recent))
        self.assertTrue(all('values' not in row and 'metrics' not in row for row in recent))
        self.assertEqual(self.data['scale']['run_records'], 304)
        self.assertEqual(self.data['scale']['datasets_published'], 6)
        self.assertEqual(self.data['scale']['datasets_registered'], 7)
        self.assertEqual(self.data['datasets'][-1]['rankable_evaluations'], 0)
        queued = [row for row in recent if not row['production_started']]
        self.assertTrue(queued)
        self.assertTrue(all(row['step'] == 0 for row in queued))

    def test_bci_trajectories_have_actual_shared_update_axis_and_selected_markers(self):
        chart = self.data['trajectories']
        self.assertEqual((chart['dataset'], chart['marker'], chart['split'], chart['n']), ('BCI', 'HER2', 'val', 585))
        for run in chart['series']:
            self.assertEqual(run['axis'], 'optimizer updates')
            self.assertEqual(run['recorded_points'], len(run['points']))
            self.assertTrue(all(point[0] % 103 == 0 for point in run['points']))
            self.assertIn(run['selected_step'], [p[0] for p in run['points']])

    def test_public_data_is_deterministic_source_bound_and_path_safe(self):
        text = json.dumps(self.data, allow_nan=False)
        if self.workspace_receipts_available:
            self.assertEqual(self.data, summary.build(HERE.parent / 'dist', HERE.parents[1]))
        self.assertTrue(all(len(p['sha256']) == 64 for p in self.data['provenance']))
        self.assertNotEqual(self.data['snapshot']['benchmark'], self.data['snapshot']['activity'])
        for private in ['/Users/', '/sofia/', '/user/', 'vsc20775', 'case_id', 'patient_id']:
            self.assertNotIn(private, text)


if __name__ == '__main__':
    unittest.main()
