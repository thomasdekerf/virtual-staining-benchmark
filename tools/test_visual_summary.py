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
        self.assertEqual(len(cohort['runs']), cohort['models'])
        index=summary.read_json(HERE.parent/'dist/data/index.json')
        expected=sum(e.get('rankable') and e['dataset']=='finland' and e['split']=='test' and e['protocol']=='Full production evaluation' for run in index['runs'] for e in run['evaluations'])
        self.assertEqual(len(cohort['runs']),expected)
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

    def test_only_completed_source_bound_runs_acquire_scores(self):
        index=summary.read_json(HERE.parent/'dist/data/index.json')
        recent=self.data['activity']['her2match']
        self.assertEqual(len(recent),11)
        for row in recent:
            run=next(r for r in index['runs'] if r['id']=='her2match__'+row['id']+'_seed42')
            scored=any(e.get('rankable') and e['split']=='test' for e in run['evaluations'])
            self.assertEqual(row['metrics_rankable'],scored)
            if not row['complete']:self.assertFalse(scored)
        self.assertEqual(self.data['scale']['run_records'],len(index['runs']))
        self.assertEqual(self.data['scale']['datasets_published'],len(index['datasets']))
        self.assertEqual(self.data['scale']['datasets_registered'],len(self.data['datasets']))
        for task in self.data.get('coverage_balance',{}).get('tasks',[]):
            run=next((r for r in index['runs'] if r['id']=='balance__'+task['id']),None)
            if run:self.assertFalse(any(e.get('rankable') for e in run['evaluations']))

    def test_research_full_validation_never_becomes_a_monitor_curve_point(self):
        for model in ['eqm','nvg','lrdm']:
            run=summary.read_json(HERE.parent/'dist/data/runs'/f'finland__research_{model}_20261017.json')
            scope=run['curve_protocol']
            self.assertEqual(scope['n'],256)
            self.assertEqual(scope['separate_full_validation_n'],5979)
            full=next(e for e in run['evaluations'] if e['split']=='val')
            self.assertEqual(full['n'],5979)
            self.assertFalse(full['rankable'])
            self.assertTrue(all(points[i][0]<points[i+1][0] for points in run['curves'].values() for i in range(len(points)-1)))

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
        self.assertGreaterEqual(self.data['snapshot']['activity'],self.data['snapshot']['benchmark'])
        for private in ['/Users/', '/sofia/', '/user/', 'vsc20775', 'case_id', 'patient_id']:
            self.assertNotIn(private, text)


if __name__ == '__main__':
    unittest.main()
