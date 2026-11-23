"""Prevent experimental recipes leaking back through a published backbone or URL."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from publication_policy import is_published_run
from publication_validation import validate_publication_scope


class PublicationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for directory in ('data/runs', 'downloads', 'assets', 'images'):
            (self.root / directory).mkdir(parents=True)
        self.paper = dict(title='U-Net: Convolutional Networks for Biomedical Image Segmentation',
                          url='https://arxiv.org/abs/1505.04597', status='published')
        self.policy = dict(schema='published-model-registry-v1', reviewed_date='2026-11-23',
                           models={'unet': self.paper}, eligible_run_ids=['unet_reference'])
        policy_digest = hashlib.sha256(json.dumps(self.policy, sort_keys=True,
                                                 separators=(',', ':')).encode()).hexdigest()
        self.run = dict(id='unet_reference', model='unet', evaluations=[dict(
            id='native-test', rankable=True, metrics={'psnr': 23.123456789})],
            gallery=[dict(gt='assets/native.webp', pred='assets/native.webp')])
        self.index = dict(runs=[self.run], models=[dict(id='unet', name='U-Net', paper=self.paper)],
            publication_scope=dict(type='published-models', policy_sha256=policy_digest,
                                   model_count=1, run_records=1, rankable_evaluations=1,
                                   removed_run_records=1))
        self.write('data/runs/unet_reference.json', self.run)
        self.write('data/publication-policy.json',
                   dict(scope=self.index['publication_scope'], models=self.index['models']))
        self.write('data/visual-summary.json', dict(
            coverage=dict(rows=[dict(id='unet')]),
            cohort=dict(runs=[dict(id='unet_reference')], default_runs=['unet_reference']),
            trajectories=dict(series=[dict(id='unet_reference')]),
            followup_campaigns=[dict(run_id='unet_reference')]))
        self.write('data/prediction-gallery.json', dict(datasets=[dict(id='example',
            models=[dict(id='unet', run_id='unet_reference')], fields=[dict(predictions=[dict(
                model='unet', run_id='unet_reference', url='assets/native.webp')])])]))
        self.write('data/training-audit.json', dict(runs=[dict(id='unet_reference')],
                   continuations=[dict(model='unet')]))
        self.write('data/gpu-activity.json', dict(jobs=[dict(model='unet')]))
        self.write('data/deferred-training.json', dict(run_records=1,
                   readiness_groups={'WAITING_DATA': 1}, runs=[dict(id='unet_reference', model='unet')]))
        self.write('data/reconciliation.json', dict(orion_rows=[dict(run='unet_reference')]))
        self.write('data/refresh-provenance.json',
                   dict(original_score_fingerprints=[dict(run='unet_reference')]))
        (self.root / 'downloads/results.csv').write_text('run_id,model,psnr\nunet_reference,U-Net,23.123456789\n')
        (self.root / 'downloads/unet_reference_curves.csv').write_text('step,loss\n1,0.25\n')
        (self.root / 'assets/native.webp').write_bytes(b'original encoded tissue bytes')

    def write(self, relative, value):
        (self.root / relative).write_text(json.dumps(value))

    def read(self, relative):
        return json.loads((self.root / relative).read_text())

    def validate(self):
        return validate_publication_scope(self.root, self.index, self.policy)

    def test_reviewed_published_baseline_and_its_exact_assets_pass(self):
        self.assertTrue(is_published_run(self.run, self.policy))
        self.assertEqual(self.validate()['score_rows'], 1)
        self.assertEqual((self.root / 'assets/native.webp').read_bytes(), b'original encoded tissue bytes')
        self.assertEqual(self.read('data/runs/unet_reference.json')['evaluations'][0]['metrics'],
                         {'psnr': 23.123456789})

    def test_published_backbone_does_not_approve_an_unreviewed_recipe(self):
        custom = dict(self.run, id='unet_custom_marker_gating')
        self.assertFalse(is_published_run(custom, self.policy))
        self.assertFalse(is_published_run(dict(self.run, id='new_unreviewed_run'), self.policy))
        self.assertFalse(is_published_run(dict(self.run, model='unpublished_architecture'), self.policy))

    def test_excluded_run_cannot_survive_as_direct_json_url(self):
        self.write('data/runs/unet_custom_marker_gating.json', dict(id='unet_custom_marker_gating', model='unet'))
        with self.assertRaisesRegex(ValueError, 'run JSON'):
            self.validate()

    def test_paper_citation_cannot_be_replaced_with_an_unverified_reference(self):
        self.index['models'][0]['paper'] = dict(self.paper, url='https://example.org/unsupported')
        with self.assertRaisesRegex(ValueError, 'reviewed source'):
            self.validate()

    def test_scope_must_bind_the_actual_reviewed_recipe_and_paper_registry(self):
        self.index['publication_scope']['policy_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
            self.validate()

    def test_excluded_recipe_is_rejected_even_when_index_and_file_are_consistent(self):
        self.index['runs'][0]['id'] = 'unet_custom_marker_gating'
        (self.root / 'data/runs/unet_reference.json').unlink()
        self.write('data/runs/unet_custom_marker_gating.json', self.index['runs'][0])
        with self.assertRaisesRegex(ValueError, 'Unreviewed experimental recipe'):
            self.validate()

    def test_gallery_cannot_retain_an_excluded_checkpoint_under_the_same_family(self):
        gallery = self.read('data/prediction-gallery.json')
        gallery['datasets'][0]['models'][0]['run_id'] = 'unet_custom_marker_gating'
        self.write('data/prediction-gallery.json', gallery)
        with self.assertRaisesRegex(ValueError, 'gallery checkpoint'):
            self.validate()

    def test_plot_and_gpu_inventories_are_checked_independently_of_index(self):
        summary = self.read('data/visual-summary.json')
        summary['cohort']['runs'].append(dict(id='excluded-run'))
        self.write('data/visual-summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'overview plot'):
            self.validate()
        summary['cohort']['runs'].pop()
        self.write('data/visual-summary.json', summary)
        self.write('data/gpu-activity.json', dict(jobs=[dict(model='unpublished_architecture')]))
        with self.assertRaisesRegex(ValueError, 'GPU job table'):
            self.validate()

    def test_experimental_results_and_curve_downloads_are_not_published(self):
        path = self.root / 'downloads/results.csv'
        original = path.read_text()
        path.write_text(original + 'excluded-run,Custom,99\n')
        with self.assertRaisesRegex(ValueError, 'result CSV'):
            self.validate()
        path.write_text(original)
        (self.root / 'downloads/excluded-run_curves.csv').write_text('step,loss\n1,0\n')
        with self.assertRaisesRegex(ValueError, 'scalar/per-tile CSV'):
            self.validate()

    def test_removed_prediction_bytes_cannot_remain_as_an_orphan_public_asset(self):
        (self.root / 'images/excluded-prediction.webp').write_bytes(b'experimental prediction')
        with self.assertRaisesRegex(ValueError, 'experimental image'):
            self.validate()

    def test_public_registry_and_visible_catalog_must_agree(self):
        public = self.read('data/publication-policy.json')
        public['models'].append(dict(id='excluded_architecture', name='Custom', paper=self.paper))
        self.write('data/publication-policy.json', public)
        with self.assertRaisesRegex(ValueError, 'model inventory differs'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
