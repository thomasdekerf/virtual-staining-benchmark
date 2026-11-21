"""Scientific gallery failures stop publication instead of substituting images."""
import copy
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
import zlib

from validate_prediction_gallery import DATASETS, validate_gallery


def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


def png():
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 256, 256, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + bytes(256 * 3)) * 256)) + chunk(b'IEND', b''))


class PredictionGalleryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        image = png(); (self.root / 'field.png').write_bytes(image)
        asset = dict(url='field.png', sha256=hashlib.sha256(image).hexdigest(),
                     pixels_sha256=hashlib.sha256(bytes(256 * 256 * 3)).hexdigest(), available=True)
        checkpoint, manifest = 'a' * 64, 'b' * 64
        self.data = dict(schema='matched-prediction-gallery-v1', datasets=[])
        self.index = dict(runs=[])
        for ds in sorted(DATASETS):
            run_id = 'run-' + ds
            self.index['runs'].append(dict(id=run_id, model='example', datasets=[ds], evaluations=[
                dict(dataset=ds, marker='H&E', split='test', checkpoint_sha256=checkpoint, manifest_sha256=manifest)]))
            fields = []
            for i in range(10):
                fields.append(dict(id=str(i), index=i, source_id='original-' + str(i), target_id='chemical-' + str(i),
                    input=copy.deepcopy(asset), target=copy.deepcopy(asset), predictions=[dict(
                        **asset, model='example', run_id=run_id, field_id=str(i),
                        checkpoint_sha256=checkpoint, target_sha256=asset['sha256'], input_sha256=asset['sha256'])]))
            self.data['datasets'].append(dict(id=ds, marker='H&E', split='test', crop_size=256,
                selection=dict(indices=list(range(10)), output_independent=True), manifest_sha256=manifest,
                models=[dict(id='example', run_id=run_id, checkpoint_sha256=checkpoint)], fields=fields))

    def first(self):
        return self.data['datasets'][0]['fields'][0]

    def validate(self):
        return validate_gallery(self.data, self.root, self.index)

    def test_full_fixed_fields_and_matched_assets_pass(self):
        result = self.validate()
        self.assertEqual(result['fields'], 70)
        self.assertEqual(result['predictions'], 70)

    def test_another_field_or_reference_cannot_be_shown(self):
        for key, value in [('field_id', '9'), ('target_sha256', 'c' * 64), ('input_sha256', 'd' * 64)]:
            original = copy.deepcopy(self.first()['predictions'][0])
            self.first()['predictions'][0][key] = value
            with self.assertRaises(ValueError):
                self.validate()
            self.first()['predictions'][0] = original

    def test_even_consistent_model_and_prediction_cannot_switch_checkpoint(self):
        self.data['datasets'][0]['models'][0]['checkpoint_sha256'] = 'c' * 64
        self.first()['predictions'][0]['checkpoint_sha256'] = 'c' * 64
        with self.assertRaisesRegex(ValueError, 'TEST evaluation evidence'):
            self.validate()

    def test_another_marker_cannot_be_used_as_the_selected_checkpoint(self):
        self.data['datasets'][0]['marker'] = 'HER2'
        with self.assertRaisesRegex(ValueError, 'TEST evaluation evidence'):
            self.validate()

    def test_missing_or_other_run_cannot_supply_a_prediction(self):
        self.first()['predictions'][0]['run_id'] = 'another-run'
        with self.assertRaisesRegex(ValueError, 'another training run'):
            self.validate()

    def test_duplicate_model_prediction_stops_publication(self):
        self.first()['predictions'].append(copy.deepcopy(self.first()['predictions'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate model prediction'):
            self.validate()

    def test_changed_image_bytes_stops_publication(self):
        (self.root / 'field.png').write_bytes(png() + b'changed')
        with self.assertRaisesRegex(ValueError, 'image bytes disagree'):
            self.validate()

    def test_traversal_or_external_asset_urls_are_rejected(self):
        for url in ['../field.png', '/field.png', 'https://example.org/field.png', 'assets/%2e%2e/field.png']:
            self.first()['input']['url'] = url
            with self.assertRaises(ValueError):
                self.validate()

    def test_output_dependent_or_changed_field_selection_is_rejected(self):
        self.data['datasets'][0]['selection']['output_independent'] = False
        with self.assertRaisesRegex(ValueError, 'output-independent'):
            self.validate()

    def test_archived_orion_input_is_explicit_and_does_not_invent_channels(self):
        orion = next(d for d in self.data['datasets'] if d['id'] == 'orion')
        orion['fields'][0]['input'] = dict(available=False, reason='Original multiplex archive unavailable.')
        result = self.validate()
        self.assertEqual(result['input_images'], 69)
        orion['fields'][0]['input'].pop('reason')
        with self.assertRaisesRegex(ValueError, 'explicit status and reason'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
