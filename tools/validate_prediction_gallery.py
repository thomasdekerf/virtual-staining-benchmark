"""Validate matched-field gallery identities and exported assets using stdlib only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
from urllib.parse import urlsplit

DIGEST = re.compile(r'[0-9a-f]{64}\Z')
DATASETS = {'orion', 'BCI', 'MIST', 'IHC4BC', 'finland', 'skin', 'her2match'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, context):
    require(isinstance(value, str) and DIGEST.fullmatch(value), context + ': invalid SHA-256')
    return value


def image_shape(raw, context):
    if raw[:8] == b'\x89PNG\r\n\x1a\n' and raw[12:16] == b'IHDR' and len(raw) >= 33:
        return struct.unpack('>II', raw[16:24]), 'PNG'
    if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
        offset, shape, lossless = 12, None, False
        while offset + 8 <= len(raw):
            kind = raw[offset:offset+4]
            length = int.from_bytes(raw[offset+4:offset+8], 'little')
            chunk = raw[offset+8:offset+8+length]
            require(len(chunk) == length, context + ': truncated WebP chunk')
            if kind == b'VP8L':
                require(len(chunk) >= 5 and chunk[0] == 0x2f, context + ': invalid lossless WebP')
                bits = int.from_bytes(chunk[1:5], 'little')
                shape, lossless = ((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1), True
            elif kind == b'VP8X' and len(chunk) >= 10:
                shape = (int.from_bytes(chunk[4:7], 'little') + 1,
                         int.from_bytes(chunk[7:10], 'little') + 1)
            elif kind == b'VP8 ' and len(chunk) >= 10 and chunk[3:6] == b'\x9d\x01\x2a':
                shape = (int.from_bytes(chunk[6:8], 'little') & 0x3fff,
                         int.from_bytes(chunk[8:10], 'little') & 0x3fff)
            offset += 8 + length + length % 2
        require(shape is not None, context + ': WebP dimensions missing')
        return shape, 'lossless WebP' if lossless else 'lossy WebP'
    if raw[:2] == b'\xff\xd8':
        offset = 2
        while offset + 3 < len(raw):
            if raw[offset] != 0xff:
                break
            while offset < len(raw) and raw[offset] == 0xff:
                offset += 1
            marker = raw[offset]; offset += 1
            if marker in {0x01, *range(0xd0, 0xda)}:
                continue
            if offset + 2 > len(raw):
                break
            length = int.from_bytes(raw[offset:offset+2], 'big')
            if marker in {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}:
                require(length >= 8 and offset + 7 <= len(raw), context + ': truncated JPEG frame')
                return (int.from_bytes(raw[offset+5:offset+7], 'big'),
                        int.from_bytes(raw[offset+3:offset+5], 'big')), 'JPEG'
            require(length >= 2, context + ': invalid JPEG segment')
            offset += length
    raise ValueError(context + ': unsupported or incomplete image encoding')


def asset(record, root, context):
    """Check a local image without following a URL or accepting an external path."""
    require(isinstance(record, dict), context + ': asset record required')
    url = record.get('url')
    if not url:
        require(record.get('available') is False and bool(str(record.get('reason', '')).strip()),
                context + ': unavailable asset needs an explicit status and reason')
        return False
    require(record.get('available') is not False, context + ': unavailable asset has a URL')
    require(isinstance(url, str) and not any(c in url for c in ('\\', '%', '\x00')),
            context + ': invalid asset URL')
    parsed = urlsplit(url)
    require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment,
            context + ': only local relative asset URLs are allowed')
    require(not url.startswith('/') and all(p not in ('', '.', '..') for p in url.split('/')),
            context + ': asset path traversal or absolute path')
    relative = PurePosixPath(url)
    path = (root / Path(*relative.parts)).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), context + ': missing local asset')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest(record.get('sha256'), context),
            context + ': image bytes disagree with SHA-256')
    shape, encoding = image_shape(raw, context)
    require(shape == (256, 256), context + ': expected the same 256-pixel crop')
    if encoding in {'JPEG', 'lossy WebP'}:
        require(bool(record.get('compression') or record.get('encoding')),
                context + ': historical compressed image needs its compression label')
    return True


def validate_gallery(data, root, index):
    root = Path(root)
    registered = {r['id']: r for r in index['runs']}
    require(data.get('schema') == 'matched-prediction-gallery-v1', 'Unknown gallery schema')
    datasets = data.get('datasets')
    require(isinstance(datasets, list) and {d.get('id') for d in datasets} == DATASETS
            and len(datasets) == len(DATASETS), 'Gallery must register all seven datasets exactly once')
    counts = dict(datasets=len(datasets), fields=0, predictions=0, unavailable=0, input_images=0)
    for dataset in datasets:
        context = 'gallery/' + dataset['id']
        require(dataset.get('split') == 'test', context + ': gallery fields must be TEST')
        require(dataset.get('crop_size') == 256, context + ': gallery must use native256 crops')
        require(dataset.get('selection', {}).get('indices') == list(range(10))
                and dataset.get('selection', {}).get('output_independent') is True,
                context + ': fixed output-independent selection receipt required')
        models, fields = dataset.get('models'), dataset.get('fields')
        require(isinstance(models, list) and models, context + ': model inventory required')
        require(isinstance(fields, list) and len(fields) == 10, context + ': exactly ten fixed fields required')
        require(all(type(f.get('index')) is int for f in fields) and [f.get('index') for f in fields] == list(range(10)), context + ': fields must be fixed TEST indices0..9')
        require(len({f.get('id') for f in fields}) == 10 and all(isinstance(f.get('id'), str) and f['id'].strip() for f in fields),
                context + ': duplicate or missing field identity')
        model_map = {m.get('id'): m for m in models}
        require(len(model_map) == len(models) and all(isinstance(k, str) and k for k in model_map),
                context + ': duplicate or missing model identity')
        manifest = dataset.get('manifest_sha256')
        if manifest:
            digest(manifest, context + '/manifest')
        else:
            require(bool(dataset.get('manifest_unavailable_reason')),
                    context + ': manifest fingerprint or explicit unavailable reason required')
        for model in models:
            name = context + '/model/' + model['id']
            run_id = model.get('run_id')
            if run_id:
                require(run_id in registered, name + ': run is absent from the benchmark snapshot')
                run = registered[run_id]
                require(dataset['id'] in run.get('datasets', []), name + ': run belongs to another dataset')
                require(model['id'] == run.get('model'), name + ': run belongs to another model family')
            if model.get('checkpoint_sha256'):
                digest(model['checkpoint_sha256'], name + '/checkpoint')
        for field in fields:
            label = context + '/field/' + str(field['id'])
            counts['fields'] += 1
            if field.get('manifest_sha256'):
                require(field['manifest_sha256'] == manifest, label + ': field manifest differs')
            input_available = asset(field.get('input'), root, label + '/input')
            counts['input_images'] += int(input_available)
            target_available = asset(field.get('target'), root, label + '/target')
            for record, kind, available in [(field.get('input'), 'input', input_available),
                                             (field.get('target'), 'target', target_available)]:
                if available:
                    digest(record.get('pixels_sha256'), label + '/' + kind + '/pixels')
                    require(bool(field.get('source_id' if kind == 'input' else 'target_id')),
                            label + ': original ' + kind + ' identity required')
            for channel in field.get('input', {}).get('channels', []):
                asset(channel, root, label + '/input-channel/' + str(channel.get('id')))
            predictions = field.get('predictions')
            require(isinstance(predictions, list), label + ': prediction inventory required')
            require(len({p.get('model') for p in predictions}) == len(predictions),
                    label + ': duplicate model prediction')
            for prediction in predictions:
                model_id = prediction.get('model')
                require(model_id in model_map, label + ': prediction model is absent from the inventory')
                model = model_map[model_id]
                name = label + '/prediction/' + model_id
                available = asset(prediction, root, name)
                if not available:
                    counts['unavailable'] += 1
                    continue
                require(target_available, name + ': available prediction lacks an exported verified chemical reference')
                require(prediction.get('field_id') == field['id'], name + ': prediction belongs to another field')
                require(model.get('run_id') and prediction.get('run_id') == model['run_id'],
                        name + ': prediction belongs to another training run')
                require(prediction.get('checkpoint_sha256') == digest(model.get('checkpoint_sha256'), name),
                        name + ': prediction checkpoint differs from selected model')
                digest(prediction.get('pixels_sha256'), name + '/pixels')
                if prediction.get('source_panel_sha256'):
                    digest(prediction['source_panel_sha256'], name + '/source-panel')
                if input_available:
                    require(prediction.get('input_sha256') == field['input']['sha256'],
                            name + ': prediction input differs from this field')
                require(prediction.get('target_sha256') == field['target']['sha256'],
                        name + ': prediction reference differs from this field')
                run = registered[model['run_id']]
                candidates = [e for e in run.get('evaluations', [])
                              if e.get('dataset') == dataset['id'] and e.get('split') == 'test'
                              and e.get('marker') == dataset.get('marker')
                              and e.get('checkpoint_sha256') == model['checkpoint_sha256']]
                require(candidates, name + ': selected checkpoint lacks matching TEST evaluation evidence')
                if manifest:
                    require(any(e.get('manifest_sha256') == manifest for e in candidates),
                            name + ': TEST manifest differs from scored checkpoint')
                counts['predictions'] += 1
            # A missing model is an explicit unavailable card, never an image from
            # another tissue field. The UI fills missing rows from the inventory.
            counts['unavailable'] += len(models) - len(predictions)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'dist')
    args = parser.parse_args()
    data = json.loads((args.root / 'data/prediction-gallery.json').read_text())
    index = json.loads((args.root / 'data/index.json').read_text())
    print(json.dumps(dict(validate_gallery(data, args.root, index), validation='passed')))


if __name__ == '__main__':
    main()
