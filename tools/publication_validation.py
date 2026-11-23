"""Check that every deployed model surface obeys the reviewed publication scope."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def validate_publication_scope(dist, index, policy=None):
    """Reject excluded records, stale downloads and orphan experimental images.

    Selection is checked against the reviewed run list, rather than accepting an
    unpublished recipe merely because its backbone has a paper. Metric identity
    and gallery/checkpoint identity are independently checked by validate_data.
    """
    from publication_policy import is_published_run, load_policy

    dist = Path(dist)
    policy = load_policy() if policy is None else policy
    scope = index.get('publication_scope', {})
    require(scope.get('type') == 'published-models', 'Missing published-model scope')
    digest = scope.get('policy_sha256', '')
    require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
            'Invalid publication-policy fingerprint')
    expected = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(digest == expected, 'Publication-policy fingerprint differs from reviewed registry')

    entries = index['runs']
    ids = {row['id'] for row in entries}
    models = {model['id']: model for model in index['models']}
    require(len(ids) == len(entries), 'Duplicate published run identity')
    require(len(models) == len(index['models']), 'Duplicate published model identity')
    for model_id, model in models.items():
        paper = model.get('paper') or {}
        source = policy['models'].get(model_id)
        require(source is not None, 'Model absent from reviewed publication registry: ' + model_id)
        require(all(paper.get(key) == source.get(key) for key in ('title', 'url', 'status')),
                'Published model paper differs from reviewed source: ' + model_id)
        require(isinstance(paper.get('title'), str) and paper['title'].strip()
                and isinstance(paper.get('url'), str) and paper['url'].startswith('https://'),
                'Published model requires an actual paper citation: ' + model_id)
    for row in entries:
        require(row['model'] in models and is_published_run(row, policy),
                'Unreviewed experimental recipe entered published scope: ' + row['id'])
    require(scope.get('model_count') == len(models) and scope.get('run_records') == len(entries),
            'Published-model counts differ from visible inventory')
    require(scope.get('rankable_evaluations') == sum(bool(e.get('rankable'))
            for row in entries for e in row.get('evaluations', [])), 'Published score count differs')
    public_policy = read_json(dist / 'data/publication-policy.json', {})
    require(public_policy.get('scope') == scope, 'Public publication-policy scope differs from the index')
    public_models = {row['id']: row for row in public_policy.get('models', [])}
    require(set(public_models) == set(models), 'Public publication-policy model inventory differs')
    require(all(public_models[mid].get('paper') == model.get('paper')
                for mid, model in models.items()), 'Public paper citation differs from model inventory')
    files = {p.stem for p in (dist / 'data/runs').glob('*.json')}
    require(files == ids, 'Excluded or missing run JSON remains in deployment')
    details = [read_json(dist / 'data/runs' / (rid + '.json')) for rid in sorted(ids)]
    for detail in details:
        require(is_published_run(detail, policy), 'Excluded recipe remains in a run detail')

    summary = read_json(dist / 'data/visual-summary.json', {})
    for row in summary.get('coverage', {}).get('rows', []):
        require(row['id'] in models, 'Experimental model remains in coverage atlas')
    for row in summary.get('cohort', {}).get('runs', []) + summary.get('trajectories', {}).get('series', []):
        require(row['id'] in ids, 'Excluded run remains in overview plot')
    require(set(summary.get('cohort', {}).get('default_runs', [])) <= ids,
            'Excluded run remains a radar default')
    for owner in (index, summary):
        for row in owner.get('followup_campaigns', []):
            require(row['run_id'] in ids, 'Excluded run remains in follow-up cards')

    gallery = read_json(dist / 'data/prediction-gallery.json', {})
    for dataset in gallery.get('datasets', []):
        inventory = {row['id']: row for row in dataset['models']}
        for model_id, row in inventory.items():
            require(model_id in models, 'Experimental model remains in gallery inventory')
            require(not row.get('run_id') or row['run_id'] in ids,
                    'Excluded recipe remains a gallery checkpoint')
        for field in dataset.get('fields', []):
            for row in field.get('predictions', []):
                require(row['model'] in inventory, 'Excluded prediction remains in gallery')
                require(not row.get('run_id') or row['run_id'] in ids,
                        'Excluded run remains in gallery prediction')

    audit = read_json(dist / 'data/training-audit.json', {})
    for row in audit.get('runs', []):
        require(row['id'] in ids, 'Excluded run remains in training audit')
    for row in audit.get('continuations', []):
        require(row['model'] in models, 'Excluded model remains in continuation plan')
    research = audit.get('additional_research_audit', {})
    require(not research.get('runs'), 'Experimental research audit remains publicly listed')
    gpu = read_json(dist / 'data/gpu-activity.json', {})
    for row in gpu.get('jobs', []):
        require(row['model'] in models, 'Excluded model remains in GPU job table')
    deferred = read_json(dist / 'data/deferred-training.json', {})
    for row in deferred.get('runs', []):
        require(is_published_run(row, policy), 'Excluded recipe remains in deferred plan')
    if deferred:
        require(deferred['run_records'] == len(deferred.get('runs', [])), 'Stale deferred-run count')
        require(sum(deferred.get('readiness_groups', {}).values()) == deferred['run_records'],
                'Stale deferred readiness groups')
    for filename in ('reconciliation.json', 'refresh-provenance.json'):
        record = read_json(dist / 'data' / filename, {})
        for key in ('dataset_rows', 'unstained_rows', 'orion_rows', 'new_result_rows', 'original_score_fingerprints'):
            for row in record.get(key, []):
                require(not row.get('run') or row['run'] in ids,
                        'Excluded run remains in public reconciliation: ' + filename)

    result_path = dist / 'downloads/results.csv'
    if result_path.is_file():
        with result_path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                require(row['run_id'] in ids, 'Excluded run remains in result CSV')
    for path in (dist / 'downloads').glob('*.csv'):
        if path.name == 'results.csv':
            continue
        require(any(path.name.startswith(rid + '_') for rid in ids),
                'Excluded or unbound scalar/per-tile CSV remains in deployment: ' + path.name)

    # A removed prediction must also disappear as a public asset, including when
    # its old filename could otherwise be opened directly without the gallery.
    references = set()
    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.startswith(('assets/', 'images/')):
            references.add(value)
    visit(index); visit(summary); visit(gallery); visit(details)
    assets = {p.relative_to(dist).as_posix() for parent in ('assets', 'images')
              for p in (dist / parent).rglob('*') if p.is_file()
              and p.suffix.lower() in {'.webp', '.png', '.jpg', '.jpeg'}}
    require(assets <= references, 'Unreferenced experimental image remains publicly accessible')
    return dict(models=len(models), runs=len(ids), score_rows=scope['rankable_evaluations'],
                referenced_images=len(assets), validation='passed')
