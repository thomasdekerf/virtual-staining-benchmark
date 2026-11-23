"""Publish reviewed paper-backed methods without changing research evidence.

The registry reviews individual recipes as well as model families. A published
backbone alone does not admit an unpublished composite or ablation. Unreviewed
new runs stay outside the website until explicitly added to the registry.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

POLICY_PATH = Path(__file__).with_name('published-models.json')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def load_policy(path=None):
    return read(path or POLICY_PATH)


def is_published_run(run, policy=None):
    policy = policy if policy is not None else load_policy()
    paper = policy['models'].get(run.get('model'))
    return bool(paper and paper.get('title') and paper.get('url', '').startswith('https://')
                and run.get('id') in policy['eligible_run_ids'])


def filter_activity(activity, index):
    """Activity comes from a separate private inventory; scope it explicitly."""
    result = copy.deepcopy(activity)
    ids = {r['id'] for r in index['runs']}
    for key, rows in result.items():
        if not isinstance(rows, list):
            continue
        if key == 'her2match':
            result[key] = [r for r in rows if 'her2match__' + r['id'] + '_seed42' in ids]
        elif key == 'recoveries':
            bindings = {'skin_asp': 'skin__asp_recovery_20261118',
                        'finland_histdist': 'finland__histdist_recovery_20261118',
                        'finland_pd_unist': 'finland__pd_unist_recovery_20261118'}
            result[key] = [r for r in rows if bindings.get(r.get('id'), r.get('id')) in ids]
    return result


def _asset_urls(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _asset_urls(item)
    elif isinstance(value, list):
        for item in value:
            yield from _asset_urls(item)
    elif isinstance(value, str) and value.startswith(('images/', 'assets/')):
        yield value


def apply_policy(dist, policy=None, archive=None):
    """Curate every public inventory and remove excluded payloads from dist.

    Optional archive is outside the deployed directory. Training files, private
    run inventories, checkpoints and full local result snapshots are untouched.
    """
    dist = Path(dist)
    policy = policy if policy is not None else load_policy()
    index = read(dist / 'data/index.json')
    before = index.get('publication_scope', {})
    all_runs = index['runs']
    runs = [r for r in all_runs if is_published_run(r, policy)]
    ids = {r['id'] for r in runs}
    excluded = {r['id'] for r in all_runs} - ids
    models = []
    for model in index['models']:
        if model['id'] not in policy['models']:
            continue
        model = copy.deepcopy(model)
        model['paper'] = copy.deepcopy(policy['models'][model['id']])
        if model.get('family') == 'Historical experiment':
            model['family'] = 'Published image-to-image translation model'
        if model['id'] == 'bbdm':
            model['family'] = 'Paired Brownian bridge diffusion'
        if not model.get('url') and model['paper'].get('code_url'):
            model['url'] = model['paper']['code_url']
        mine = [r for r in runs if r['model'] == model['id']]
        model.update(runs=len(mine), evaluated_runs=sum(any(e.get('rankable') and e['split'] == 'test' for e in r['evaluations']) for r in mine),
                     datasets=sorted({e['dataset'] for r in mine for e in r['evaluations'] if e.get('rankable')}))
        models.append(model)
    canonical = json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()
    scope = dict(type='published-models', policy_sha256=hashlib.sha256(canonical).hexdigest(),
                 reviewed_date=policy['reviewed_date'], model_count=len(models), run_records=len(runs),
                 rankable_evaluations=sum(bool(e.get('rankable')) for r in runs for e in r['evaluations']),
                 unfiltered_run_records=before.get('unfiltered_run_records', len(all_runs)),
                 note='Only reviewed methods with a paper reference and conventional dataset adaptations are shown. Unpublished local research recipes and ablations are omitted. These are local benchmark results, not scores reported by the papers.')
    scope['removed_run_records'] = scope['unfiltered_run_records'] - len(runs)
    index.update(runs=runs, models=models, publication_scope=scope)
    for dataset in index['datasets']:
        if dataset['id'] == 'orion':
            dataset['description'] = '17-channel multiplex fluorescence translated to paired H&E. Published baselines include regression, GAN, diffusion and transformer models.'

    def remove(path):
        if archive is not None:
            target = Path(archive) / path.relative_to(dist)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
        path.unlink()

    # Purge old direct downloads as well as the visible selectors.
    details = []
    for path in (dist / 'data/runs').glob('*.json'):
        if path.stem not in ids:
            remove(path)
        else:
            details.append(read(path))
    for path in (dist / 'downloads').iterdir():
        if not path.is_file() or path.name == 'results.csv':
            continue
        if not any(path.name.startswith(ident + '_') for ident in ids):
            remove(path)
    csv_path = dist / 'downloads/results.csv'
    if csv_path.exists():
        with csv_path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            csv_rows = [r for r in reader if r['run_id'] in ids]
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, header)
            writer.writeheader()
            writer.writerows(csv_rows)

    gallery_path = dist / 'data/prediction-gallery.json'
    gallery = read(gallery_path)
    for dataset in gallery['datasets']:
        dataset['models'] = [m for m in dataset['models'] if m['id'] in policy['models'] and m.get('run_id') in ids]
        mids = {m['id'] for m in dataset['models']}
        for field in dataset['fields']:
            field['predictions'] = [p for p in field['predictions'] if p['model'] in mids and p.get('run_id') in ids]
    gallery['publication_scope'] = scope
    write(gallery_path, gallery)
    used_assets = set(_asset_urls(details)) | set(_asset_urls(gallery))
    for directory in ['assets', 'images']:
        for path in (dist / directory).rglob('*'):
            if path.is_file() and str(path.relative_to(dist)) not in used_assets:
                remove(path)

    # Keep all original workbook reconciliation totals explicitly historical.
    audit = index['audit']
    audit.setdefault('historical_original_ranked_rows', audit.get('original_ranked_rows_preserved', 0))
    proof_path = dist / 'data/refresh-provenance.json'
    proof = read(proof_path)
    proof.setdefault('historical_original_ranked_rows', proof['original_ranked_rows_preserved'])
    proof['original_score_fingerprints'] = [r for r in proof['original_score_fingerprints'] if r['run'] in ids]
    proof['original_ranked_rows_preserved'] = len(proof['original_score_fingerprints'])
    audit['original_ranked_rows_preserved'] = proof['original_ranked_rows_preserved']
    for key in ['new_result_rows']:
        if key in proof:
            proof[key] = [r for r in proof[key] if r['run'] in ids]
    proof['sources'] = [s for s in proof.get('sources', [])
                       if not any(ident in s.get('file', '') for ident in excluded)
                       and '/research/' not in s.get('file', '')
                       and not re.search(r'\b(?:eqm|nvg|lrdm|itarflow)\b',
                                         s.get('file', '') + ' ' + s.get('label', ''), re.I)]
    proof['publication_scope'] = scope
    write(proof_path, proof)
    audit['unstained_result_rows_verified'] = sum(bool(e.get('rankable')) and r['task'] == 'unstained' for r in runs for e in r['evaluations'])
    audit['current_unstained_result_rows_verified'] = audit['unstained_result_rows_verified']
    audit['publication_scope'] = scope
    audit['notes'] = [n for n in audit.get('notes', []) if 'published-model' not in n] + [
        'The published-model website is a curated subset. Workbook inventory and reconciliation totals describe the complete historical research ledger; visible counts and downloads use only the reviewed subset. Original research results remain preserved locally.']
    reconciliation_path = dist / 'data/reconciliation.json'
    reconciliation = read(reconciliation_path)
    for key in ['dataset_rows', 'unstained_rows', 'orion_rows', 'new_result_rows']:
        reconciliation[key] = [r for r in reconciliation.get(key, []) if r.get('run') in ids]
    reconciliation.update(audit=audit, publication_scope=scope)
    write(reconciliation_path, reconciliation)

    deferred_path = dist / 'data/deferred-training.json'
    deferred = read(deferred_path)
    deferred['runs'] = [r for r in deferred['runs'] if r['id'] in ids]
    deferred.update(run_records=len(deferred['runs']), readiness_groups=dict(Counter(r['readiness'] for r in deferred['runs'])), publication_scope=scope)
    write(deferred_path, deferred)
    gpu = index.get('gpu_activity')
    if gpu:
        gpu['jobs'] = [j for j in gpu['jobs'] if j['model'] in policy['models']]
        running = [j for j in gpu['jobs'] if j['status'] == 'RUNNING']
        gpu['own_training_gpus'] = sum(j['gpus'] for j in running)
        gpu['own_running_training_jobs'] = len(running)
        gpu['limits'].pop('independent_eqm_dependency_removed', None)
        gpu['blocked'].update(run_records=len(deferred['runs']), readiness_groups=deferred['readiness_groups'])
        gpu['blocked']['reason'] = 'Original data or archived parent state required for these published-method runs'
        gpu['publication_scope'] = scope
        write(dist / 'data/gpu-activity.json', gpu)
    index['followup_campaigns'] = [r for r in index.get('followup_campaigns', []) if r['run_id'] in ids]
    training_path = dist / 'data/training-audit.json'
    training = read(training_path)
    training['runs'] = [r for r in training['runs'] if r['id'] in ids]
    training.pop('additional_research_audit', None)
    training.update(run_records=len(training['runs']), counts=dict(Counter(r['classification'] for r in training['runs'])),
                    runs_with_training_evidence=sum(r['classification'] != 'registered_without_training_evidence' for r in training['runs']), publication_scope=scope)
    training['continuations'] = [r for r in training.get('continuations', []) if r['model'] in policy['models']]
    training['blocked'] = dict(total_run_records=len(deferred['runs']), reason=training['blocked']['reason'])
    write(training_path, training)
    index['training_audit'] = {k: v for k, v in training.items() if k != 'runs'} | {'download': 'data/training-audit.json'}
    write(dist / 'data/publication-policy.json', dict(scope=scope, models=[dict(id=m['id'], name=m['name'], paper=m['paper']) for m in models]))
    write(dist / 'data/index.json', index)
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, default=Path(__file__).resolve().parents[1] / 'dist')
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(apply_policy(args.dist, archive=args.archive), indent=2))


if __name__ == '__main__':
    main()
