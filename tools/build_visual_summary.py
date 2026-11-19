"""Export small, source-bound chart examples without changing historical scores.

The benchmark snapshot and later activity receipts have separate clocks. Counts
refer to the historical exported inventory; new jobs are shown as activity and
never fabricated into scored evaluations. All public fields are allowlisted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_RUNS = ['finland__unet_seed42', 'finland__nafnet_seed42', 'finland__restormer_seed42']
RADAR_METRICS = ['psnr', 'ssim', 'lpips', 'dists', 'conch_cosine']
TRAJECTORY_RUNS = ['unet_BCI_HER2', 'pix2pix_BCI_HER2', 'restormer_BCI_HER2']
HER2_MODELS = ['bbdm', 'unet', 'pix2pix', 'pyramid', 'asp', 'pspstain', 'cut', 'cyclegan', 'nafnet', 'restormer', 'cssp2p']


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def normalize(value, low, high, higher):
    """Fixed-cohort min/max; null stays unmeasured, equal values are neutral."""
    if value is None:
        return None
    if low == high:
        return .5
    score = (value - low) / (high - low)
    return score if higher else 1 - score


def log_step(text):
    steps = [int(v) for v in re.findall(r'\bstep[=:]\s*(\d+)', text or '')]
    steps += [int(v) for v in re.findall(r'"step"\s*:\s*(\d+)', text or '')]
    return max(steps, default=0)


def source_record(path, label, as_of=None, href=None):
    row = {'label': label, 'sha256': file_sha256(path)}
    if as_of:
        row['as_of'] = as_of
    if href:
        row['href'] = href
    return row


def activity(workspace, names):
    her2_path = workspace / 'artifacts/her2match/training_20261118/latest_status.json'
    campaign_path = workspace / 'artifacts/her2match/training_20261118/campaign.json'
    recovery_path = workspace / 'artifacts/benchmark_recovery_20261118_v1/latest_status.json'
    flow_path = workspace / 'artifacts/itar768_recovery_20261118_v1/continuation_submission.json'
    flow_live_path = workspace / 'artifacts/itar768_recovery_20261118_v1/continuation_live/status.json'
    h, campaign, b = map(read_json, [her2_path, campaign_path, recovery_path])
    flow, flow_live = map(read_json, [flow_path, flow_live_path])
    activity_stamp = max(h['captured_utc'], b['observed_utc'], flow.get('observed_utc', flow['submitted_utc']))
    source_list = [source_record(her2_path, 'HER2Match production status and finite training logs', h['captured_utc']),
                   source_record(campaign_path, 'HER2Match submitted model inventory'),
                   source_record(recovery_path, 'Benchmark recovery status and evaluation stages', b['observed_utc']),
                   source_record(flow_path, 'Guarded iTARFlow continuation submission', flow.get('observed_utc')),
                   source_record(flow_live_path, 'Guarded iTARFlow committed optimizer progress', flow.get('observed_utc'))]
    queue = {line.split('|')[0]: line.split('|')[1] for line in h.get('queue', '').splitlines() if '|' in line}
    rows = []
    for i, model in enumerate(HER2_MODELS):
        task = h['tasks'].get(model, {})
        status = task.get('status.json') or {}
        phase = (task.get('launch_state.json') or {}).get('phase', '')
        production = phase == 'production_training' or bool(status)
        step = max(status.get('step', 0), log_step(task.get('log_tail', '')) if production else 0)
        state = 'complete' if status.get('complete') else ('checkpointed / queued' if status.get('budget_pause') else ('training' if production else 'queued'))
        if queue.get(f'{h["array_job_id"]}_{i}') == 'RUNNING' and production:
            state = 'training'
        smoke = task.get('smoke_smoke_receipt.json') or {}
        rows.append(dict(id=model, name=names.get(model, 'BBDM' if model == 'bbdm' else model), dataset='her2match',
            job_id=f'{h["array_job_id"]}_{i}', state=state, step=step,
            max_steps=status.get('max_steps'), complete=bool(status.get('complete')),
            production_started=production, smoke_passed=bool(smoke.get('passed')),
            metrics_rankable=False, observed_utc=h['captured_utc'],
            note='Production progress; full selected-checkpoint test evaluation pending.' if production else 'Accepted task; no production optimizer updates recorded.'))
    asp = b['runs']['skin_asp']
    recovery_specs = [('skin_asp', 'Skin ASP', 'skin', '1640924_0', asp['step'], True,
                       'Adaptive-loss division by zero reproduced and corrected; training complete. Core image metrics complete; learned-nucleus evaluation pending.'),
                      ('finland_histdist', 'HistDiST', 'finland', '1640939_1',
                       log_step(b['logs']['recovery-1640939_1.out']['latest_progress']), False,
                       'FP32 continuation with original optimizer and selection state. Original first nonfinite operation unconfirmed.'),
                      ('finland_pd_unist', 'PD-UNIST', 'finland', '1640940_2',
                       log_step(b['logs']['recovery-1640940_2.out']['latest_progress']), False,
                       'FP32 continuation; original nonfinite-gradient cause remains unconfirmed.')]
    recoveries = [dict(id=ident, name=name, dataset=dataset, job_id=job, state='training complete / evaluation' if complete else 'training',
                       step=step, max_steps=None, complete=complete, metric_status='Full benchmark pending',
                       observed_utc=b['observed_utc'], note=note)
                  for ident, name, dataset, job, step, complete, note in recovery_specs]
    recoveries.append(dict(id='itarflow768', name='iTARFlow 768', dataset='finland', job_id=flow['job_id'],
        state=flow_live['state'], step=flow_live['step'], max_steps=48000, complete=False,
        metric_status='Full evaluation pending', observed_utc=flow.get('observed_utc'),
        note='Original failure did not recur on the unchanged 128-image monitor. Guarded continuation preserves original 16k best and stopping state.'))
    papis_path = workspace / 'artifacts/histology_evaluators_20261118/papis_real_he_demo/evaluation.json'
    haps_path = workspace / 'artifacts/histology_evaluators_20261118/haps_uncalibrated_real_he_demo.json'
    papis, haps = read_json(papis_path), read_json(haps_path)
    source_list += [source_record(papis_path, 'Executed PaPIS pretrained-encoder research profile'),
                    source_record(haps_path, 'Executed HAPS uncalibrated RetCCL features')]
    assert papis['canonical'] is False and haps['calibrated_haps_score'] is False
    evaluators = [dict(id='papis', name='PaPIS', state='Research profile operational', trained_head=None,
        note='Author pretrained DeepCMorph features executed on real images. Exact author metric settings unavailable; component scores retained.',
        source_url='https://arxiv.org/html/2507.12624v1'),
        dict(id='haps', name='HAPS', state='Encoder operational; calibration labels needed', trained_head=False,
        note='Pretrained RetCCL runs; four-weight/bias head trainer implemented. Genuine expert similarity ratings needed; no surrogate ratings fitted.',
        source_url='https://arxiv.org/html/2605.20362v1')]
    # Later snapshots are separate artifacts: preserve original diagnostic and
    # submission receipts, and update activity without altering scored cohorts.
    overlay_path = workspace / 'artifacts/benchmark_site_visuals_20261118/activity_refresh_2300.json'
    if overlay_path.exists():
        overlay = read_json(overlay_path)
        if overlay['schema'] != 'benchmark-activity-refresh-v1' or overlay['observed_utc'] < activity_stamp:
            raise ValueError('Activity overlay must be a newer explicitly captured snapshot')
        row_keys = {'id', 'name', 'dataset', 'job_id', 'state', 'step', 'max_steps', 'complete',
                    'production_started', 'smoke_passed', 'metrics_rankable', 'observed_utc',
                    'note', 'metric_status', 'test_metrics', 'test_tiles'}
        def amended(original, additions):
            by_id = {r['id']: r for r in additions}
            if set(by_id) != {r['id'] for r in original}:
                raise ValueError('Activity refresh changed the submitted task inventory')
            return [dict(row, **{k: v for k, v in by_id[row['id']].items() if k in row_keys}) for row in original]
        rows = amended(rows, overlay['her2match'])
        recoveries = amended(recoveries, overlay['recoveries'])
        activity_stamp = overlay['observed_utc']
        source_list.append(source_record(overlay_path, 'Later read-only activity refresh', activity_stamp))
        for source in overlay['source_hashes']:
            source_list.append({'label': source['label'], 'sha256': source['sha256'], 'as_of': activity_stamp})
    current_path=workspace/'artifacts/coverage_balance_20261119/live_audit/cluster_snapshot.json'
    if current_path.exists():
        live=read_json(current_path);activity_stamp=live['captured_utc']
        source_list.append(source_record(current_path,'Current read-only production and recovery audit',activity_stamp))
        for row in rows:
            files=live['her2match'][row['id']]['files'];saved=files.get('status.json',{}).get('data',{});final=files.get('final_evaluation.json',{}).get('data')
            row.update(step=saved.get('step',0),max_steps=saved.get('max_steps'),complete=bool(saved.get('complete')),production_started=bool(saved.get('step')),observed_utc=activity_stamp,metrics_rankable=bool(final),state='evaluated' if final else 'training complete / evaluation' if saved.get('complete') else 'checkpointed / queued' if saved.get('budget_pause') else 'training')
            launch=files.get('launch_state.json',{}).get('data',{})
            row['job_id']=launch.get('job_id',row['job_id'])
            row['note']='Full selected-checkpoint validation and test results are published; clinical HER2 grading was not evaluated.' if final else 'Production continues; final selected-checkpoint test results remain pending.'
        for row in recoveries:
            if row['id']=='itarflow768':
                saved=live['itarflow']['continuation_v1']['files']['status.json']['data'];row.update(step=saved['step'],complete=saved.get('complete',False),state='training complete / held-out evaluation',metric_status='Study validation evaluation running',note='Training stopped at 33k; selected checkpoint 25k. The original 128-image monitor remains separate from the held-out evaluation.',observed_utc=activity_stamp)
            else:
                files=live['recovery'][row['id']]['files'];saved=files['status.json']['data'];complete=bool(files.get('evaluation_v1/complete.json'))
                row.update(step=saved['step'],complete=saved.get('complete',False),state='complete' if complete else 'training complete / evaluation' if saved.get('complete') else 'training',metric_status='Full benchmark complete' if complete else 'Full benchmark pending',observed_utc=activity_stamp)
                if complete:row['note']='Numerical recovery and full image/foundation-model/four-detector benchmark evaluation completed; parent evidence retained.'
    return dict(observed_utc=activity_stamp, her2match=rows, recoveries=recoveries, evaluators=evaluators), source_list


def build(dist, workspace):
    index_path = dist / 'data/index.json'
    data = read_json(index_path)
    entries = data['runs']
    names = {row['id']: row['name'] for row in data['models']}
    all_evaluations = [evaluation for run in entries for evaluation in run['evaluations']]
    evaluated = [run for run in entries if any(e.get('rankable') for e in run['evaluations'])]
    tested = [run for run in entries if any(e.get('rankable') and e['split'] == 'test' for e in run['evaluations'])]
    recorded_compute = [run for run in entries if finite(run.get('gpu_hours'))]
    scalar_series = sum(len(run.get('curve_counts', {})) for run in entries)
    scalar_points = sum(sum(run.get('curve_counts', {}).values()) for run in entries)
    scale = dict(run_records=len(entries), model_families_with_runs=len({r['model'] for r in entries}),
        catalog_models=len(data['models']), evaluation_records=len(all_evaluations),
        rankable_evaluations=sum(bool(e.get('rankable')) for e in all_evaluations),
        evaluated_runs=len(evaluated), tested_runs=len(tested),
        evaluated_model_families=len({r['model'] for r in evaluated}),
        tested_model_families=len({r['model'] for r in tested}),
        scalar_series=scalar_series, scalar_points=scalar_points,
        paired_fields=sum(r.get('image_count', 0) for r in entries),
        recorded_gpu_hours=math.fsum(r['gpu_hours'] for r in recorded_compute),
        compute_runs=len(recorded_compute), datasets_published=len(data['datasets']),
        datasets_registered=len(data['datasets']) if any(d['id']=='her2match' for d in data['datasets']) else len(data['datasets'])+1, her2match_tasks=len(HER2_MODELS),
        original_breast_runs=data['audit']['breast_runs'], workbook_verified_cells=data['audit']['verified_cells'])
    recent, receipt_sources = activity(workspace, names)
    sources = [source_record(index_path, 'Current reconciled benchmark inventory', data['snapshot'], 'data/index.json')]
    sources += receipt_sources
    def trained(run):
        return bool(any(e.get('rankable') for e in run['evaluations']) or sum(run.get('curve_counts',{}).values()) or (run.get('last_step') or 0)>0)
    datasets=[]
    for dataset in data['datasets']:
        mine=[r for r in entries if dataset['id'] in r['datasets']]
        scored=[r for r in mine if any(e.get('rankable') and e['dataset']==dataset['id'] for e in r['evaluations'])]
        tested=[r for r in mine if any(e.get('rankable') and e['dataset']==dataset['id'] and e['split']=='test' for e in r['evaluations'])]
        datasets.append(dict(id=dataset['id'],name=dataset['name'],runs=len(mine),evaluated_runs=len(scored),
            tested_model_families=len({r['model'] for r in tested}),trained_model_families=len({r['model'] for r in mine if trained(r)}),
            benchmark_trained_model_families=len({r['model'] for r in mine if trained(r) and '__research_' not in r['id']}),
            registered_model_families=len({r['model'] for r in mine}),
            rankable_evaluations=sum(bool(e.get('rankable')) and e['dataset']==dataset['id'] for r in mine for e in r['evaluations']),
            **({k:dataset[k] for k in ['counts','cases'] if k in dataset})))
    if not any(d['id']=='her2match' for d in datasets):
        datasets.append(dict(id='her2match',name='HER2Match',runs=11,evaluated_runs=0,rankable_evaluations=0,tested_model_families=0,trained_model_families=0,registered_model_families=11,pairs=21172,counts=dict(train=11610,val=3582,test=5980)))
    else:next(d for d in datasets if d['id']=='her2match')['pairs']=21172
    cells=[]
    model_ids=sorted({r['model'] for r in entries}|{'bbdm'},key=lambda k:names.get(k,k).casefold())
    for model in model_ids:
        row=dict(id=model,name=names.get(model,'BBDM' if model=='bbdm' else model),cells=[])
        for dataset in datasets:
            mine=[r for r in entries if r['model']==model and dataset['id'] in r['datasets']]
            scored=[r for r in mine if any(e.get('rankable') and e['dataset']==dataset['id'] for e in r['evaluations'])]
            tests=[r for r in mine if any(e.get('rankable') and e['dataset']==dataset['id'] and e['split']=='test' for e in r['evaluations'])]
            state='evaluated' if scored else 'trained / unscored' if any(trained(r) for r in mine) else 'registered' if mine else 'none'
            if dataset['id']=='her2match' and not scored and model in HER2_MODELS:
                state=next(r for r in recent['her2match'] if r['id']==model)['state']
            row['cells'].append(dict(dataset=dataset['id'],runs=len(mine),evaluated_runs=len(scored),tested_runs=len(tests),
                score_rows=sum(bool(e.get('rankable')) and e['dataset']==dataset['id'] for r in mine for e in r['evaluations']),status=state))
        cells.append(row)
    coverage=dict(datasets=[dict(id=d['id'],name=d['name']) for d in datasets],rows=cells,
        note='Cells count run records; pooled records occur in each applicable dataset. Dataset bars use distinct model families with full held-out tests. Trained research families with other protocols remain separate from the native benchmark targets.')
    balance=None
    plan_path=workspace/'artifacts/coverage_balance_20261119/prepared/campaign.json'
    if plan_path.exists():
        plan=read_json(plan_path);sources.append(source_record(plan_path,'Focused coverage-balancing campaign'))
        public_tasks=[]
        for task in plan['tasks']:
            scope=['her2match' if d=='HER2Match' else d for d in task.get('datasets',[task['dataset']])]
            public_tasks.append(dict(id=task['id'],model=task['model'],datasets=scope,marker=task['marker'],state=task.get('state','prepared'),job_id=task.get('job_id')))
        table=[]
        for d in datasets:
            additions=sorted({t['model'] for t in public_tasks if d['id'] in t['datasets']})
            if not additions:continue
            current_models={r['model'] for r in entries if d['id'] in r['datasets'] and trained(r) and '__research_' not in r['id']}
            table.append(dict(id=d['id'],name=d['name'],added_models=additions,target_model_families=len(current_models|set(additions)),target_kind='planned benchmark families'))
        submission=read_json(workspace/'artifacts/coverage_balance_20261119/submission.json') if (workspace/'artifacts/coverage_balance_20261119/submission.json').exists() else {}
        balance=dict(task_count=len(public_tasks),job_id=submission.get('array_job_id',submission.get('job_id')),snapshot=plan.get('created'),models=[dict(id=m,name=names.get(m,m)) for m in sorted({t['model'] for t in public_tasks})],tasks=public_tasks,datasets=table,target_note='Targets count task-specific benchmark families after the new runs finish. Research validation-only families are not counted as completed benchmark targets.')
    cohort_entries = [(run, evaluation) for run in entries for evaluation in run['evaluations']
                      if evaluation.get('rankable') and evaluation['dataset'] == 'finland'
                      and evaluation['split'] == 'test' and evaluation['protocol'] == 'Full production evaluation']
    assert len({e['manifest_sha256'] for _, e in cohort_entries}) == 1, 'Cohort manifest mismatch'
    metrics = []
    for key in RADAR_METRICS:
        metadata = data['metrics'][key]
        values = [e['metrics'][key] for _, e in cohort_entries]
        assert all(e.get('coverage', {}).get(key, e['n']) == e['n'] == 5179 for _, e in cohort_entries)
        metrics.append(dict(key=key, label=metadata['label'], direction='higher' if metadata['direction'] == '↑' else 'lower',
            unit=metadata['unit'], min=min(values), max=max(values), coverage_n=5179,
            normalization=f'Min/max across all {len(cohort_entries)} recorded cohort models; larger plotted values are better.'))
    cohort_runs = []
    for run, evaluation in cohort_entries:
        values = {k: evaluation['metrics'][k] for k in RADAR_METRICS}
        normalized = {m['key']: normalize(values[m['key']], m['min'], m['max'], m['direction'] == 'higher') for m in metrics}
        cohort_runs.append(dict(id=run['id'], name=run['name'], values=values, normalized=normalized,
            training_hours=run.get('training_hours'), gpu_hours=run.get('gpu_hours'),
            href=f'#/run/{run["id"]}', coverage={k: evaluation.get('coverage', {}).get(k, evaluation['n']) for k in RADAR_METRICS},
            checkpoint_sha256=evaluation['checkpoint_sha256']))
    cohort = dict(dataset='finland', dataset_name='Finland mouse prostate', split='test', protocol='Full production evaluation',
        metric_protocol=cohort_entries[0][1]['metric_protocol'], n=5179, metric_count=len(metrics),
        manifest_sha256=cohort_entries[0][1]['manifest_sha256'], models=len(cohort_runs),
        default_runs=DEFAULT_RUNS, metrics=metrics, runs=cohort_runs,
        normalization_basis=f'All {len(cohort_entries)} fully evaluated models in this frozen Finland test cohort, unaffected by the selected display subset.',
        note='Relative profiles are not probabilities, absolute quality or a combined clinical score. Raw values and protocol remain authoritative. One training seed per model; no significance claim.')
    trajectory_series = []
    for ident in TRAJECTORY_RUNS:
        path = dist / 'data/runs' / f'{ident}.json'
        run = read_json(path)
        points = run['curves']['val/lpips']
        assert all(points[i][0] < points[i + 1][0] for i in range(len(points) - 1))
        assert all(p[0] % 103 == 0 for p in points), 'Unexpected BCI training-update cadence'
        assert all(e['n'] == 585 for e in run['evaluations'] if e['split'] == 'val')
        trajectory_series.append(dict(id=ident, name=run['name'], points=points,
            recorded_points=run['curve_counts']['val/lpips'], axis='optimizer updates',
            href=f'#/run/{ident}', selected_step=run.get('position', {}).get('global_step'),
            source_sha256=file_sha256(path)))
        sources.append(source_record(path, f'{run["name"]} BCI HER2 validation trajectory', data['snapshot'], f'data/runs/{ident}.json'))
    trajectories = dict(metric='lpips', dataset='BCI', dataset_name='BCI', marker='HER2', split='val', n=585,
        protocol='Box SSIM · 256px · n=585', x_label='Optimizer updates', y_label='Validation LPIPS ↓',
        series=trajectory_series, note='Actual recorded values; no smoothing or extrapolation. Same BCI validation cohort. Training recipes differ; equal updates do not imply equal training compute. Validation trajectories do not use test scores.')
    return dict(schema='virtual-staining-visual-summary-v1', snapshot=dict(benchmark=data['snapshot'], activity=recent['observed_utc']),
        scale=scale, datasets=datasets, coverage=coverage, cohort=cohort, trajectories=trajectories,
        activity=recent, provenance=sources,coverage_balance=balance,
        notes=['Current result inventory and activity retain explicit source and capture timestamps.',
            f'{len(entries)} run records include evaluated, deferred, continuation and unscored entries; they are not independently completed training experiments.',
            'Recorded GPU-hours sum available per-run evidence only. Missing allocations and parent compute are not estimated; a continuation is not total training compute.',
            'Scalar measurements count logged metric values across series; they are not independent samples or optimizer updates.',
            'HER2Match completed full-split results are published only from their selected-checkpoint final reports; unfinished runs remain unscored.',
            'Unmeasured coverage is never replaced by a zero score. Run coverage is evidence scope, not model quality.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--dist', type=Path, default=Path(__file__).resolve().parents[1] / 'dist')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    payload = build(args.dist.resolve(), args.workspace.resolve())
    output = args.output or args.dist / 'data/visual-summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n'
    for forbidden in ['/Users/', '/sofia/', '/user/', 'vsc20775', 'case_id', 'patient_id']:
        if forbidden in text:
            raise ValueError(f'Private path or subject metadata leaked: {forbidden}')
    output.write_text(text)
    print(json.dumps(dict(output=str(output), bytes=len(text.encode()), scale=payload['scale'], cohort_models=payload['cohort']['models'], validation='passed')))


if __name__ == '__main__':
    main()
