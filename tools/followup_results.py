"""Scientific guards for completed integrations and live continuation records."""
import json
import math
import re


def verify_native_gate(receipt, source_sha256, config=None):
    if not (receipt.get('passed') is True and receipt.get('world_size') == 4
            and receipt.get('validation_tiles_recomputed') == 9
            and receipt.get('resumed_step', 0) > receipt.get('first_leg_step', 0)
            and receipt.get('source_sha256') == source_sha256):
        raise ValueError('Actual source-bound four-rank resume gate required')
    if config is not None:
        recipe=receipt.get('recipe',{})
        prepared_protocol={k:v for k,v in config.get('breast_protocol',{}).items() if k!='status'}
        gate_protocol={k:v for k,v in recipe.get('breast_protocol',{}).items() if k!='status'}
        if not (receipt.get('full_width') is True and receipt.get('resume') is True
                and all(recipe.get(key)==config.get(key) for key in ('data','model','loss','validation'))
                and prepared_protocol==gate_protocol
                and all(recipe.get('train',{}).get(key)==config.get('train',{}).get(key)
                        for key in ('batch_size','lr','betas','amp','max_steps','warmup_steps','grad_clip'))):
            raise ValueError('Native gate differs from the frozen production recipe')


def verify_completed_integration(report, config, task, expected_counts):
    if (report.get('source_sha256') != task['source_sha256']
            or set(report.get('full_splits', {})) != {'val', 'test'}
            or not re.fullmatch('[0-9a-f]{64}',report.get('checkpoint_sha256',''))
            or report.get('full_split_counts') != expected_counts):
        raise ValueError('Complete selected-checkpoint val/test result required')
    expected = config['breast_protocol']['manifest_sha256']
    for split, record in report['full_splits'].items():
        per_dataset = record.get('per_dataset', {})
        if (record.get('manifest_sha256') != expected[split]
                or record.get('n') != expected_counts[split]
                or set(per_dataset) != {task['dataset']}
                or per_dataset[task['dataset']].get('n') != expected_counts[split]):
            raise ValueError('Completed integration changed its frozen evaluation cohort')
        measurements=per_dataset[task['dataset']].get('metrics',{})
        if not all(isinstance(measurements.get(key),(int,float))
                   and not isinstance(measurements.get(key),bool)
                   and math.isfinite(measurements[key])
                   for key in ('psnr','ssim','lpips','mae','delta_e76')):
            raise ValueError('Completed integration is missing finite core image metrics')


def apply_live_execution(run, record, captured):
    """Inherited steps and new optimizer work are reported separately."""
    run['last_step'] = record['step']
    run['parent_step'] = record.get('parent_step', 0)
    run['new_optimizer_updates'] = record.get('new_optimizer_updates', 0)
    run['registered_only'] = run['new_optimizer_updates'] == 0
    run['gate_passed'] = bool(record.get('gate_passed'))
    run['status_as_of'] = captured
    run['training_job'] = record['job_id']
    scheduler = record.get('scheduler') or {}
    raw_status=record.get('status')
    saved=raw_status if isinstance(raw_status,dict) else record.get('files',{}).get('status.json',{}).get('data',{})
    if saved.get('complete'):
        run['training_complete'] = True
    if run.get('evaluation_complete'):
        run.update(state='Evaluated', scheduler_status='Complete')
    elif scheduler.get('state') == 'RUNNING':
        run.update(state='Training' if run['gate_passed'] else 'Native update check running', scheduler_status='RUNNING')
    elif scheduler.get('state') == 'PENDING':
        run.update(state='Checkpointed / queued' if run['new_optimizer_updates'] else 'Queued / native update check required', scheduler_status='PENDING')
    elif run.get('training_complete'):
        run.update(state='Training complete / evaluation pending', scheduler_status='Training complete')
    else:
        run.update(state='Checkpointed; scheduler state not captured', scheduler_status='Not captured')
    run['scheduler_reason'] = scheduler.get('reason')
    run['training_dependency'] = scheduler.get('dependency')


def new_research_validation(text,parent_step,expected_n=256):
    """Exclude inherited checks and separate full-validation readouts."""
    selected={}
    for line in text.splitlines():
        if not line.strip():continue
        row=json.loads(line)
        if not (row.get('step',0)>parent_step and row.get('n')==expected_n
                and row.get('stage')=='main' and row.get('test_used') is False):continue
        values={key:row[key] for key in ('psnr','ssim','mae','delta_e76','lpips')
                if isinstance(row.get(key),(int,float)) and math.isfinite(row[key])}
        if 'lpips' not in values:raise ValueError('Continuation validation requires finite LPIPS')
        entry=dict(step=row['step'],n=expected_n,metrics=values)
        if row['step'] in selected and selected[row['step']]!=entry:
            raise ValueError('Conflicting validation records at one continuation update')
        selected[row['step']]=entry
    return [selected[step] for step in sorted(selected)]


def verify_gpu_snapshot(record):
    running=[job for job in record['jobs'] if job['status']=='RUNNING' and job.get('kind')!='held-out evaluation']
    evaluating=[job for job in record['jobs'] if job['status']=='RUNNING' and job.get('kind')=='held-out evaluation']
    if (sum(job['gpus'] for job in running)!=record['own_training_gpus']
            or len(running)!=record['own_running_training_jobs']):
        raise ValueError('Allocated training GPUs must count only running training jobs')
    if sum(job['gpus'] for job in evaluating)!=record.get('own_evaluation_gpus',0):
        raise ValueError('Held-out evaluation GPUs must be identified separately from training')
    partition=record['partition']
    if not (0<=record['own_training_gpus']<=partition['allocated_gpus']<=partition['schedulable_gpus']<=partition['physical_gpus']
            and partition['allocated_gpus']+partition['free_schedulable_gpus']==partition['schedulable_gpus']
            and 0<=partition['largest_free_gpus_per_node']<=partition['free_schedulable_gpus']):
        raise ValueError('GPU availability snapshot is inconsistent')
    preparing=[job for job in record.get('preparation_jobs',[]) if job['status']=='RUNNING']
    if sum(job['gpus'] for job in preparing)!=record.get('own_preparation_gpus',0):
        raise ValueError('Data-preparation GPUs must not be called model-training GPUs')
    if record['own_training_gpus']+record.get('own_preparation_gpus',0)+record.get('own_evaluation_gpus',0)>partition['allocated_gpus']:
        raise ValueError('Training plus preparation exceeds actual allocated partition GPUs')
    blocked=record.get('blocked',{})
    if blocked.get('readiness_groups') and sum(blocked['readiness_groups'].values())!=blocked['run_records']:
        raise ValueError('Deferred readiness groups must describe every blocked record once')
