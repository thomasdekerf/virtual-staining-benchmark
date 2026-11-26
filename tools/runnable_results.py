"""Reviewed BCI coverage and separately recorded Finland continuation evidence."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from her2_campaigns import (digest,verify_resume_gate,verify_final_report,
    review_new_campaign_protocol,current_snapshot,read,HEX)

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_bbdm_extension(gate,provenance,task,parent):
    require(gate.get('passed') is True and gate.get('world_size')==4
        and gate.get('source_sha256')==provenance.get('source_sha256')==task['source_sha256']
        and gate.get('parent_step')==provenance.get('parent_step')==task['parent_step']==100000
        and gate.get('observed_step')==100001
        and gate.get('parent_checkpoint_sha256')==provenance.get('parent_checkpoint_sha256')
        and gate.get('actual_online_and_ema_updates') is True
        and gate.get('selected_reference_preserved') is True and gate.get('test_used') is False
        and provenance.get('test_used_for_continuation_decision') is False,
        'Finland BBDM extension must bind its actual terminal-parent online/EMA resume')
    require(gate.get('learning_rates')==provenance.get('learning_rates')==[.0001]
        and gate.get('optimizer_min_max_steps')==[100001.,100001.]
        and gate.get('scheduler_last_epoch')==100001
        and gate.get('online_learned_sha256')!=provenance.get('parent_online_learned_sha256')
        and gate.get('ema_learned_sha256')!=provenance.get('parent_ema_learned_sha256')
        and all(HEX.fullmatch(gate.get(k,'')) for k in ['online_learned_sha256','ema_learned_sha256'])
        and provenance.get('metadata_only_rebinding_verified_after_reload') is True
        and provenance.get('inherited_files',{}).get('best.pt')==parent.get('checkpoint_sha256'),
        'Finland BBDM extension changed positive-LR learned state or selected parent reference')

def integrate(site,workspace,data,runs,register,fresh_run,status,history,ingest,source,write,previous_evals):
    import yaml
    path=workspace/'artifacts/run_update_20261126/site_evidence/unmapped_jobs.json'
    if not path.exists():return None
    observed=read(path);source(path,'Current native BCI campaign and Finland terminal-parent continuation metadata')
    for row in observed['staged_files']:
        require(sha(site/'.source-cache'/row['path'])==row['sha256'],'Current runnable campaign staged bytes changed')
    _,snapshot=current_snapshot(workspace);queue={row['job']:row for row in snapshot['active_jobs']}
    policy=read(site/'tools/published-models.json');reference=read(site/'dist/data/protocol-compatibility.json')
    plan_path=workspace/'artifacts/runnable_audit_20261125/prepared_bci/campaign.json';plan=read(plan_path)
    by_id={row['task']['id']:row for row in observed['records']};names={m['id']:m['name'] for m in data['models']}
    public=[];jobs=data['gpu_activity']['jobs']
    for task in plan['tasks']:
        rid=task['public_run_id'];require(rid in policy['eligible_run_ids'],'BCI task lacks an explicit reviewed public ID')
        record=by_id[task['id']];directory=site/'.source-cache/runnable_20261125/bci_published_coverage_20261125_v1/runs'/task['id']
        smoke=directory.parent.parent/'smokes'/task['id'];prepared=plan_path.parent/'configs'/Path(task['config']).name
        require(sha(prepared)==task['config_sha256'],'BCI prepared configuration bytes changed')
        cfg=yaml.safe_load(prepared.read_text());runtime=yaml.safe_load((directory/'config.yml').read_text())
        require(digest(cfg)==digest(runtime)==task['scientific_config_sha256'],'BCI native scientific recipe changed')
        check=snapshot['source_checks'][task['code_dir']]
        require(check.get('passed') and check.get('actual_sha256')==task['source_sha256'],'BCI frozen native source changed')
        training_source=read(directory/'training_source.json')
        require(training_source.get('source_sha256')==task['source_sha256']
            and training_source.get('scientific_config_sha256')==task['scientific_config_sha256']
            and training_source.get('world_size')==4 and training_source.get('model')==task['model'],
            'BCI runtime source/config/model binding changed')
        verify_resume_gate(read(smoke/'smoke_receipt.json'),cfg,task,read(smoke/'resume_provenance.json'))
        run=register(runs.get(rid) or fresh_run(rid,task['model'],'BCI',name=names[task['model']]))
        run.update(task='ihc',marker='HER2',variant='BCI HER2 · native published model · seed42',seed=42,
            train_tiles=3311,crop_size=256,gate_passed=True,source_sha256=task['source_sha256'],
            scientific_config_sha256=task['scientific_config_sha256'],config_sha256=task['config_sha256'],
            configuration=prepared.read_text(),status_as_of=observed['captured_utc'],clinical_score_evaluation=False,
            evaluations=[],sources=[],training_job=f'1662002_{task["index"]}')
        status(run,directory);history(run,directory,force=True)
        run['status_as_of']=observed['captured_utc'];run['evaluation_complete']=False
        final=directory/'final_evaluation.json'
        if final.exists():
            require(run['training_complete'],'BCI final report precedes native training completion')
            verify_final_report(read(final),cfg,task);ingest(run,final,'BCI','coverage_breast')
            run.update(evaluation_complete=True,state='Evaluated',scheduler_status='Complete')
            # Reuse a fully fingerprinted frozen source tree for common-code
            # equivalence, rather than assuming the donor and new cohort match.
            candidates=[]
            for campaign in snapshot['campaigns']:
                other_plan=workspace/campaign['path']
                if any(t.get('source_sha256')==task['source_sha256'] for t in read(other_plan)['tasks']):
                    if (other_plan.parent/'code').is_dir():candidates.append(other_plan.parent/'code')
            require(candidates,'BCI common metric/data source is unavailable for independent review')
            review=review_new_campaign_protocol(run,task,plan_path,site,reference,previous_evals,source_code_dir=candidates[0])
            require(review and all(e.get('comparison_protocol_review_sha256') for e in run['evaluations'] if e.get('rankable')),
                'BCI new scores lack a bound common metric/data protocol review')
            source(review,'BCI independently checked shared metric/data code equivalence')
        else:
            q=queue.get(run['training_job'],{})
            run.update(state='Training' if q.get('state')=='RUNNING' else 'Checkpointed / queued',scheduler_status=q.get('state','Not captured'),scheduler_reason=q.get('reason'))
        for file,label in [(prepared,'Frozen BCI published-model recipe'),(smoke/'smoke_receipt.json','Actual BCI four-GPU learned update/resume gate')]:
            binding=source(file,label);run['sources'].append(dict(file=binding['file'],sha256=binding['sha256']))
        public.append(dict(run_id=rid,model=names[task['model']],dataset='BCI',job=run['training_job'],state=run['state'],kind='New model integration'))
        q=queue.get(run['training_job'])
        if q and not any(j['job']==q['job'] for j in jobs):
            jobs.append(dict(job=q['job'],run_id=rid,model=task['model'],dataset='BCI',status=q['state'],gpus=4,step=run['last_step'],reason=q['reason'],kind='training'))
    record=by_id['bbdm_finland_HE'];task=record['task'];rid=task['public_run_id']
    require(rid in policy['eligible_run_ids'],'Finland extension lacks an explicit reviewed public ID')
    directory=site/'.source-cache/runnable_20261125/bbdm_finland_plateau_20261125_v1/runs/bbdm_finland_HE'
    gate=read(directory/'continuation_gate.json');provenance=read(directory/'continuation_provenance.json')
    verify_bbdm_extension(gate,provenance,task,runs['balance__bbdm_finland_HE'])
    require(sha(directory/'continuation_provenance.json')==gate['provenance_sha256'],'Finland extension provenance bytes changed')
    run=register(runs.get(rid) or fresh_run(rid,'bbdm','finland',name=names['bbdm']))
    run.update(task='unstained',marker='H&E',variant='Original constant-LR plateau continuation · seed42',seed=42,
        parent_run='balance__bbdm_finland_HE',compute_parent_run='balance__bbdm_finland_HE',parent_step=100000,
        source_sha256=task['source_sha256'],gate_passed=True,training_job='1662029_0',
        evaluations=[],sources=[],compute_complete=False,evaluation_complete=False,
        notes='Separate ongoing continuation of the original100k parent. Its native constant learning rate, EMA, optimizer and patience are preserved. Parent selected-checkpoint results remain separately scored; no continuation score is inferred.')
    status(run,directory);history(run,directory,force=True)
    # Avoid inherited parent logs being counted twice in public scalar totals.
    run['curves']={tag:[point for point in points if point[0]>100000] for tag,points in run['curves'].items()}
    run['curve_counts']={tag:len(points) for tag,points in run['curves'].items()}
    curve_file=site/'dist/downloads'/f'{rid}_curves.csv'
    import csv
    with curve_file.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['tag','step_or_epoch','value'])
        writer.writerows([tag,x,y] for tag,points in run['curves'].items() for x,y in points)
    run.update(status_as_of=observed['captured_utc'],new_optimizer_updates=run['last_step']-100000,
        training_complete=False,evaluation_complete=False)
    q=queue.get(run['training_job'],{});run.update(state='Training' if q.get('state')=='RUNNING' else 'Checkpointed / queued',scheduler_status=q.get('state','Not captured'),scheduler_reason=q.get('reason'))
    for filename in ['continuation_gate.json','continuation_provenance.json','config.yml']:
        binding=source(directory/filename,'Finland BBDM unchanged-native terminal-parent '+filename)
        run['sources'].append(dict(file=binding['file'],sha256=binding['sha256']))
    public.append(dict(run_id=rid,model=names['bbdm'],dataset='Finland',job=run['training_job'],state=run['state'],kind='Native plateau continuation; parent scores separate'))
    if q and not any(j['job']==q['job'] for j in jobs):
        jobs.append(dict(job=q['job'],run_id=rid,model='bbdm',dataset='Finland',status=q['state'],gpus=4,step=run['last_step'],reason=q['reason'],kind='training'))
    ids={row['run_id'] for row in public};data['followup_campaigns']=[row for row in data.get('followup_campaigns',[]) if row['run_id'] not in ids]+public
    data['runnable_campaign']=dict(snapshot=observed['captured_utc'],runs=public,download='data/runnable-campaign.json')
    write(site/'dist/data/runnable-campaign.json',data['runnable_campaign'])
    return data['runnable_campaign']
