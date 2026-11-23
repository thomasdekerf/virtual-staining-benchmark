"""Source-bound published HER2Match campaigns and native continuation evidence."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import re


TARGET_MODELS=('asp','bbdm','bcistainer','cssp2p','cut','cyclegan','dvst','dgr','diffvs','hat','hemit','histdist','i2imamba','implicitstainer','mcs_stain','miphei_vit','nafnet','palette','pd_unist','pgvms','pix2pix','pspstain','pyramid','restormer','stable','stainexpert','swinir','tdkstain','unet','unistainnet')
CORE=('psnr','ssim','lpips','mae','delta_e76')
DATASET_COUNTS={'HER2Match':{'train':11610,'val':3582,'test':5980},'BCI':{'train':3311,'val':585,'test':977},'IHC4BC':{'train':15434,'val':3491,'test':1804}}
HEX=re.compile('[0-9a-f]{64}')

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def digest(config):
    if not isinstance(config.get('train'),dict):raise ValueError('Frozen training recipe is missing')
    value=copy.deepcopy(config);value.pop('_config_path',None)
    value['train'].pop('max_seconds',None)
    value.get('breast_protocol',{}).pop('status',None)
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def verify_resume_gate(receipt,config,task,before=None):
    if not (receipt.get('passed') is True and receipt.get('source_sha256')==task['source_sha256']
            and receipt.get('world_size')==4 and receipt.get('resume') is True
            and receipt.get('full_width') is True and receipt.get('validation_tiles_recomputed')==9
            and receipt.get('resumed_step',0)>receipt.get('first_leg_step',0)
            and receipt.get('positive_learning_rate') is True
            and receipt.get('actual_model_tensor_update') is True):
        raise ValueError('Actual native four-rank positive-LR tensor-update/resume gate required')
    first,last=receipt.get('first_leg_model_tensor_sha256',''),receipt.get('resumed_model_tensor_sha256','')
    if not (HEX.fullmatch(first) and HEX.fullmatch(last) and first!=last):
        raise ValueError('Native gate must bind different actual model tensor hashes')
    if digest(receipt.get('recipe',{}))!=digest(config):
        raise ValueError('Native gate changed the frozen scientific recipe')
    if task.get('requires_decoupled_registration_gate'):
        after=receipt.get('native_dgr_evidence') or {}
        prior=(before or {}).get('native_dgr_evidence') or {}
        names={'generator','reg_gt','reg_x','discriminator'}
        if not (set(after)==names and set(prior)==names and receipt['resumed_step']>=max(104,task.get('smoke_steps',104))):
            raise ValueError('DGR rescue must exercise both registration branches through104 real updates')
        for name in names:
            a,b=after[name],prior[name]
            if not (finite(a.get('max_optimizer_step')) and finite(b.get('max_optimizer_step'))
                    and a['max_optimizer_step']>b['max_optimizer_step']>=0
                    and a.get('parameter_states',0)>0
                    and HEX.fullmatch(a.get('learned_parameters_sha256','')) and HEX.fullmatch(b.get('learned_parameters_sha256',''))
                    and a['learned_parameters_sha256']!=b['learned_parameters_sha256']
                    and a.get('learning_rates') and all(finite(v) and math.isclose(v,1e-5,rel_tol=1e-8,abs_tol=1e-12) for v in a['learning_rates'])):
                raise ValueError('All four native DGR rescue systems must make real constant-LR learned-weight updates')
    if task['model'] in {'tdkstain','pgvms'}:
        names={'generator','discriminator','estimator' if task['model']=='tdkstain' else 'projections'}
        after=receipt.get('native_specialist_evidence') or {}
        prior=(before or {}).get('native_specialist_evidence') or {}
        if set(after)!=names or set(prior)!=names:raise ValueError('All three native specialist optimizers and modules required')
        for name in names:
            a,b=after[name],prior[name]
            if not (all(finite(v) and v>0 for v in [a.get('max_optimizer_step'),a.get('min_optimizer_step'),b.get('max_optimizer_step'),b.get('min_optimizer_step')])
                    and a['max_optimizer_step']>b['max_optimizer_step']
                    and a.get('parameter_states',0)>0 and b.get('parameter_states',0)>0
                    and HEX.fullmatch(a.get('learned_parameters_sha256','')) and HEX.fullmatch(b.get('learned_parameters_sha256',''))
                    and a['learned_parameters_sha256']!=b['learned_parameters_sha256']
                    and a.get('learning_rates') and all(finite(v) and v>0 for v in a['learning_rates'])):
                raise ValueError('Every native specialist optimizer must resume actual positive-LR updates')
    return True

def verify_continuation_gate(gate,provenance,config,task):
    if not (gate.get('passed') is True and gate.get('source_sha256')==task['source_sha256']
            and gate.get('world_size')==4 and gate.get('parent_step')==task['parent_step']
            and gate.get('observed_step',0)>task['parent_step'] and gate.get('test_used') is False
            and provenance.get('parent_checkpoint_sha256')==task['parent_checkpoint_sha256']
            and provenance.get('source_sha256')==task['source_sha256']
            and provenance.get('manifest_sha256')==config['breast_protocol']['manifest_sha256']):
        raise ValueError('Native continuation parent/source/data/update proof required')
    if task.get('continuation_kind')=='explicit_positive_lr_extension':
        rates=gate.get('positive_learning_rates',{})
        if not (set(rates)=={'optimizer','discriminator_optimizer'}
                and all(values and all(finite(v) and v>0 for v in values) for values in rates.values())
                and gate.get('parameter_updates')=={'generators':True,'discriminators':True}
                and gate.get('original_stopping_reference_preserved') is True):
            raise ValueError('Disclosed CUT stage must update both native positive-LR systems')
    elif not (gate.get('nonzero_parameter_update') is True and gate.get('optimizer_and_schedule_restored') is True
              and gate.get('learning_rates') and all(finite(v) and v>0 for v in gate['learning_rates'])):
        raise ValueError('Native continuation must preserve its positive LR floor and optimizer state')
    return True

def counts(config):
    expected=DATASET_COUNTS[config['data']['datasets'][0]]
    return {split:expected[split] for split in ('val','test')}

def verify_final_report(report,config,task):
    expected=counts(config)
    if not (report.get('source_sha256')==task['source_sha256']
            and HEX.fullmatch(report.get('checkpoint_sha256',''))
            and set(report.get('full_splits',{}))=={'val','test'}
            and report.get('full_split_counts')==expected
            and report.get('protocol',{}).get('manifest_sha256')==config['breast_protocol']['manifest_sha256']
            and report.get('clinical_score_evaluation') is False):
        raise ValueError('Complete source-bound val/test evaluation on the frozen cohort required')
    for split,record in report['full_splits'].items():
        expected_dataset=config['data']['datasets'][0]
        per=record.get('per_dataset',{})
        if not (record.get('n')==expected[split]
                and record.get('manifest_sha256')==config['breast_protocol']['manifest_sha256'][split]
                and set(per)=={expected_dataset} and per[expected_dataset].get('n')==expected[split]
                and all(finite(per[expected_dataset].get('metrics',{}).get(key)) for key in CORE)
                and all(finite(per[expected_dataset].get('dab',{}).get(key)) for key in ('mean_absolute_error','mean_pred','mean_target'))
                and per[expected_dataset]['dab'].get('samples')==expected[split]
                and 'correlation' in per[expected_dataset]['dab']
                and (per[expected_dataset]['dab']['correlation'] is None or finite(per[expected_dataset]['dab']['correlation']))):
            raise ValueError('Complete finite core/DAB measurements for each frozen full split required')
    return True

def verify_parent_reuse(report,parent):
    matched=[e for e in parent['evaluations'] if e.get('rankable') and e.get('checkpoint_sha256')==report['checkpoint_sha256']]
    if not matched:return False
    for split,record in report['full_splits'].items():
        old=next((e for e in matched if e['split']==split),None)
        entry=next(iter(record['per_dataset'].values()))
        if not (old and old['n']==record['n'] and old['manifest_sha256']==record['manifest_sha256']
                and all(entry['metrics'][key]==old['metrics'][key] for key in CORE)):
            raise ValueError('An unchanged parent checkpoint produced different full-split evidence')
    return True

def scheduler_state(record,saved,gate_passed):
    queue=record.get('scheduler') or {};account=record.get('accounting') or {}
    phase=record.get('files',{}).get('launch_state.json',{}).get('data',{}).get('phase','')
    if saved.get('complete'):return 'Training complete / evaluation pending',queue.get('state',account.get('state','COMPLETED'))
    if queue.get('state')=='RUNNING':
        return ('Training' if gate_passed and phase=='production_training' else 'Native update check running'),'RUNNING'
    if queue.get('state')=='PENDING':return 'Checkpointed / queued' if saved.get('step') else 'Queued / native update check required','PENDING'
    terminal=account.get('state','NOT_IN_QUEUE')
    if terminal in {'FAILED','OUT_OF_MEMORY','TIMEOUT','NODE_FAIL','CANCELLED'}:return 'Interrupted / '+terminal.lower().replace('_',' '),terminal
    return 'Prepared / awaiting allocation',terminal

def preparation_summary(snapshot):
    jobs=[]
    for record in snapshot.get('preparation',[]):
        receipt=(record.get('receipt') or {}).get('data',{})
        if receipt and not (receipt.get('used_splits')==['train'] and receipt.get('source_rows')==11610
                and receipt.get('train_manifest_sha256')=='6c8cb9f704e5df8062b15c78b1d6226569e46dd7e24fcb919dbbcf6869fe7cb2'
                and receipt.get('cellpose_version')=='2.2.3' and receipt.get('nuclei_models')==['nuclei','cyto2']
                and receipt.get('complete')==(record['kind']=='full_maps')
                and receipt.get('prepared_rows')==record['required_rows'] and receipt.get('net_avg') is True
                and record['cached_map_files']==record['required_rows']):
            raise ValueError('TDK preparation changed native teachers or TRAIN-only source identity')
        q=record.get('scheduler') or {}
        jobs.append(dict(job=record['job'],model='tdkstain',dataset='HER2Match',kind='data preparation',stage=record['kind'],status=q.get('state',record.get('accounting',{}).get('state','NOT_IN_QUEUE')),gpus=record['gpus'],dependency=record.get('dependency'),required_rows=record['required_rows'],cached_map_files=record['cached_map_files'],complete=bool(receipt.get('complete')),probe_passed=bool(receipt and record['kind']=='probe' and (q.get('state') or record.get('accounting',{}).get('state'))=='COMPLETED'),heldout_images_used=False,native_teacher_version='Cellpose2.2.3 nuclei/cyto2'))
    return jobs

def build_capacity(snapshot,runs,previous):
    jobs=[];seen=set()
    for rid,record in snapshot['records'].items():
        if rid not in runs:continue
        task=record['task'];q=record.get('scheduler');saved=record['files'].get('status.json',{}).get('data',{})
        if runs[rid].get('operational_superseded'):continue
        if not q and not rid.startswith(('her2match_complete__','continuation__')) and rid!='her2match__restormer_continuation_20261121':continue
        job=task.get('job_id')
        if not job or job in seen:continue
        seen.add(job);status=q['state'] if q else record.get('accounting',{}).get('state','COMPLETED' if saved.get('complete') else 'NOT_IN_QUEUE')
        jobs.append(dict(job=job,run_id=rid,model=task['model'],dataset=task['dataset'],status=status,gpus=task.get('per_task_gpus',4),step=saved.get('step',saved.get('global_step',task.get('parent_step') or 0)),new_optimizer_updates=max(0,(saved.get('step',saved.get('global_step',task.get('parent_step') or 0)) or 0)-(task.get('parent_step') or 0)),reason=q['reason'] if q else 'Selected checkpoint evaluated' if runs[rid].get('evaluation_complete') else 'Native terminal status' if saved.get('complete') else 'No current allocation'))
    for job in jobs:
        run=runs.get(job.get('run_id'),{})
        if job['status'] in {'FAILED','TIMEOUT','CANCELLED'} and run.get('evaluation_complete'):
            job.update(evaluation_recovered=True,reason='Original allocation ended with an evaluation error; full selected-checkpoint evaluation was recovered separately.')
    external=snapshot.get('external_validation',{})
    if external:
        if not (external.get('trained_on_her2match') is False and external.get('in_domain_training_complete') is False
                and external.get('training_updates')==0 and external.get('evaluation_scope')=='EXTERNAL_AUTHOR_EMA_TRANSFER'):
            raise ValueError('External author-checkpoint transfer cannot masquerade as in-domain training')
        q=external.get('scheduler') or {}
        jobs.append(dict(job=external['job_id'],model='bcistainer',dataset='HER2Match',status=q.get('state',external.get('accounting',{}).get('state','NOT_IN_QUEUE')),gpus=external['gpus'],step=0,reason=q.get('reason','External author EMA transfer validation'),kind='held-out evaluation',scope='external author EMA transfer',training_updates=0))
    # This existing one-GPU dependent evaluation belongs to a published BBDM run.
    for q in snapshot.get('active_jobs',[]):
        if q['job']=='1642833_15' and q['job'] not in seen:
            jobs.append(dict(job=q['job'],run_id='balance__bbdm_finland_HE',model='bbdm',dataset='Finland',status=q['state'],gpus=1,step=runs.get('balance__bbdm_finland_HE',{}).get('last_step',0),reason=q['reason'],kind='held-out evaluation'))
    nodes=snapshot['parsed_nodes'];running=[j for j in jobs if j['status']=='RUNNING']
    result=copy.deepcopy(previous)
    preparation=preparation_summary(snapshot)
    result.update(preparation_jobs=preparation,own_preparation_gpus=sum(j['gpus'] for j in preparation if j['status']=='RUNNING'),snapshot=snapshot['captured_utc'],jobs=jobs,own_training_gpus=sum(j['gpus'] for j in running if j.get('kind')!='held-out evaluation'),own_running_training_jobs=sum(j.get('kind')!='held-out evaluation' for j in running),own_evaluation_gpus=sum(j['gpus'] for j in running if j.get('kind')=='held-out evaluation'),own_running_evaluation_jobs=sum(j.get('kind')=='held-out evaluation' for j in running),download='data/gpu-activity.json')
    result['partition']=dict(name='zen4_h200',physical_gpus=sum(n['gpus'] for n in nodes),schedulable_gpus=sum(n['gpus'] for n in nodes if n['usable']),allocated_gpus=sum(n['allocated_gpus'] for n in nodes),free_schedulable_gpus=sum(n['free_gpus'] for n in nodes),largest_free_gpus_per_node=max((n['free_gpus'] for n in nodes),default=0),nodes=nodes)
    result.setdefault('limits',{}).update(artificial_two_new_allocation_cap_removed=True,new_campaign_artificial_concurrency_cap=None,original_per_run_gpu_count_and_global_batch_preserved=True)
    return result

def current_snapshot(workspace):
    candidates=[]
    for path in (workspace/'artifacts').glob('her2match_complete_*/live_audit/site_snapshot.json'):
        value=read(path,{})
        if value.get('captured_utc'):candidates.append((value['captured_utc'],path,value))
    if not candidates:return None,{}
    _,path,value=max(candidates,key=lambda row:row[0]);return path,value

def integrate(site,workspace,data,runs,register,fresh_run,status,history,ingest,source,write):
    import yaml
    snapshot_path,snapshot=current_snapshot(workspace)
    if not snapshot:return None
    captured=snapshot['captured_utc'];policy=read(site/'tools/published-models.json')
    names={m['id']:m['name'] for m in data['models']};new_ids=[]
    source(snapshot_path,'Fresh read-only campaign gates, native updates and actual GPU queue')
    for plan_record in snapshot['campaigns']:
        plan_path=workspace/plan_record['path'];plan=read(plan_path)
        source(plan_path,'Frozen canonical published-model campaign and scientific configuration hashes')
        for task in plan['tasks']:
            rid=task.get('public_run_id')
            if not rid or rid not in policy['eligible_run_ids']:continue
            if task['model'] not in policy['models']:raise ValueError('Campaign method has no reviewed paper')
            record=snapshot['records'].get(rid,{})
            if not record:raise ValueError('Campaign task lacks a bound observation')
            check=snapshot['source_checks'].get(task['code_dir'],{})
            if not (check.get('passed') and check.get('actual_sha256')==task['source_sha256']):raise ValueError('Frozen source fingerprint changed')
            prepared=plan_path.parent/'configs'/Path(task['config']).name
            if not prepared.exists():prepared=plan_path.parent/'configs'/(task['id']+'.yml')
            raw=prepared.read_bytes();expected=task.get('config_sha256',task.get('prepared_config_sha256'))
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Prepared campaign config bytes changed')
            cfg=yaml.safe_load(raw)
            if task.get('scientific_config_sha256') and digest(cfg)!=task['scientific_config_sha256']:raise ValueError('Scientific configuration changed')
            if cfg['data']['datasets']!=task.get('datasets',[task['dataset']]) or cfg['data']['marker']!=task['marker']:raise ValueError('Task/config dataset identity mismatch')
            if task['dataset']=='HER2Match':
                checks={row['path']:row for row in snapshot['manifest_checks']}
                for split,expected_hash in cfg['breast_protocol']['manifest_sha256'].items():
                    row=checks.get(str(Path(cfg['data']['root'])/cfg['data'][split+'_manifest']),{})
                    if not (row.get('passed') is True and row.get('actual_sha256')==expected_hash
                            and row.get('rows')==DATASET_COUNTS['HER2Match'][split]):
                        raise ValueError('Actual HER2Match manifest hash or full split size changed')
            directory=site/'.source-cache'/record['task']['stage']
            runtime=directory/'submitted_config.yml'
            if runtime.exists() and digest(yaml.safe_load(runtime.read_text()))!=digest(cfg):raise ValueError('Runtime scientific recipe changed')
            training_source=record['files'].get('training_source.json',{}).get('data')
            if training_source and not (training_source.get('source_sha256')==task['source_sha256']
                    and training_source.get('world_size')==4
                    and training_source.get('scientific_config_sha256')==digest(cfg)
                    and training_source.get('model')==task['model']):raise ValueError('Runtime training source or frozen recipe changed')
            dataset='her2match' if task['dataset']=='HER2Match' else task['dataset']
            continuation=task.get('continuation_kind') in {'explicit_positive_lr_extension','unchanged_native_positive_floor'}
            parent_id='her2match__cut_seed42' if task['model']=='cut' and continuation else 'balance__'+task['id'] if continuation else None
            run=register(runs.get(rid) or fresh_run(rid,task['model'],dataset,name=names[task['model']],variant=('Explicit positive-LR extension' if task.get('continuation_kind')=='explicit_positive_lr_extension' else 'Native positive-floor continuation' if continuation else 'HER2Match author cases · canonical published model')+' · seed42'))
            run.update(source_sha256=task['source_sha256'],source_inventory_sha256=task['source_sha256'],frozen_config_sha256=hashlib.sha256(raw).hexdigest(),scientific_config_sha256=digest(cfg),crop_size=cfg['data']['crop_size'],seed=cfg.get('seed',42),train_tiles=DATASET_COUNTS[task['dataset']]['train'],training_job=record['task'].get('job_id'),status_as_of=captured,clinical_score_evaluation=False,configuration=runtime.read_text() if runtime.exists() else raw.decode(),parent_step=task.get('parent_step',0),gate_passed=False,evaluation_complete=False,max_steps=cfg['train'].get('max_steps'))
            if parent_id:run.update(parent_run=parent_id,compute_parent_run=parent_id,compute_complete=False,continuation_kind=task['continuation_kind'],notes=task['reason'])
            recovery=task.get('numerical_recovery')
            if recovery:
                if not (recovery.get('parent_checkpoint_reused') is False and recovery.get('fresh_same_seed') is True
                        and recovery.get('test_used_for_hyperparameter_selection') is False and recovery.get('retained_architecture_objectives') is True):
                    raise ValueError('Numerical rescue must preserve its disclosed fresh-start/paper-method scope')
                run.update(numerical_recovery=copy.deepcopy(recovery),variant='Fresh seed42 '+names[task['model']]+' numerical recovery · FP32',notes='Fresh same-seed rescue after the separately retained colour-collapse failure. Parent weights and optimizer are not reused. Submitted/running repair does not establish successful quality; only complete validation-selected held-out results can be scored.')
            status(run,directory);history(run,directory,force=True)
            gate=record['files'].get('continuation_gate.json' if continuation else 'smoke/smoke_receipt.json',{}).get('data')
            if gate:
                gate_key='continuation_gate.json' if continuation else 'smoke/smoke_receipt.json'
                run['native_gate_evidence']=dict(file=str((directory/gate_key).relative_to(workspace)),sha256=record['files'][gate_key]['sha256'],source_sha256=task['source_sha256'],world_size=4,actual_tensor_update=True,positive_learning_rate=True)
                if continuation:verify_continuation_gate(gate,record['files'].get('continuation_provenance.json',{}).get('data',{}),cfg,task)
                else:verify_resume_gate(gate,cfg,task,record['files'].get('smoke/resume_provenance.json',{}).get('data'))
                run['gate_passed']=True
            saved=record['files'].get('status.json',{}).get('data',{})
            run['last_step']=saved.get('step',saved.get('global_step',task.get('parent_step',0))) or 0
            run['new_optimizer_updates']=max(0,run['last_step']-(task.get('parent_step') or 0))
            run['registered_only']=run['new_optimizer_updates']==0
            run['training_complete']=bool(saved.get('complete'))
            run['state'],run['scheduler_status']=scheduler_state(record,saved,run['gate_passed'])
            run['scheduler_reason']=(record.get('scheduler') or {}).get('reason')
            run['launch_phase']=record['files'].get('launch_state.json',{}).get('data',{}).get('phase')
            if run.get('numerical_recovery') and run['scheduler_status']=='CANCELLED' and run['new_optimizer_updates']==0 and record.get('accounting',{}).get('elapsed')=='00:00:00':
                run.update(state='Superseded before GPU allocation; no model updates',operational_superseded=True)
            if not run.get('training_job') and task.get('preflight_prerequisites'):
                run.update(state='Preparing original specialist prerequisites',preparation_prerequisites=task['preflight_prerequisites'],notes='; '.join(task['preflight_prerequisites'])+'. No training job has been submitted yet.')
            final=directory/'final_evaluation.json'
            if final.exists():
                if not run['gate_passed'] or not run['training_complete']:raise ValueError('A final report cannot bypass native training gate/completion')
                report=read(final);verify_final_report(report,cfg,task)
                if parent_id and verify_parent_reuse(report,runs[parent_id]):
                    run.update(evaluation_reuses_parent=True,evaluation_reference_run=parent_id,evaluation_complete=True,state='Completed; selected checkpoint unchanged',checkpoint_sha256=report['checkpoint_sha256'],selection=report['selection'],position=report['position'])
                    binding=source(final,'Full continuation evaluation confirms unchanged selected parent checkpoint');run['sources'].append({'file':binding['file'],'sha256':binding['sha256']})
                else:
                    ingest(run,final,dataset,'coverage_her2match' if dataset=='her2match' else 'coverage_breast');run['evaluation_complete']=True
            binding=source(plan_path,'Reviewed canonical campaign task and frozen configuration')
            run['sources'].append({'file':binding['file'],'sha256':binding['sha256'],'record':task['id']})
            if rid=='her2match_complete__dgr_20261123':
                run.update(quality_failure=True,quality_warning='This completed DGR run collapsed to a saturated almost constant colour output. Complete evaluation is not a successful training outcome; an isolated repair is being prepared. The original poor result remains visible.')
            new_ids.append(rid)
    collapsed=runs.get('her2match_complete__dgr_20261123')
    rescues=[r for r in runs.values() if r.get('numerical_recovery') and r['model']=='dgr' and 'her2match' in r['datasets']]
    if collapsed and rescues:
        collapsed['quality_warning']='This completed DGR run collapsed to a saturated almost constant colour output. It is a failed-quality result. An isolated fresh same-seed RGB rescue has been submitted; its useful quality is not established by submission or a startup gate.'
    quality_path=snapshot_path.parent.parent/'independent_review/mcs_distinct_val_prediction_diagnosis.json'
    quality=read(quality_path,{})
    mcs=runs.get('her2match_complete__mcs_stain_20261123')
    if quality and mcs:
        if not (quality.get('read_only') is True and quality.get('split')=='val' and quality.get('optimizer_steps_performed')==0):
            raise ValueError('Numerical quality audit must remain validation-only and perform no updates')
        mcs.update(quality_failure=True,quality_warning='The recorded MCS-Stain run collapsed to an almost constant white output on four distinct validation cases. Its real poor evaluation remains visible; a plateau does not establish useful prediction quality.',quality_warning_as_of=quality.get('observed_utc'))
        binding=source(quality_path,'Independent four-case validation-only MCS colour-collapse diagnosis')
        mcs['sources'].append({'file':binding['file'],'sha256':binding['sha256']})
    # The unchanged Restormer selected parent is evidence, not another ranked model.
    rid='her2match__restormer_continuation_20261121';record=snapshot['records'].get(rid)
    if rid in runs and record:
        run=runs[rid];directory=site/'.source-cache/followups/restormer'
        status(run,directory);history(run,directory,force=True)
        saved=record['files'].get('status.json',{}).get('data',{})
        run['last_step']=saved.get('step',saved.get('global_step',run.get('last_step',0)))
        run['new_optimizer_updates']=max(0,run['last_step']-80000);run['training_complete']=bool(saved.get('complete'));run['status_as_of']=captured
        run['state'],run['scheduler_status']=scheduler_state(record,saved,run.get('gate_passed',False))
        final=read(directory/'final_evaluation.json',{})
        if final:
            cfg=yaml.safe_load((directory/'config.yml').read_text())
            provenance=read(directory/'continuation_provenance.json')
            native_task=dict(source_sha256=run['source_sha256'],parent_step=80000,parent_checkpoint_sha256=provenance['parent_checkpoint_sha256'])
            verify_continuation_gate(read(directory/'continuation_gate.json'),provenance,cfg,native_task)
            verify_final_report(final,cfg,native_task)
        parent=runs['her2match__restormer_seed42']
        if final and verify_parent_reuse(final,parent):
            run.update(evaluation_complete=True,evaluation_reuses_parent=True,evaluation_reference_run=parent['id'],state='Completed; selected checkpoint unchanged',checkpoint_sha256=final['checkpoint_sha256'],position=final['position'],selection=final['selection'])
            binding=source(directory/'final_evaluation.json','Completed native continuation retains its original selected parent');run['sources'].append({'file':binding['file'],'sha256':binding['sha256']})
    # Genuine tile grades are unavailable; WSI metadata must never become crop labels.
    blocked_id='her2match_complete__bcistainer_blocked_20261123'
    if blocked_id in policy['eligible_run_ids']:
        run=register(runs.get(blocked_id) or fresh_run(blocked_id,'bcistainer','her2match',variant='Original grade-conditioned classifier/comparator · blocked'))
        run.update(state='Blocked / genuine tile-level HER2 grades unavailable',scheduler_status='BLOCKED',scientific_blocker='The published classifier/comparator requires genuine crop-level HER2 grades. HER2Match supplies WSI-level metadata only; no slide score is inferred as a crop label.',notes='The published classifier/comparator requires genuine crop-level HER2 grades. HER2Match supplies WSI-level metadata only; no slide score is inferred as a crop label.',registered_only=True,last_step=0,training_complete=False,gate_passed=False,clinical_score_evaluation=False,status_as_of=captured)
    # Refresh actual legacy execution states without claiming smoke steps as production.
    for rid,record in snapshot['records'].items():
        if rid not in runs or rid in new_ids or rid=='her2match__restormer_continuation_20261121':continue
        run=runs[rid];saved=record['files'].get('status.json',{}).get('data',{})
        if saved:run['last_step']=saved.get('step',saved.get('global_step',run.get('last_step',0)));run['training_complete']=bool(saved.get('complete'))
        run['status_as_of']=captured
        if not any(e.get('rankable') for e in run['evaluations']):run['state'],run['scheduler_status']=scheduler_state(record,saved,run.get('gate_passed',False))
    retired_path=snapshot_path.parent.parent/'submission_v4.json'
    retired=read(retired_path,{})
    old=runs.get('her2match_complete__dgr_rgb_rescue_20261123')
    if old and retired.get('phase')=='CANCELLED_BEFORE_ALLOCATION_SUPERSEDED_BY_V4B':
        if not (retired.get('source_sha256')==old['source_sha256'] and old.get('new_optimizer_updates',0)==0
                and any('CANCELLED' in row and '|00:00:00|' in row for row in retired.get('accounting',[]))):
            raise ValueError('Retired diagnostic attempt must be source-bound and prove zero allocation/update work')
        old.update(state='Superseded before GPU allocation; no model updates',scheduler_status='CANCELLED',operational_superseded=True,superseded_by=retired['superseded_by'],superseded_public_run_id=retired['superseded_public_run_id'],notes=retired['cancellation_reason'])
        binding=source(retired_path,'Source-bound cancellation and replacement receipt for zero-update diagnostic attempt')
        old['sources'].append({'file':binding['file'],'sha256':binding['sha256']})
    tdk=next((r for r in runs.values() if r['id']=='her2match_complete__tdkstain_20261123'),None)
    if tdk and snapshot.get('preparation'):
        preparation=preparation_summary(snapshot)
        tdk['preparation_jobs']=preparation
        tdk['training_dependency']='afterok:'+next(row['job'] for row in preparation if row['stage']=='full_maps')
        if not any(row['stage']=='full_maps' and row['complete'] for row in preparation):
            tdk['state']='Queued / waiting for original TRAIN-only Cellpose maps'
            tdk['notes']='Original Cellpose2.2.3 nuclei/cyto2 supervision must be prepared on all11610TRAIN fields. A one-field probe is never a complete production map set. Training starts only after successful full-map preparation; held-out images are not used.'
    data['gpu_activity']=build_capacity(snapshot,runs,data.get('gpu_activity',{}));write(site/'dist/data/gpu-activity.json',data['gpu_activity'])
    data['followup_campaigns']=[r for r in data.get('followup_campaigns',[]) if r['run_id'] in runs and runs[r['run_id']]['model'] in policy['models']]
    for task in data['followup_campaigns']:task['state']=runs[task['run_id']]['state']
    for rid in new_ids:
        run=runs[rid]
        data['followup_campaigns'].append(dict(run_id=rid,model=names[run['model']],dataset=run['datasets'][0],job=run.get('training_job'),state=run['state'],kind='Budget-truncated native continuation' if run.get('parent_run') else 'Complete HER2Match model coverage'))
    data['her2match_campaign']=dict(snapshot=captured,target_model_families=len(TARGET_MODELS),target_models=list(TARGET_MODELS),preparation_jobs=preparation_summary(snapshot),clinical_score_evaluation=False,download='data/her2match-campaign.json',note='Every square represents the original published method or a conventional dataset adaptation. Queued, training and blocked entries carry no fabricated test score.')
    external=snapshot.get('external_validation',{})
    if external:
        external_public={key:external.get(key) for key in ['job_id','evaluation_scope','trained_on_her2match','in_domain_training_complete','training_updates','checkpoint_sha256']}
        external_public['status']=(external.get('scheduler') or {}).get('state',external.get('accounting',{}).get('state','NOT_IN_QUEUE'))
        data['her2match_campaign']['external_validation']=external_public
    data['her2match_campaign']=campaign_summary(dict(data,runs=list(runs.values())))
    write(site/'dist/data/her2match-campaign.json',data['her2match_campaign'])
    # Extend the separately downloadable audit to cover the newly observed records.
    audit_path=site/'dist/data/training-audit.json';audit=read(audit_path,{})
    if audit:
        known={r['id']:r for r in audit['runs']}
        for rid in new_ids+[blocked_id]:
            if rid not in runs:continue
            run=runs[rid]
            classification=('completed_with_quality_failure' if run.get('training_complete') else 'quality_failure_requires_repair') if run.get('quality_failure') else 'validation_patience_exhausted' if run.get('training_complete') else 'incomplete_below_configured_budget' if run.get('gate_passed') else 'registered_without_training_evidence'
            row=dict(id=rid,model=run['model'],datasets=run['datasets'],marker='HER2',classification=classification,position=run.get('last_step',0),configured_budget=run.get('max_steps'),validation_scope='original full frozen validation split',reason=run.get('quality_warning',run.get('scientific_blocker',run.get('notes',run['state']))),quality_failure=bool(run.get('quality_failure')),selected_parent_unchanged=bool(run.get('evaluation_reuses_parent')))
            if rid in known:known[rid].update(row)
            else:audit['runs'].append(row)
        audit['snapshot']=captured;write(audit_path,audit);data['training_audit']={k:v for k,v in audit.items() if k!='runs'}|{'download':'data/training-audit.json'}
    return captured


def campaign_summary(index):
    campaign=index.get('her2match_campaign')
    if not campaign:return None
    result=copy.deepcopy(campaign);names={m['id']:m['name'] for m in index['models']};rows=[]
    for model in TARGET_MODELS:
        mine=[r for r in index['runs'] if r['model']==model and 'her2match' in r['datasets']]
        tested=[r for r in mine if any(e.get('rankable') and e['dataset']=='her2match' and e['split']=='test' for e in r['evaluations'])]
        active=[r for r in mine if r.get('scheduler_status') in {'RUNNING','PENDING'}]
        selected=(sorted(tested,key=lambda r:(bool(r.get('evaluation_complete')),r.get('status_as_of',''),r.get('parent_step',0)),reverse=True) or active or sorted(mine,key=lambda r:r.get('status_as_of',''),reverse=True) or [None])[0]
        state='evaluated' if tested else 'training' if any(r.get('scheduler_status')=='RUNNING' for r in mine) else 'queued' if any(r.get('scheduler_status')=='PENDING' for r in mine) else 'blocked' if any(r.get('scientific_blocker') for r in mine) else 'prepared'
        rows.append(dict(model=model,name=names[model],status=state,run_id=selected['id'] if selected else None,step=selected.get('last_step',0) if selected else 0,job=selected.get('training_job') if selected else None,gate_passed=bool(selected and selected.get('gate_passed')),reason=next((r['scientific_blocker'] for r in mine if r.get('scientific_blocker')),None),scientific_status=selected.get('state') if selected else 'Original specialist preparation pending',test_score_available=bool(tested),quality_warning=next((r['quality_warning'] for r in mine if r.get('quality_failure')),None),repair_jobs=[r['training_job'] for r in mine if r.get('training_job') and r.get('scheduler_status') in {'RUNNING','PENDING'} and (r.get('numerical_recovery') or 'repair' in r['id'] or 'rescue' in r['id'])]))
    result.update(rows=rows,counts={state:sum(row['status']==state for row in rows) for state in ['evaluated','training','queued','blocked','prepared']},registered_model_families=sum(row['run_id'] is not None for row in rows),quality_warning_count=sum(bool(row.get('quality_warning')) for row in rows))
    return result
