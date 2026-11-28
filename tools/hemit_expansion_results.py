"""Expose every catalog decision and actual HEMIT submissions without scores.

The original final-evaluation importer remains the authority for scored rows.
This overlay only publishes fresh, hash-bound execution and numerical histories.
"""
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import re

from hemit_dataset_results import CHANNELS, COUNTS, MARKER, merge_history, read, require, sha


def bound(workspace, binding, path_key='path', hash_key='sha256'):
    path=workspace/binding[path_key]
    require(sha(path)==binding[hash_key], 'HEMIT expansion evidence bytes changed: '+str(path))
    return path


def queue_rows(snapshot):
    require(snapshot['queue']['returncode']==0, 'HEMIT publication needs a successful fresh queue query')
    rows={}
    for line in snapshot['queue']['stdout'].splitlines():
        fields=line.split('|')
        if len(fields)>=7:
            rows[fields[0]]=dict(job=fields[0],name=fields[1],state=fields[2],reason=fields[4],node=fields[6])
    return rows


def integrate(site,workspace,data,runs,register,fresh_run,source,write):
    root=workspace/'artifacts/hemit_expansion_20261127'
    audit_path=root/'review/full_panel_audit.json'
    pointer_path=root/'site_evidence/latest.json'
    if not audit_path.exists() or not pointer_path.exists():return None
    audit=read(audit_path);items=audit['items']
    decisions={r['model']:r for r in items}
    names={m['id']:m['name'] for m in data['models']}
    require(len(items)==len(decisions)==46 and set(decisions)==set(names), 'HEMIT audit must account for all46 catalog families exactly once')
    manifest_path=bound(workspace,read(pointer_path),'manifest')
    evidence=read(manifest_path)
    snapshot_path=bound(workspace,evidence,'progress_snapshot','progress_sha256')
    snapshot=read(snapshot_path);queue=queue_rows(snapshot)
    captured=snapshot['captured_utc']
    bindings=[source(audit_path,'All46 published-family HEMIT eligibility decisions'),
              source(manifest_path,'Read-only fresh HEMIT text metadata capture'),
              source(snapshot_path,'Current HEMIT scheduler and status snapshot')]
    staged={}
    for row in evidence['files']:
        path=bound(workspace,row,'local_path')
        staged[(row['campaign'],row['model'],row['name'])]=path
    state=read(workspace/'artifacts/new_datasets_20261125/campaign_state.json')
    submitted={}
    registry_path=root/'campaign_registry.json'
    if registry_path.exists():
        registry=read(registry_path)
        for entry in registry['campaigns']:
            plan_path=bound(workspace,entry,'campaign','campaign_sha256');plan=read(plan_path)
            receipt_path=bound(workspace,entry,'submission_receipt','submission_sha256');receipt=read(receipt_path)
            require(receipt.get('held_before_review') is True and receipt.get('per_task_gpus')==4,
                    'HEMIT submissions require held four-GPU review')
            jobs=receipt['task_job_ids']
            require(len(jobs)==len(plan['tasks']) and len(set(jobs))==len(jobs),'HEMIT submission/task cardinality changed')
            bindings.extend([source(plan_path,'Frozen HEMIT expansion campaign'),source(receipt_path,'Durable HEMIT expansion job receipt')])
            for task,job in zip(plan['tasks'],jobs):
                require(task['model'] not in submitted, 'Duplicate HEMIT expansion model')
                submitted[task['model']]=dict(task=task,job=job,campaign=Path(entry['remote_root']).name,plan=plan,prepared=plan_path.parent/'configs'/Path(task['config']).name)
    # Parent+native continuations remain one logical run, with original scores.
    baseline=set(('unet','pix2pix','nafnet','hemit','cut','uformer','bbdm','i2sb'))
    original_jobs={'bbdm':'1678822_6','i2sb':'1663216_7'}
    original_jobs.update(state.get('datasets',{}).get('HEMIT',{}).get('current_operational_job_ids',{}))
    rows=[];hemit_jobs=[]
    for model in names:
        decision=decisions[model]
        rid=f'hemit_dataset__{model}_seed42'
        run=runs.get(rid)
        task_record=submitted.get(model)
        reason=decision.get('reason',decision.get('eligibility_reason',''))
        eligible=decision.get('runnable_HEMIT',False) or bool(task_record)
        status='preparing' if eligible else 'integrating' if decision['classification']=='integration' else 'blocked'
        job=None;production=False
        if model in baseline:
            require(run is not None,'Original HEMIT model disappeared')
            final=bool(run.get('training_complete') and run.get('evaluation_complete'))
            status='evaluated' if final else 'queued'
            if not final:
                job=original_jobs.get(model)
                q=queue.get(job,{})
                status='training' if q.get('state')=='RUNNING' else 'queued'
                campaign='hemit_native_schedule_20261125_v1' if model=='i2sb' else 'hemit_benchmark_20261125_v1'
                production=True
        elif task_record:
            task=task_record['task'];job=task_record['job'];campaign=task_record['campaign'];q=queue.get(job,{})
            require(task.get('source_sha256',task_record['plan'].get('source_sha256')),'Expansion source fingerprint missing')
            import yaml
            prepared=task_record['prepared'];require(sha(prepared)==task['config_sha256'],'Prepared expansion configuration changed')
            config=yaml.safe_load(prepared.read_text());protocol=config['paired_protocol']
            require(protocol['target_space']=='mif' and protocol['target_channels']==list(CHANNELS) and protocol['split_counts']==COUNTS and protocol['world_size']==4,'Expansion changed the numerical HEMIT protocol')
            run=register(run or fresh_run(rid,model,'hemit_dataset',name=names[model]))
            run.update(task='he_to_mif',marker=MARKER,target_space='mif',target_channels=list(CHANNELS),
                       seed=42,variant='HEMIT numerical mIF · native published method · seed42',
                       crop_size=256,train_tiles=COUNTS['train'],training_job=job,
                       source_sha256=task.get('source_sha256',task_record['plan'].get('source_sha256')),
                       sources=list(bindings)+[source(prepared,'Frozen HEMIT published-method configuration')],configuration=prepared.read_text(),config_sha256=task['config_sha256'],
                       native_minimum=protocol['native_schedule_continuation']['minimum'],curve_axis='epoch' if task['trainer']=='virtual_staining.train_pyramid' else 'step',
                       normalization=copy.deepcopy(runs['hemit_dataset__unet_seed42']['normalization']),
                       curve_protocol=dict(position_units='completed native epochs' if task['trainer']=='virtual_staining.train_pyramid' else 'optimizer updates',
                           cohort='Full630-field author VAL; normalized numerical mIF channels',inherited_parent_deduplicated=True),
                       training_complete=False,evaluation_complete=False,
                       notes=reason,clinical_score_evaluation=False)
            status='checking' if q.get('state')=='RUNNING' else 'queued'
        if run is not None and status!='evaluated' and job:
            q=queue.get(job,{})
            files={name:path for (c,m,name),path in staged.items() if c==campaign and m==model}
            saved=read(files['status.json']) if 'status.json'in files else {}
            launch=read(files['launch_state.json']) if 'launch_state.json'in files else {}
            production=production or launch.get('phase') in {'production','production_training','training','held_out_evaluation','evaluation','complete'} or bool(saved)
            if q.get('state')=='RUNNING':status='training' if production else 'checking'
            if 'failure.json'in files and q.get('state')!='RUNNING':status='failed';reason=read(files['failure.json']).get('error','Native training check failed; diagnosis required')
            run.update(state={'training':'Training','checking':'Training readiness check','queued':'Queued','failed':'Failed / diagnosis pending'}[status],
                       scheduler_status=q.get('state','Not in active queue'),scheduler_reason=q.get('reason'),
                       training_job=job,status_as_of=captured,live_status=saved,training_complete=False,evaluation_complete=False)
            if saved:run['last_step']=saved.get('step',saved.get('global_step'));run['last_epoch']=saved.get('epoch')
            if 'config.yml'in files:
                run['configuration']=files['config.yml'].read_text();run['config_sha256']=sha(files['config.yml'])
                import yaml
                config=yaml.safe_load(run['configuration'])
                run['curve_axis']='epoch' if task_record and task_record['task'].get('trainer') in {'virtual_staining.train_pyramid','train_pyramid'} else run.get('curve_axis','step')
            if 'history.jsonl'in files:
                history=[files['history.jsonl']]
                parent=staged.get(('hemit_benchmark_20261125_v1',model,'history.jsonl'))
                if model=='i2sb' and parent:history.insert(0,parent)
                axis=run.get('curve_axis','step')
                run['curves']=merge_history(history,axis);run['curve_counts']={k:len(v) for k,v in run['curves'].items()}
                with (site/'dist/downloads'/f'{rid}_curves.csv').open('w',newline='') as handle:
                    writer=csv.writer(handle);writer.writerow(['tag','step_or_epoch','value'])
                    writer.writerows([tag,x,y] for tag,points in run['curves'].items() for x,y in points)
            run['sources']=run.get('sources',[])+[source(path,'Fresh HEMIT '+name) for name,path in files.items()]
            hemit_jobs.append(dict(job=job,run_id=rid,model=model,dataset='HEMIT',status=q.get('state','Not in active queue'),gpus=4,
                                   step=run.get('last_step'),reason=q.get('reason'),kind='training' if production else 'native update/resume check'))
        rows.append(dict(model=model,name=names[model],run_id=rid if run else None,status=status,reason=reason,
                         classification=decision['classification'],runnable_HEMIT=eligible,job=job,
                         step=run.get('last_step') if run else None,quality_note=run.get('quality_note') if run else None,
                         training_complete=status=='evaluated',evaluated=status=='evaluated'))
    counts=dict(Counter(row['status'] for row in rows))
    campaign=data['hemit_campaign']|dict(snapshot=captured,model_families=len(baseline|set(submitted)),catalog_families=46,
        completed_models=counts.get('evaluated',0),unfinished_models=sum(row['status']!='evaluated' and bool(row['run_id']) for row in rows),
        counts=counts,rows=rows,runs=[row for row in rows if row['run_id']],sources=bindings,
        download='data/hemit-dataset.json',note='All46 catalog methods are accounted for. Preparation and blockers are explicit; only actual submitted runs enter the benchmark. Native GPU checks precede production. Final scores require adequate training and complete VAL630/TEST945 evaluation.')
    data['hemit_campaign']=campaign;write(site/'dist/data/hemit-dataset.json',campaign)
    gpu=data.get('gpu_activity',{});jobs=[]
    for old in gpu.get('jobs',[]):
        if old.get('dataset')=='HEMIT':continue
        old=dict(old);q=queue.get(old['job'])
        if q:old.update(status=q['state'],reason=q['reason'],progress_snapshot_is_older=True)
        elif old.get('status') in {'RUNNING','PENDING','COMPLETING'}:old.update(status='Not in active queue',reason='Fresh scheduler does not list this job; no completion inferred')
        jobs.append(old)
    jobs.extend(hemit_jobs)
    running=[j for j in jobs if j['status']=='RUNNING']
    gpu.update(snapshot=captured,jobs=jobs,own_training_gpus=sum(j['gpus'] for j in running if j.get('kind')!='held-out evaluation'),
               own_running_training_jobs=sum(j.get('kind')!='held-out evaluation' for j in running),
               own_evaluation_gpus=sum(j['gpus'] for j in running if j.get('kind')=='held-out evaluation'),own_running_evaluation_jobs=0,
               hemit_running_gpus=sum(j['gpus'] for j in hemit_jobs if j['status']=='RUNNING'),
               completing_jobs_excluded_from_useful_training=True)
    limits_path=root/'scheduler/fresh_limits.json'
    if limits_path.exists():
        limits=read(limits_path);nodes=[]
        for command in limits['commands']:
            if command['command'][:3]!=['scontrol','show','nodes']:continue
            require(command['returncode']==0,'HEMIT GPU capacity query failed')
            for line in command['stdout'].splitlines():
                fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
                if not fields.get('NodeName','').startswith('acc'):continue
                total=re.search(r'(?:^|,)gres/gpu=(\d+)',fields.get('CfgTRES',''))
                allocated=re.search(r'(?:^|,)gres/gpu=(\d+)',fields.get('AllocTRES',''))
                total=int(total.group(1)) if total else 0;allocated=int(allocated.group(1)) if allocated else 0
                usable=not any(tag in fields['State'] for tag in ('DOWN','DRAIN','NOT_RESPONDING','FAIL','RESERVED'))
                nodes.append(dict(name=fields['NodeName'],state=fields['State'],gpus=total,allocated_gpus=allocated,free_gpus=max(0,total-allocated) if usable else 0,usable=usable))
        require(len(nodes)==22,'HEMIT needs every H200 node in the capacity snapshot')
        gpu['partition']=dict(name='zen4_h200',physical_gpus=sum(n['gpus'] for n in nodes),
            schedulable_gpus=sum(n['gpus'] if n['usable'] else n['allocated_gpus'] for n in nodes),
            allocated_gpus=sum(n['allocated_gpus'] for n in nodes),free_schedulable_gpus=sum(n['free_gpus'] for n in nodes),
            allocated_on_draining_nodes=sum(n['allocated_gpus'] for n in nodes if not n['usable']),
            largest_free_gpus_per_node=max(n['free_gpus'] for n in nodes),nodes=nodes,snapshot=limits['captured_utc'])
        gpu['sources']=gpu.get('sources',[])+[source(limits_path,'Current H200 partition, user association and normal QOS limits')]
    gpu['limits'].update(new_campaign_artificial_concurrency_cap=None,hemit_additional_submitted_models=len(submitted),hemit_catalog_methods_accounted=46,
                         memory_per_cpu_mb=7712,user_association_finite_gpu_limit_reported=False,normal_qos_finite_gpu_limit_reported=False,
                         inherited_project_limits_fully_visible=False,remaining_credit_balance_verified=False)
    data['gpu_activity']=gpu;write(site/'dist/data/gpu-activity.json',gpu)
    return campaign
