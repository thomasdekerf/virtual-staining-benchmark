"""Publish the released numerical HEMIT cohort with explicit native lineage.

This dataset is distinct from the HEMIT architecture and from ORION's reverse
translation task. No RGB colour, DAB, clinical or inferred per-field results are
created here. Frozen final reports supply scores; logs supply trajectories only.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
from pathlib import Path

MODELS = ('unet','pix2pix','nafnet','hemit','cut','uformer','bbdm','i2sb')
FINAL_MODELS = MODELS[:6]
CHANNELS = ('DAPI','CD3','panCK')
COUNTS = dict(train=3717,val=630,test=945)
MARKER = 'DAPI / CD3 / panCK'
HEX = re.compile('[0-9a-f]{64}')
PROTOCOL = 'Numerical mIF · uint8/255 · native center256 · full author split'

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(v): return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def require(ok,message):
    if not ok: raise ValueError(message)
def jsonl(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def verify_gate(gate,source_sha,native=False,model=None):
    require(gate.get('passed') is True and gate.get('source_sha256')==source_sha
        and gate.get('world_size')==4 and gate.get('validation_tiles_recomputed')==9
        and gate.get('full_width') is True and gate.get('resume') is True
        and gate.get('positive_learning_rate') is True and gate.get('actual_model_tensor_update') is True
        and gate.get('resumed_step',0)>gate.get('first_leg_step',0),
        'HEMIT requires an actual source-bound four-rank full-capacity update/resume gate')
    if native:
        require(gate.get('actual_learned_optimizer_resume') is True and gate.get('test_used') is False,
            'HEMIT continuation must resume real learned optimizer state without TEST')
        expected={'generator','discriminator'} if model in {'pix2pix','hemit'} else {'generator_and_nce','discriminator'} if model=='cut' else {'generator'}
        require(set(gate.get('native_learned_evidence',{}))==expected,
            'HEMIT continuation must bind every native learned branch')
        for state in gate.get('native_learned_evidence',{}).values():
            require(HEX.fullmatch(state.get('learned_parameters_sha256',''))
                and state.get('parameter_states',0)>0 and state.get('min_optimizer_step',0)>0
                and state.get('learning_rates') and all(finite(v) and v>0 for v in state['learning_rates']),
                'HEMIT continuation learned-state evidence is incomplete')

def merge_history(paths,axis):
    """Continuation logs include inherited history. Equal native positions dedupe.

    Epoch methods use their one-based history epoch, never the checkpoint's
    zero-based epoch. Unequal overlapping scalars are rejected, not overwritten.
    """
    points={}
    for path in paths:
        for row in jsonl(path):
            x=row.get('epoch') if axis=='epoch' else row.get('global_step',row.get('step'))
            require(finite(x),'HEMIT history lacks its native position')
            for group,prefix in [('train','train/'),('validation','val/')]:
                for key,value in row.get(group,{}).items():
                    if not finite(value) or key=='samples': continue
                    # Raw PCC totals retain their actual denominator in the tag.
                    name='mif_'+key if group=='validation' else key
                    series=points.setdefault(prefix+name,{})
                    require(x not in series or series[x]==value,'Conflicting HEMIT lineage measurement')
                    series[x]=value
    return {tag:[[x,values[x]] for x in sorted(values)] for tag,values in sorted(points.items())}

def merge_allocations(paths):
    segments={}
    for path in paths:
        for row in jsonl(path):
            key=tuple(row.get(k) for k in ('job_id','restart','start_unix','end_unix'))
            require(finite(row.get('seconds')) and row['seconds']>=0 and row.get('world_size')==4,
                'HEMIT allocation requires actual four-rank elapsed seconds')
            require(key not in segments or segments[key]==row,'Conflicting allocation segment')
            segments[key]=row
    seconds=sum(row['seconds'] for row in segments.values())
    return dict(training_hours=seconds/3600,gpu_hours=4*seconds/3600,
        allocation_segments=len(segments),definition='Recorded four-GPU allocation wall time across deduplicated parent and continuation segments; excludes technical gates and final held-out evaluation.')

def replay_patience(history,minimum,min_delta):
    best=math.inf;bad=0
    for row in history:
        position=row.get('epoch') if minimum['kind']=='epochs' else row.get('global_step',row.get('step'))
        value=row['validation']['lpips']
        require(finite(position) and finite(value),'Native patience replay needs actual finite validation history')
        if value<best-min_delta:best=value;bad=0
        elif position>minimum['horizon']:bad+=1
        else:bad=0
    return bad

def verify_final(report,completion,status,protocol,report_sha,adequacy,history,min_delta):
    require(completion.get('complete') is True and status.get('complete') is True
        and completion.get('final_report_sha256')==report_sha
        and completion.get('selected_checkpoint_sha256')==report.get('checkpoint_sha256')
        and HEX.fullmatch(report.get('checkpoint_sha256',''))
        and completion.get('source_sha256')==report.get('source_sha256')
        and completion.get('test_used_for_selection') is False,
        'HEMIT final report is not bound to completed selected-checkpoint evidence')
    require(adequacy.get('independently_replayed_native_adequacy') is True,
        'HEMIT native horizon and fresh validation patience were not independently replayed')
    minimum=protocol['native_schedule_continuation']['minimum']
    position=status.get('epoch') if minimum['kind']=='epochs' else status.get('step')
    # CUT does not serialize its bad-round counter in status.json. Replay all
    # actual inherited and continuation checks with the frozen min_delta.
    bad_checks=replay_patience(history,minimum,min_delta)
    require(position>=minimum['horizon'] and status.get('early_stop') is True
        and bad_checks>=minimum['patience'],
        'HEMIT cannot be final before its native horizon and fresh patience')
    require(report.get('full_split_counts')==dict(val=630,test=945)
        and set(report.get('full_splits',{}))=={'val','test'},'HEMIT requires both complete author splits')
    for split,record in report['full_splits'].items():
        require(record.get('n')==COUNTS[split]
            and record.get('manifest_sha256')==protocol['manifest_sha256'][split]
            and set(record.get('per_dataset',{}))=={'HEMIT'}
            and record['per_dataset']['HEMIT'].get('n')==COUNTS[split],
            'HEMIT evaluation changed the frozen author cohort')
        require(all(finite(record['per_dataset']['HEMIT']['metrics'].get(k)) for k in ('mae','psnr','ssim','lpips')),
            'HEMIT core numerical metrics must be finite')

def numerical_metrics(record):
    n=record['n'];raw=record['per_dataset']['HEMIT'];vals={'mif_'+k:raw['metrics'][k] for k in ('psnr','ssim','lpips','mae')}
    coverage={k:n for k in vals};channels={}
    for channel in CHANNELS:
        row=raw['fluorescence_channels'][channel]
        defined,total=row['pcc_defined_tiles'],row['pcc_total_tiles']
        require(isinstance(defined,int) and 0<=defined<=total==n,'Invalid HEMIT defined-tile PCC coverage')
        channels[channel]={k:row[k] for k in ('mae','psnr','ssim','pcc_mean_defined_tiles','pcc_defined_tiles','pcc_total_tiles')}
        channels[channel]['pcc_undefined_tiles']=total-defined
        for k in ('mae','psnr','ssim'):
            key=f'mif_{channel}_{k}';require(finite(row[k]),'Nonfinite numerical channel metric');vals[key]=row[k];coverage[key]=n
        key=f'mif_{channel}_pcc';coverage[key]=defined
        if finite(row['pcc_mean_defined_tiles']): vals[key]=row['pcc_mean_defined_tiles']
        else: require(defined==0,'Undefined PCC with nonzero defined count')
    return vals,coverage,channels

def metric_glossary():
    entries={}
    for key,label,direction,unit,description in [
        ('psnr','mIF PSNR','↑','dB','Tile-mean PSNR of normalized numerical DAPI/CD3/panCK channels.'),
        ('ssim','mIF box SSIM','↑','index','Tile-mean 11×11 box SSIM on numerical channels.'),
        ('lpips','mIF feature distance','↓','distance','Operational LPIPS feature distance on normalized DAPI/CD3/panCK stack; not a biological or clinical stain score.'),
        ('mae','mIF MAE','↓','normalized intensity','Tile-mean absolute error on released uint8/255 numerical channels. Storage-range scaling is not physical calibration.')]:
        entries['mif_'+key]=dict(label=label,direction=direction,unit=unit,description=description)
    for channel in CHANNELS:
        for key in ('mae','psnr','ssim'):
            base=entries['mif_'+key];entries[f'mif_{channel}_{key}']=base|dict(label=channel+' '+{'mae':'MAE','psnr':'PSNR','ssim':'SSIM'}[key],description=base['description']+' '+channel+' only.')
        entries[f'mif_{channel}_pcc']=dict(label=channel+' PCC (defined tiles)',direction='↑',unit='correlation',description='Mean per-tile Pearson correlation only where defined. Explicit defined/total counts accompany each result; undefined tiles are omitted, never replaced by zero.')
        entries[f'mif_pcc_sum_per_tile_{channel}']=dict(label=channel+' PCC sum / all tiles',direction='—',unit='total-tile mean',description='Training-log diagnostic: sum of defined per-tile Pearson correlations divided by TOTAL tiles. Undefined tiles contribute zero to this total-denominator reduction. This is not the conditional mean PCC over defined tiles; inspect the valid fraction.')
        entries[f'mif_pcc_valid_fraction_{channel}']=dict(label=channel+' PCC defined fraction',direction='—',unit='fraction of fields',description='Training-log fraction of all validation fields with defined per-tile Pearson correlation. Complements PCC sum / all tiles; it is measurement coverage, not a quality score.')
    return entries

def integrate(site,workspace,data,runs,register,fresh_run,source,write,changes,previous_evals):
    import yaml
    base=site/'.source-cache/hemit_dataset_20261125'
    if not base.exists(): return None
    evidence=workspace/'artifacts/new_datasets_20261125'
    report_path=evidence/'progress/20261126T085808Z_hemit_report.json';audit=read(report_path)
    inventory_path=workspace/'artifacts/run_update_20261126/site_evidence/hemit_staging.json'
    inventory=read(inventory_path)
    for row in inventory['files']:
        require(sha(site/'.source-cache'/row['path'])==row['sha256'],'Staged HEMIT bytes differ from captured evidence')
    source(inventory_path,'Read-only HEMIT source staging inventory')
    source(report_path,'Independently replayed HEMIT native horizons, fresh patience and selected-checkpoint evidence')
    by_model={row['model']:row for row in audit['model_rows']}
    names={m['id']:m['name'] for m in data['models']};rows=[]
    latest=read(workspace/'artifacts/run_update_20261126/site_evidence/fresh_campaign_snapshot.json')
    captured=latest['captured_utc'];queue={q['job']:q for q in latest.get('active_jobs',[])}
    for i,model in enumerate(MODELS):
        native=model in FINAL_MODELS;parent=base/'parent_runs'/model;directory=base/('native_runs' if native else 'parent_runs')/model
        cfg=yaml.safe_load((directory/'config.yml').read_text());protocol=cfg['paired_protocol'];training_source=read(directory/'training_source.json')
        require(protocol.get('task')=='he_to_mif' and protocol.get('target_space')=='mif'
            and protocol.get('target_channels')==list(CHANNELS) and protocol.get('split_counts')==COUNTS
            and protocol.get('world_size')==training_source.get('world_size')==4
            and training_source.get('model')==model and training_source.get('dataset')=='HEMIT'
            and HEX.fullmatch(training_source.get('source_sha256','')),'HEMIT numerical task/source identity changed')
        require('uint8/255' in protocol['normalization'] and 'no resizing' in protocol['sampling'],
            'HEMIT storage normalization and native sampling changed')
        source_sha=training_source['source_sha256'];parent_gate=base/'parent_technical_gates'/model/'smoke_receipt.json'
        verify_gate(read(parent_gate),source_sha)
        bindings=[source(parent_gate,'HEMIT native four-GPU parent update/resume proof')]
        if native:
            gate_path=base/'native_technical_gates'/model/'continuation_gate.json';gate=read(gate_path)
            verify_gate(gate,source_sha,True,model)
            require(gate['scientific_config_sha256']==training_source['scientific_config_sha256']
                and gate['migration_provenance_sha256']==sha(directory/'continuation_provenance.json'),
                'HEMIT continuation source/config/provenance binding changed')
            bindings.append(source(gate_path,'HEMIT native-horizon continuation gate'))
        for filename in ['config.yml','training_source.json','history.jsonl','allocation_time.jsonl','status.json']:
            bindings.append(source(directory/filename,'HEMIT '+filename))
        status=read(directory/'status.json');axis='epoch' if model in {'pix2pix','hemit'} else 'step'
        paths=[parent/'history.jsonl']+([directory/'history.jsonl'] if native else [])
        curves=merge_history(paths,axis);compute=merge_allocations([parent/'allocation_time.jsonl']+([directory/'allocation_time.jsonl'] if native else []))
        rid=f'hemit_dataset__{model}_seed42';run=register(runs.get(rid) or fresh_run(rid,model,'hemit_dataset',name=names[model]))
        run.update(task='he_to_mif',marker=MARKER,variant='HEMIT numerical mIF · native horizon · seed 42',
            target_space='mif',target_channels=list(CHANNELS),clinical_score_evaluation=False,seed=42,
            configuration=(directory/'config.yml').read_text(),crop_size=256,train_tiles=3717,
            source_sha256=source_sha,scientific_config_sha256=training_source['scientific_config_sha256'],
            config_sha256=sha(directory/'config.yml'),sources=bindings,evaluations=[],curves=curves,
            curve_counts={k:len(v) for k,v in curves.items()},curve_axis=axis,
            curve_protocol=dict(position_units='completed native epochs' if axis=='epoch' else 'optimizer updates',
                cohort='Full 630-field author VAL; normalized numerical mIF channels',inherited_parent_deduplicated=True,
                selected_final_evaluation_appended=False),
            last_step=status['step'],last_epoch=status.get('epoch'),training_complete=native,evaluation_complete=native,
            gate_passed=True,status_as_of=captured,live_status=status,training_hours=compute['training_hours'],
            gpu_hours=compute['gpu_hours'],compute_complete=native,time_definition=compute['definition'],
            allocation_segments=compute['allocation_segments'],selection=protocol['selection'],
            normalization=dict(method='released uint8 / 255',physical_intensity_calibration=False,
                stored_channels=['panCK','CD3','DAPI'],canonical_reindex=[2,1,0]),
            native_minimum=by_model[model]['native_minimum'],gallery=[],distributions={},
            availability=dict(prediction_gallery='No retained field predictions in this snapshot',
                per_tile_distributions='Unavailable: final reports contain means and channel PCC counts only'),
            notes='Released author-field split; no patient/WSI mapping is available. Numerical fluorescence channels use storage-range normalization, without physical intensity calibration.')
        if native:
            final_path=directory/'final_evaluation.json';final=read(final_path);completion=read(directory/'continuation_complete.json')
            for filename,item in by_model[model]['evidence'].items():
                require(sha(directory/filename)==item['sha256'],'HEMIT final native evidence changed since independent adequacy replay')
            verify_final(final,completion,status,protocol,sha(final_path),by_model[model],jsonl(directory/'history.jsonl'),cfg['validation']['early_stopping']['min_delta'])
            require(completion['scientific_config_sha256']==training_source['scientific_config_sha256'],'HEMIT final scientific recipe changed')
            binding=source(final_path,'HEMIT complete selected-checkpoint numerical VAL/TEST report')
            run['sources'].append(binding)
            for split,record in final['full_splits'].items():
                vals,coverage,channels=numerical_metrics(record)
                eid=hashlib.sha256(f'{rid}|{split}|{PROTOCOL}'.encode()).hexdigest()[:12]
                evaluation=dict(id=eid,dataset='hemit_dataset',marker=MARKER,split=split,n=COUNTS[split],
                    protocol=PROTOCOL,comparison_protocol=PROTOCOL,metrics=vals,coverage=coverage,
                    fluorescence_channels=channels,rankable=True,partial=False,
                    target_space='mif',target_channels=list(CHANNELS),crop_size=256,
                    checkpoint_sha256=final['checkpoint_sha256'],manifest_sha256=record['manifest_sha256'],
                    source_sha256=source_sha,metric_protocol=final['metric_protocol'],
                    evidence=dict(file=binding['file'],sha256=binding['sha256']),as_of=captured,
                    evaluation_scope='in_domain_frozen_author_field_split',clinical_score_evaluation=False)
                run['evaluations'].append(evaluation)
                changes.append(dict(run=rid,evaluation=eid,dataset='hemit_dataset',split=split,
                    new_evaluation=True,new_since_previous_snapshot=(rid,eid) not in previous_evals,source_sha256=binding['sha256']))
            # CUT/Uformer may validly retain a selected parent checkpoint. Only
            # the child completion receipt establishes final training adequacy.
            run.update(state='Evaluated',scheduler_status='Complete',checkpoint_sha256=final['checkpoint_sha256'],
                position=final['position'],selected_position_units='checkpoint epoch is zero-based; history epochs are completed one-based epochs' if axis=='epoch' else 'optimizer updates',
                selected_completed_epoch=final['position'].get('epoch',-1)+1 if axis=='epoch' else None,
                evaluation_seconds=final['evaluation_seconds'],evaluation_source=dict(file=binding['file'],sha256=binding['sha256']),
                native_adequacy_verified=True,parent_best_reused=completion['parent_best_reused'],
                stop_reason='Native horizon completed, followed by fresh validation patience at positive learning rates')
            quality={'cut':'Poor correspondence: selected TEST DAPI and panCK PCC are negative despite completed training.',
                'uformer':'Weak CD3 correspondence: inspect the defined-tile CD3 PCC alongside image-feature distance.'}.get(model)
            if quality:run.update(quality_note=quality,quality_warning=quality)
        else:
            job=f'1662934_{i}';q=queue.get(job,{})
            run.update(state='Training' if q.get('state')=='RUNNING' else 'Checkpointed / queued',
                scheduler_status=q.get('state','Not captured'),scheduler_reason=q.get('reason'),training_job=job,
                evaluation_complete=False,training_complete=False,compute_complete=False,
                stop_reason=None,notes=run['notes']+' Final native-horizon continuation and full selected-checkpoint evaluation are pending; no score is published.')
        csv_path=site/'dist/downloads'/f'{rid}_curves.csv'
        with csv_path.open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['tag','step_or_epoch','value'])
            writer.writerows([tag,x,y] for tag,points in curves.items() for x,y in points)
        rows.append(dict(id=rid,model=model,name=names[model],state=run['state'],training_complete=native,
            evaluated=native,step=run['last_step'],completed_epoch=run.get('last_epoch'),
            native_minimum=run['native_minimum'],quality_note=run.get('quality_note'),
            checkpoint_sha256=run.get('checkpoint_sha256'),curve_axis=axis))
    dataset=dict(id='hemit_dataset',name='HEMIT',task='he_to_mif',direction='H&E → numerical multiplex IF',
        tissue='Released registered tissue fields',markers=[MARKER],counts=COUNTS,
        target_space='mif',target_channels=list(CHANNELS),
        description='Paired H&E and numerical DAPI/CD3/panCK fluorescence channels from the released author-field splits.',
        notes='Native 256-pixel crops from 1024-pixel fields; fixed center crops for VAL/TEST, with no resizing. Released uint8/255 storage scaling is not physical fluorescence calibration. Author fields are separated; patient/WSI identities are unavailable.',
        prediction_gallery_available=False)
    data['datasets']=[d for d in data['datasets'] if d['id']!='hemit_dataset']+[dataset]
    data['metrics'].update(metric_glossary())
    campaign=dict(snapshot=captured,dataset='hemit_dataset',task='he_to_mif',model_families=8,
        completed_models=6,unfinished_models=2,full_split_counts=dict(val=630,test=945),runs=rows,
        target_channels=list(CHANNELS),normalization='Exact released uint8 /255 storage range; no physical calibration',
        interpretation='Numerical fluorescence channels; LPIPS is an operational feature distance. PCC is averaged over defined tiles with explicit coverage. CUT has poor correspondence; Uformer has weak CD3 correspondence.',
        download='data/hemit-dataset.json')
    data['hemit_campaign']=campaign;write(site/'dist/data/hemit-dataset.json',campaign)
    deferred=dict(snapshot=captured,dataset='PRECISE',state='user_deferred',active_objective=False,
        note='PRECISE was explicitly deferred by the user on 26 November 2026; its jobs are cancelled.')
    data['deferred_datasets']=[deferred];write(site/'dist/data/deferred-datasets.json',[deferred])
    # Use the same captured queue and node clock as the current capacity report.
    gpu=data.get('gpu_activity',{});jobs=gpu.setdefault('jobs',[])
    for i,model in [(6,'bbdm'),(7,'i2sb')]:
        job=f'1662934_{i}';q=queue.get(job)
        if q and not any(j['job']==job for j in jobs):
            jobs.append(dict(job=job,run_id=f'hemit_dataset__{model}_seed42',model=model,
                dataset='HEMIT',status=q['state'],gpus=4,step=runs[f'hemit_dataset__{model}_seed42']['last_step'],
                reason=q['reason'],kind='training'))
    running=[j for j in jobs if j['status']=='RUNNING']
    gpu.update(snapshot=captured,own_training_gpus=sum(j['gpus'] for j in running if j.get('kind')!='held-out evaluation'),
        own_running_training_jobs=sum(j.get('kind')!='held-out evaluation' for j in running),
        own_evaluation_gpus=sum(j['gpus'] for j in running if j.get('kind')=='held-out evaluation'),
        own_running_evaluation_jobs=sum(j.get('kind')=='held-out evaluation' for j in running))
    write(site/'dist/data/gpu-activity.json',gpu)
    return campaign
