"""Refresh the public snapshot from locally staged, source-bound production evidence.

The original workbook reconciliation remains intact. New evidence is reconciled
separately, and an old scored value cannot be silently replaced by a different
checkpoint or metric value. No training, selection or evaluation is performed.
"""
from __future__ import annotations
import argparse,csv,copy,hashlib,json,math,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import yaml
from snapshot_evidence import current_cluster_snapshot,current_campaign_snapshot,current_followup_snapshot
from protocol_compatibility import attach_reviewed_protocols
from followup_results import apply_live_execution,new_research_validation,verify_completed_integration,verify_native_gate

NAMES={'swinir':'SwinIR','hemit':'HEMIT','implicitstainer':'ImplicitStainer','miphei_vit':'MiPHEI-ViT','bbdm':'BBDM','eqm':'EqM','nvg':'NVG','lrdm':'LRDM','itarflow':'Conditional iTARFlow','asp':'ASP','histdist':'HistDiST','pd_unist':'PD-UNIST','unet':'U-Net','pix2pix':'Pix2Pix','pyramid':'Pyramid Pix2Pix','pspstain':'PSPStain','cut':'CUT','cyclegan':'CycleGAN','nafnet':'NAFNet','restormer':'Restormer','cssp2p':'CSSP2P-GAN'}
DETECTORS={'stardist','stardist2x','hovernet','cellpose_sam'}

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
def metrics(obj):return {('delta_e76' if k=='delta_e' else k):v for k,v in (obj or {}).items() if finite(v) and k not in {'samples','n','count'}}
def clean(obj):
    if isinstance(obj,dict):return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,list):return [clean(v) for v in obj]
    if isinstance(obj,str):return re.sub(r'/(?:sofia|user|mnt|media|home|Users)/[^\s\"\',;]+','<local path>',obj)
    if isinstance(obj,float) and not math.isfinite(obj):return None
    return obj

def write(path,obj):Path(path).write_text(json.dumps(clean(obj),separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
def evaluation_id(rid,dataset,split,protocol,n):return hashlib.sha256(f'{rid}|{dataset}|{split}|{protocol}|{n}'.encode()).hexdigest()[:12]
def fresh_run(rid,model,dataset,**kw):return dict(id=rid,model=model,name=NAMES.get(model,model),variant='Production · seed 42',datasets=[dataset],task='ihc' if dataset=='her2match' else 'unstained',marker='HER2' if dataset=='her2match' else 'H&E',state='Registered',seed=42,evaluations=[],sources=[],curves={},gallery=[],distributions={},curve_counts={},curve_axis='step') | kw

def write_csv(path,header,rows):
    ending='\r\n' if path.exists() and b'\r\n' in path.read_bytes() else '\n'
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle,lineterminator=ending);writer.writerow(header);writer.writerows(rows)

def has_training(run):
    state=run.get('live_status') or {}
    return bool(any(e.get('rankable') for e in run['evaluations']) or sum(run.get('curve_counts',{}).values()) or (run.get('last_step') or state.get('step') or state.get('global_step') or 0)>0)

def refresh(site,workspace):
    out=site/'dist';cache=site/'.source-cache';baseline_path=cache/'published_baseline_20261119.json'
    current=read(out/'data/index.json');assert current
    if not baseline_path.exists():write(baseline_path,current)
    baseline=read(baseline_path)
    baseline_pairs={(r['id'],e['id']) for r in baseline['runs'] for e in r['evaluations'] if e.get('rankable')}
    data=copy.deepcopy(current);runs={r['id']:read(out/'data/runs'/f'{r["id"]}.json') for r in current['runs']}
    published=json.loads(subprocess.check_output(['git','-C',str(site),'show','HEAD:dist/data/index.json']))
    previous_evals={(r['id'],e['id']):copy.deepcopy(e) for r in published['runs'] for e in r['evaluations'] if e.get('rankable')}
    models={m['id']:m for m in data['models']};snapshot_path,snapshot=current_cluster_snapshot(workspace)
    captured=snapshot.get('captured_utc') or datetime.now(timezone.utc).isoformat()
    campaign_path,campaign_snapshot=current_campaign_snapshot(workspace)
    followup_path,followup_snapshot=current_followup_snapshot(workspace)
    followup_records=followup_snapshot.get('records',{})
    captured=max(captured,followup_snapshot.get('captured_utc',captured))
    sources=[];changes=[];touched=set()
    def source(path,label):
        p=Path(path);row={'file':str(p.relative_to(workspace)),'sha256':sha(p),'label':label,'observed_utc':captured}
        sources.append(row);return row
    def register(run):
        runs[run['id']]=run
        mid=run['model']
        if mid not in models:models[mid]={'id':mid,'name':NAMES.get(mid,mid),'priority':None,'family':'Recorded production / research model','notes':'Evidence-bound addition to the current snapshot. Read its task-specific protocol.','source':{'type':'verified result refresh'}}
        return run
    def status(run,directory):
        p=directory/'status.json';s=read(p,{})
        if not s:return
        source(p,'Current saved training status')
        run['live_status']=s;run['last_step']=s.get('step',s.get('global_step'));run['status_as_of']=captured
        run['training_complete']=bool(s.get('complete') or s.get('state')=='complete');run['max_steps']=s.get('max_steps')
        run['state']='Training completed / evaluation pending' if s.get('complete') or s.get('state')=='complete' else 'Paused for continuation' if s.get('budget_pause') or s.get('state')=='paused' else 'Training snapshot'
        run['scheduler_status']=run['state']
        run['stop_reason']='Validation plateau' if s.get('early_stop') else 'Configured schedule completed' if s.get('complete') else None
        cfg=directory/'config.yml'
        if cfg.exists():
            run['configuration']=cfg.read_text();source(cfg,'Frozen run configuration')
            c=yaml.safe_load(cfg.read_text()) or {};run['crop_size']=c.get('data',{}).get('crop_size',256)
        run['evaluations']=[e for e in run['evaluations'] if e['split']!='monitor']
        vals=metrics(s.get('metrics'));n=s.get('metrics',{}).get('samples')
        if vals and 'psnr' in vals:
            proto=f'Latest training validation · {run.get("crop_size",256)}px · '+(f'n={n:,}' if n else 'tile count not recorded')
            run['evaluations'].append(dict(id=evaluation_id(run['id'],run['datasets'][0],'monitor',proto,n),dataset=run['datasets'][0],marker=run['marker'],split='monitor',n=n,protocol=proto,metrics=vals,rankable=False,evidence={'file':str(p.relative_to(workspace)),'sha256':sha(p)},position=run['last_step'],as_of=captured))
    def history(run,directory,force=False):
        p=directory/'history.jsonl'
        if not p.exists():return
        # New histories use their recorded global step; existing dense scalar CSVs
        # remain intact unless the raw history demonstrably extends the same axis.
        parsed={}
        for line in p.read_text().splitlines():
            try:r=json.loads(line)
            except ValueError:continue
            x=r.get('global_step',r.get('step',r.get('epoch')))
            if not finite(x):continue
            for group,prefix in [('train','train/'),('validation','val/')]:
                for key,value in (r.get(group) or {}).items():
                    if finite(value) and key!='samples':parsed.setdefault(prefix+key,{})[x]=value
        if not parsed:return
        source(p,'Recorded training / validation history')
        full_file=out/'downloads'/f'{run["id"]}_curves.csv';previous={}
        if full_file.exists() and not force:
            for row in csv.DictReader(full_file.open()):previous.setdefault(row['tag'],{})[float(row['step_or_epoch'])]=float(row['value'])
        if previous and not force:
            overlap=sum(len(set(previous.get(tag,{}))&set(points)) for tag,points in parsed.items() if tag.startswith('val/'))
            if overlap==0:return
        for tag,points in parsed.items():
            target=previous.setdefault(tag,{})
            for x,y in points.items():target.setdefault(x,y)
        run['curves']={tag:[[x,points[x]] for x in sorted(points)] for tag,points in previous.items()}
        run['curve_counts']={k:len(v) for k,v in run['curves'].items()}
        write_csv(full_file,['tag','step_or_epoch','value'],([tag,x,y] for tag,points in run['curves'].items() for x,y in points))
        run['curves']={k:(p if len(p)<=700 else [p[i] for i in sorted(set(np.linspace(0,len(p)-1,700,dtype=int)))]) for k,p in run['curves'].items()}
        run['curve_snapshot_utc']=captured
    def ingest(run,path,dataset,kind):
        obj=read(path,{});full=obj.get('full_splits') or {}
        if not full:return
        complete=(obj.get('benchmark_profile_complete') is True and obj.get('production_result') is True and obj.get('core_metrics_complete') is True) if kind=='unstained' else (kind=='her2match' or obj.get('all_literature_metrics_complete',True))
        if kind=='unstained' and not complete:return
        if kind in {'her2match','coverage_her2match'}:
            assert obj.get('evaluation_scope')==('in_domain_HER2Match' if kind=='her2match' else 'in_domain_frozen_split') and set(full)=={'val','test'}
            assert {k:v['n'] for k,v in full.items()}=={'val':3582,'test':5980}
        assert obj.get('checkpoint_sha256'),f'Unbound checkpoint: {path}'
        receipt=source(path,'Complete selected-checkpoint evaluation')
        before={e['id']:copy.deepcopy(e) for e in run['evaluations'] if e.get('rankable')}
        for split,record in full.items():
            if kind in {'her2match','coverage_her2match','coverage_breast'}:
                v=record['per_dataset']['HER2Match' if dataset=='her2match' else dataset];vals=metrics(v['metrics']);dab=v['dab'];vals.update(dab_od_mae=dab['mean_absolute_error'],dab_correlation=dab['correlation'],dab_pred_mean=dab['mean_pred'],dab_target_mean=dab['mean_target']);coverage={k:record['n'] for k in vals};protocol='FP32 · 0.46 µm/px · center256 · full HER2Match split' if dataset=='her2match' else f'Box SSIM · 256px · n={record["n"]:,}';extra={'dab_samples':dab['samples']}
            else:
                if kind=='unstained':assert set(record.get('learned_nuclei',{}))==DETECTORS
                vals=metrics(record['metrics']);coverage={}
                for group in ['image_metrics','sampled_stain_and_nuclei']:
                    for key,value in record.get('extended_metrics',{}).get(group,{}).items():
                        if isinstance(value,dict):
                            coverage[key]=value.get('valid_tiles')
                            if finite(value.get('mean')):vals[key]=value['mean']
                for detector,det in record.get('learned_nuclei',{}).items():
                    for key,value in det.get('metrics',{}).items():
                        if isinstance(value,dict):
                            coverage[detector+'_'+key]=value.get('valid_tiles')
                            if finite(value.get('mean')):vals[detector+'_'+key]=value['mean']
                    for key,value in det.items():
                        if finite(value) and key not in {'seconds','sample_count','normalization_fallback_calls'}:vals[detector+'_'+key]=value
                vals.update(metrics({k:v for k,v in record.get('distribution',{}).items() if k in {'fid_clean','kid_clean'}}));protocol=f'Box SSIM · 256px · n={record["n"]:,}' if kind=='orion' else 'Full production evaluation';extra={'nuclei_protocol':{k:{field:v.get(field) for field in ['protocol','detector_adapter_sha256','checkpoint_sha256','normalization','sample_count']} for k,v in record.get('learned_nuclei',{}).items()}} if kind=='unstained' else {}
            eid=evaluation_id(run['id'],dataset,split,protocol,record['n'])
            e=dict(id=eid,dataset=dataset,marker=run['marker'],split=split,n=record['n'],protocol=protocol,metrics=vals,rankable=True,partial=False,coverage=coverage,group_macro=record.get('group_macro_bootstrap'),bootstrap_unit=record.get('bootstrap_unit'),manifest_sha256=record['manifest_sha256'],checkpoint_sha256=obj['checkpoint_sha256'],metric_protocol=obj.get('metric_protocol',obj.get('protocol')),evidence={'file':receipt['file'],'sha256':receipt['sha256']},as_of=captured,**extra)
            if eid in before:
                old=before[eid];assert old.get('checkpoint_sha256') in [None,e['checkpoint_sha256']],f'Changed old selected checkpoint {run["id"]}'
                for key,value in old['metrics'].items():assert key in vals and abs(vals[key]-value)<1e-9,f'Changed original scored metric {run["id"]}: {key}'
            run['evaluations']=[x for x in run['evaluations'] if x['id']!=eid]+[e]
            changes.append(dict(run=run['id'],evaluation=eid,dataset=dataset,split=split,new_evaluation=(run['id'],eid) not in baseline_pairs,new_since_previous_snapshot=(run['id'],eid) not in previous_evals,source_sha256=receipt['sha256']))
        run.update(state='Evaluated',scheduler_status='Complete',checkpoint_sha256=obj['checkpoint_sha256'],selection=obj.get('selection'),position=obj.get('position'),evaluation_source={'file':receipt['file'],'sha256':receipt['sha256']},remaining_metrics=obj.get('remaining_metrics',{}))
        for key in ['training_hours','gpu_hours','source_sha256','evaluation_seconds','time_definition']:
            if obj.get(key) is not None:run[key]=obj[key]
        binding={'file':receipt['file'],'sha256':receipt['sha256']}
        if binding not in run['sources']:run['sources'].append(binding)
    # Current original-production status and immutable complete benchmark files.
    for dataset in ['orion','finland','skin']:
        for directory in sorted((cache/dataset).glob('*')):
            rid=directory.name if dataset=='orion' else dataset+'__'+directory.name
            if rid not in runs or not directory.is_dir():continue
            run=runs[rid];status(run,directory);history(run,directory)
            if dataset!='orion':
                p=directory/'evaluation_v1/complete.json'
                if p.exists():ingest(run,p,dataset,'unstained')
            elif (directory/'final_evaluation.json').exists():
                ingest(run,directory/'final_evaluation.json',dataset,'orion')
            touched.add(rid)
    if not any(d['id']=='her2match' for d in data['datasets']):
        data['datasets'].append(dict(id='her2match',name='HER2Match',task='ihc',direction='H&E → HER2 IHC',tissue='Human breast · HER2',markers=['HER2'],counts=dict(train=11610,val=3582,test=5980),cases=dict(train=6,val=6,test=5),description='21,172 registered H&E/HER2 pairs with the original author case-disjoint train, validation and test split.',notes='Resampled from 0.25 to 0.46 µm/pixel; 256-pixel crops. Test results reload the checkpoint selected by validation LPIPS. Clinical HER2 grading is not evaluated.',source='https://zenodo.org/records/15797050'))
    for directory in sorted((cache/'her2match').glob('*')):
        if not directory.is_dir() or not (directory/'config.yml').exists():continue
        model=directory.name;rid='her2match__'+model+'_seed42';run=register(runs.get(rid) or fresh_run(rid,model,'her2match',train_tiles=11610,variant='HER2Match author cases · 0.46 µm/px · seed 42'))
        status(run,directory);history(run,directory,force=True)
        if (directory/'final_evaluation.json').exists():ingest(run,directory/'final_evaluation.json','her2match','her2match')
        touched.add(rid)
    for name,dataset,model in [('skin_asp','skin','asp'),('finland_histdist','finland','histdist'),('finland_pd_unist','finland','pd_unist')]:
        directory=cache/'recovery'/name
        if not (directory/'status.json').exists():continue
        rid=dataset+'__'+model+'_recovery_20261118';run=register(runs.get(rid) or fresh_run(rid,model,dataset,variant='Numerical recovery continuation · seed 42',parent_run=dataset+'__'+model+'_seed42',compute_parent_run=dataset+'__'+model+'_seed42',notes='Original model/optimizer restored; parent results and failure evidence retained separately. Precision/loss amendment recorded in configuration. Continuation allocation is not total training compute.'))
        status(run,directory);history(run,directory,force=True)
        if (directory/'evaluation_v1/complete.json').exists():ingest(run,directory/'evaluation_v1/complete.json',dataset,'unstained')
        touched.add(rid)
    # Research evaluation protocols stay separate from the full benchmark cohort.
    for directory in sorted((cache/'research/generative').glob('*')):
        if not directory.is_dir() or directory.name not in {'eqm','nvg','lrdm'}:continue
        model=directory.name;rid='finland__research_'+model+'_20261017';run=register(runs.get(rid) or fresh_run(rid,model,'finland',variant='Generative architecture research · native 256px',notes='Research validation protocol: curves use 256 monitoring tiles; the separate selected-checkpoint full validation uses 5,979 tiles. No full test report or selected checkpoint digest is available, so these results are unranked.'))
        status(run,directory)
        run['notes']='Research validation protocol: trajectories use the 256-tile selection monitor; a separate selected-checkpoint report measures all 5,979 validation tiles. No full test report or selected checkpoint digest is available, so results remain unranked.'
        p=directory/'full_validation.json';o=read(p,{})
        if o:
            source(p,'Full research validation result')
            # The report is parsed below only when explicitly a selected-checkpoint full validation.
            assert o.get('test_used') is False and o.get('n')==5979
            proto='Research validation · source-only single draw · checkpoint digest not recorded'
            vals={k:o[k] for k in ['psnr','ssim','mae','delta_e76','lpips'] if finite(o.get(k))}
            run['evaluations']=[e for e in run['evaluations'] if e.get('protocol')!=proto]
            run['evaluations'].append(dict(id=evaluation_id(rid,'finland','val',proto,5979),dataset='finland',marker='H&E',split='val',n=5979,protocol=proto,metrics=vals,rankable=False,position=o.get('step'),evidence={'file':str(p.relative_to(workspace)),'sha256':sha(p)},as_of=captured))
            run['state']='Training completed / research validation available'
            run['position']={'global_step':o.get('step'),'stage':o.get('stage')}
            vp=directory/'validation.jsonl'
            if vp.exists():
                source(vp,'Research stage-specific validation trajectory')
                curves={};monitor_ns=set()
                for line in vp.read_text().splitlines():
                    try:entry=json.loads(line)
                    except ValueError:continue
                    if entry.get('stage')!=o.get('stage') or entry.get('n')==o['n']:continue
                    monitor_ns.add(entry.get('n'))
                    for key in ['psnr','ssim','mae','delta_e76','lpips']:
                        if finite(entry.get(key)):curves.setdefault('val/'+key,[]).append([entry['step'],entry[key]])
                assert all(all(points[i][0]<points[i+1][0] for i in range(len(points)-1)) for points in curves.values()), 'Research monitoring history must preserve one monotonic inference scope'
                run['curves']=curves;run['curve_counts']={k:len(v) for k,v in curves.items()}
                assert len(monitor_ns)==1, 'Research trajectory mixes monitor sample scopes'
                run['curve_protocol']={'split':'training selection monitor','n':next(iter(monitor_ns)),'stage':o['stage'],'separate_full_validation_n':o['n']}
                write_csv(out/'downloads'/f'{rid}_curves.csv',['tag','step_or_epoch','value'],([tag,x,y] for tag,points in curves.items() for x,y in points))
        touched.add(rid)
    directory=cache/'research/itarflow'
    if (directory/'status.json').exists():
        rid='finland__research_itarflow768_recovery_20261118'
        run=register(runs.get(rid) or fresh_run(rid,'itarflow','finland',variant='768-wide direct flow · guarded continuation',notes='Training complete; original disjoint 128-crop selection monitor is not the full held-out evaluation. Full evaluation remains pending. Parent training compute is not included in continuation time.'))
        status(run,directory)
        selection=read(directory/'selected_checkpoint.json',{})
        if selection:
            source(directory/'selected_checkpoint.json','Selected iTARFlow continuation checkpoint')
            run['position']={'global_step':selection['step']};run['checkpoint_sha256']=selection.get('checkpoint_sha256')
        mp=directory/'monitor_history.json';monitor=read(mp,[])
        if monitor:
            source(mp,'Original disjoint 128-image iTARFlow monitor history')
            points=[[v['step'],v['lpips']] for v in monitor]
            run['curves']={'val/lpips':points};run['curve_counts']={'val/lpips':len(points)}
            write_csv(out/'downloads'/f'{rid}_curves.csv',['tag','step_or_epoch','value'],(['val/lpips',x,y] for x,y in points))
            proto='Disjoint training selection monitor · 128 crops'
            best=selection.get('lpips')
            if finite(best):run['evaluations']=[dict(id=evaluation_id(rid,'finland','monitor',proto,128),dataset='finland',marker='H&E',split='monitor',n=128,protocol=proto,metrics={'lpips':best},rankable=False,position=selection['step'],checkpoint_sha256=selection.get('checkpoint_sha256'),evidence={'file':str(mp.relative_to(workspace)),'sha256':sha(mp)},as_of=captured)]
        touched.add(rid)
        completed=cache/'research/itarflow_evaluation/complete.json'
        report=cache/'research/itarflow_evaluation/evaluation_v1/direct_rgb768/zeros/summary.json'
        if completed.exists():
            receipt=read(completed);summary=read(report)
            assert receipt['schema']=='scaled_itarflow_recovery_VAL512_complete_v1' and receipt['n']==512 and receipt['split']=='val' and receipt['test_used'] is False
            assert receipt['summary_sha256']==sha(report) and receipt['selected_checkpoint_sha256']==run['checkpoint_sha256']==summary['core']['provenance']['checkpoint_sha256']
            assert receipt['selected_step']==selection['step'] and receipt['readout']=='zeros' and receipt['denoise_steps']==10
            source(completed,'Completed original 512-field study validation');binding=source(report,'Selected-checkpoint 512-field study validation report')
            vals=metrics(summary['core']['candidate']['tile_mean']);vals['dists']=summary['dists']['candidate_minus_parent']['candidate_tile_mean'];coverage={k:512 for k in vals}
            assert summary['core']['candidate']['n_tiles']==512 and set(summary['nuclei'])==DETECTORS
            for detector,record in summary['nuclei'].items():
                assert record['n_tiles']==512
                for key,value in record['candidate'].items():
                    if isinstance(value,dict) and finite(value.get('mean')):
                        vals[detector+'_'+key]=value['mean'];coverage[detector+'_'+key]=value['valid_tiles']
            proto='Study validation · disjoint 512 native fields · zero prior · 10 steps'
            run['evaluations'].append(dict(id=evaluation_id(rid,'finland','val',proto,512),dataset='finland',marker='H&E',split='val',n=512,protocol=proto,metrics=vals,rankable=False,coverage=coverage,checkpoint_sha256=receipt['selected_checkpoint_sha256'],reported_indices_sha256=receipt['reported_indices_sha256'],position={'global_step':receipt['selected_step']},evidence={'file':binding['file'],'sha256':binding['sha256']},quality_success=summary['quality_success'],as_of=captured))
            run['state']='Training complete / study validation evaluated';run['notes']='Selected 25k recovery checkpoint evaluated on the original disjoint 512-field validation study. This is separate from the 128-field selection monitor and the native 5,979-field validation / 5,179-field test benchmark. The original quality-success criterion was not met.'
    for child in runs.values():
        if child.get('parent_run') in runs:
            parent=runs[child['parent_run']]
            parent['state']='Original run stopped / separate continuation available'
            parent['scheduler_status']='Separate recovery continuation available'
            parent['continuation_run']=child['id']
    plan_path=workspace/'artifacts/coverage_balance_20261119/prepared/campaign.json'
    plan=read(plan_path,{})
    if plan:
        source(plan_path,'Focused coverage-balancing campaign submission')
        for task in plan['tasks']:
            if not task.get('job_id'):continue
            dataset='her2match' if task['dataset']=='HER2Match' else task['dataset']
            rid='balance__'+task['id'];model=task['model']
            run=register(runs.get(rid) or fresh_run(rid,model,dataset,task='ihc' if dataset in {'BCI','MIST','IHC4BC','her2match'} else 'unstained',marker=task['marker'],variant='Coverage balancing · '+task['selected_marker_scope']+' · seed 42',notes='Accepted production task; real-data resume gate and complete selected-checkpoint evaluation required before reporting scores.'))
            run['state']=task.get('state','Queued');run['training_job']=task['job_id'];run['source_sha256']=task.get('source_sha256');run['config_sha256']=task.get('config_sha256');run['status_as_of']=captured
            directory=cache/'coverage_balance'/task['id']
            balance_snapshot=snapshot.get('coverage_balance',{})
            live=balance_snapshot.get('runs',{}).get(task['id'],balance_snapshot.get(task['id'],{}))
            smoke_record=balance_snapshot.get('smokes',{}).get(task['id'],live.get('smoke',{}))
            smoke=smoke_record.get('receipts',{}).get('smoke_receipt.json',smoke_record.get('receipt',{})).get('data',{})
            campaign_row=next((row for row in campaign_snapshot.get('tasks',[]) if row['task']['id']==task['id']),{})
            if campaign_row:smoke=campaign_row.get('smoke_receipt') or smoke
            run['gate_passed']=bool(smoke.get('passed') and smoke.get('world_size')==4 and smoke.get('validation_tiles_recomputed')==9 and smoke.get('resumed_step',0)>smoke.get('first_leg_step',0) and smoke.get('source_sha256')==task['source_sha256'])
            run['last_step']=0;status(run,directory);history(run,directory,force=True)
            run['registered_only']=not bool(run.get('last_step'))
            run['evaluation_complete']=False
            launch=read(directory/'launch_state.json',{})
            if launch:
                source(directory/'launch_state.json','Current coverage campaign execution phase')
                run['execution_phase']=launch.get('phase');run['training_job']=launch.get('job_id',run['training_job'])
            final=directory/('evaluation_v1/complete.json' if task['evaluation_kind']=='unstained_full' else 'final_evaluation.json')
            if final.exists():
                assert run['gate_passed'],f'Native resume gate required: {task["id"]}'
                result=read(final);assert result.get('checkpoint_sha256') and set(result.get('full_splits',{}))=={'val','test'}
                if task['evaluation_kind']!='unstained_full':
                    assert result['source_sha256']==task['source_sha256']
                    cfg=yaml.safe_load((directory/'config.yml').read_text());expected=cfg['breast_protocol']['manifest_sha256']
                    assert all(record['manifest_sha256']==expected[split] for split,record in result['full_splits'].items())
                    dataset_record=next(d for d in data['datasets'] if d['id']==dataset)
                    expected_counts={split:(dataset_record['counts'][split] if dataset=='her2match' else dataset_record['counts'][split][task['marker']]) for split in ('val','test')}
                    assert result['full_split_counts']==expected_counts
                    assert all(record['n']==expected_counts[split] and set(record['per_dataset'])=={task['dataset']} and record['per_dataset'][task['dataset']]['n']==expected_counts[split] for split,record in result['full_splits'].items()), 'Incomplete coverage evaluation cohort'
                ingest(run,final,dataset,'unstained' if task['evaluation_kind']=='unstained_full' else 'coverage_her2match' if dataset=='her2match' else 'coverage_breast')
                run['notes']='Native four-GPU pause/resume gate and complete selected-checkpoint validation/test evaluation verified. HER2 additions cover this marker only; test never selected weights.'
                run['evaluation_complete']=True
            elif live.get('files',{}).get('failure.json'):
                run['state']='Failed / recovery required';run['scheduler_status']=run['state']
            elif not run.get('training_complete'):
                for line in snapshot.get('queue',{}).get('stdout','').splitlines():
                    parts=[part.strip() for part in line.split('|')]
                    if len(parts)>2 and parts[0]==run['training_job']:
                        run['scheduler_status']=parts[2]
                        if parts[2]=='RUNNING':run['state']='Training'
                        elif parts[2]=='PENDING':run['state']='Checkpointed / queued' if run.get('last_step') else 'Queued'
            if snapshot_path and live:
                receipt=source(snapshot_path,'Dated coverage campaign gate and execution audit')
                run['gate_evidence']={'file':receipt['file'],'sha256':receipt['sha256']}
            if finite(run.get('training_hours')):
                run['compute_complete']=False
                run['time_definition']='Sum of recorded completed allocation segments. Interrupted allocations may be absent after the original array requeue fault; this is a lower bound, not total training time.'
            if campaign_row:
                saved=campaign_row.get('status') or {}
                run['last_step']=saved.get('step',saved.get('global_step',run.get('last_step')))
                run['training_complete']=bool(saved.get('complete'));run['registered_only']=not bool(run.get('last_step'))
                run['status_as_of']=campaign_snapshot['snapshot_utc']
                migration=campaign_snapshot.get('operational_migration',{})
                if migration and task['index']>=12:
                    run['original_training_job']=task['job_id'];run['training_job']=str(migration['replacement_array'])+'_'+str(task['index'])
                    if not run.get('evaluation_complete'):
                        scheduler=None
                        for line in campaign_snapshot.get('squeue',{}).get('stdout','').splitlines():
                            fields=[v.strip() for v in line.split('|')] if '|' in line else line.split()
                            if len(fields)>2 and fields[0]==run['training_job']:scheduler=fields[2]
                        run['scheduler_status']=scheduler or ('Training complete / evaluation pending' if saved.get('complete') else 'Checkpointed; scheduler state not captured')
                        run['state']='Training' if scheduler=='RUNNING' else 'Training complete / evaluation pending' if saved.get('complete') else 'Checkpointed / queued' if scheduler=='PENDING' else 'Checkpointed; scheduler state not captured'
                        run['execution_phase']='Resuming under corrected task-local requeue wrapper'
                binding=source(campaign_path,'Latest source-bound campaign checkpoint and migration audit')
                run['sources'].append({'file':binding['file'],'sha256':binding['sha256']})
            binding={'file':str(plan_path.relative_to(workspace)),'sha256':sha(plan_path)}
            if binding not in run['sources']:run['sources'].append(binding)
            touched.add(rid)
    # New integrations and continuations remain unscored until complete source
    # reports arrive. Inherited checkpoints do not count as new optimizer work.
    followups=[]
    integration_root=workspace/'artifacts/experiment_audit_20261121/swinir_campaign'
    integration_plan=read(integration_root/'prepared/campaign.json',{})
    integration_submission=read(integration_root/'submission.json',{})
    if integration_plan and integration_submission.get('array_job_id'):
        captured=max(captured,integration_submission.get('released_utc',captured))
        receipt=source(integration_root/'submission.json','Source-verified SwinIR training submission')
        for task in integration_plan['tasks']:
            dataset='her2match' if task['dataset']=='HER2Match' else task['dataset']
            rid='missing__'+task['id']+'_20261121'
            run=register(runs.get(rid) or fresh_run(rid,'swinir',dataset,task='ihc',marker=task['marker'],variant='Native SwinIR · paired RGB L1 · seed 42'))
            run.update(state='Queued / native training check required',scheduler_status='PENDING',training_job=str(integration_submission['array_job_id'])+'_'+str(task['index']),source_sha256=task['source_sha256'],config_sha256=task['config_sha256'],status_as_of=integration_submission.get('released_utc'),registered_only=True,last_step=0,gate_passed=False,training_complete=False,crop_size=256,notes='Author-capacity same-resolution SwinIR, adapted with paired RGB L1. Actual four-rank pause/resume check required before production; no production updates observed at submission.')
            config=integration_root/'prepared/configs'/(task['id']+'.yml')
            if config.exists():
                assert sha(config)==task['config_sha256'],'Prepared SwinIR config changed'
                run['configuration']=config.read_text()
            run['sources'].append({'file':receipt['file'],'sha256':receipt['sha256']})
            directory=cache/'followups/swinir'/task['id']
            status(run,directory);history(run,directory,force=True)
            live=followup_records.get(rid,{})
            smoke=read(cache/'followups/swinir_smokes'/task['id']/'smoke_receipt.json',{})
            final=directory/'final_evaluation.json'
            if final.exists():
                cfg=yaml.safe_load(config.read_text())
                verify_native_gate(smoke,task['source_sha256'],cfg)
                dataset_record=next(d for d in data['datasets'] if d['id']==dataset)
                expected_counts={split:(dataset_record['counts'][split] if dataset=='her2match' else dataset_record['counts'][split][task['marker']]) for split in ('val','test')}
                verify_completed_integration(read(final),cfg,task,expected_counts)
                ingest(run,final,dataset,'coverage_her2match' if dataset=='her2match' else 'coverage_breast')
                run.update(evaluation_complete=True,gate_passed=True,training_complete=True,registered_only=False,notes='Author-capacity SwinIR completed its actual four-rank pause/resume check and complete selected-checkpoint validation/test evaluation.')
                source(cache/'followups/swinir_smokes'/task['id']/'smoke_receipt.json','Actual native SwinIR pause/resume proof')
            if live:apply_live_execution(run,live,followup_snapshot['captured_utc'])
            followups.append(dict(run_id=rid,model='SwinIR',dataset=dataset,job=run['training_job'],state=run['state'],kind='New model integration'))
        models['swinir']['notes']='Native author-capacity same-resolution SwinIR is integrated on BCI/HER2Match with paired RGB L1. Completed results require verified four-GPU resume checks and full selected-checkpoint held-out evaluations.'
    audit_root=workspace/'artifacts/experiment_audit_20261121'
    continuation=read(audit_root/'continuation_submission.json',{})
    preparation=read(audit_root/'restormer_continuation_preparation.json',{})
    if continuation.get('array_job_id') and preparation:
        captured=max(captured,continuation['submitted_utc'])
        rid='her2match__restormer_continuation_20261121'
        run=register(runs.get(rid) or fresh_run(rid,'restormer','her2match',variant='Validation-guided continuation from 80,000 updates',parent_run='her2match__restormer_seed42',compute_parent_run='her2match__restormer_seed42'))
        run.update(state='Queued / post-budget update check required',scheduler_status='PENDING',training_job=str(continuation['array_job_id'])+'_0',source_sha256=continuation['source_sha256'],status_as_of=continuation['submitted_utc'],registered_only=True,last_step=0,parent_step=preparation['parent_step'],new_optimizer_updates=0,training_complete=False,gate_passed=False,crop_size=256,notes='Original 80k checkpoint, optimizer, scheduler horizon, nonzero LR floor and validation patience retained. A real post-budget parameter-update check precedes continuation. Native historical worker augmentation RNG was not saved; bitwise replay is unavailable.')
        receipt=source(audit_root/'continuation_submission.json','Native Restormer continuation submission')
        run['sources'].append({'file':receipt['file'],'sha256':receipt['sha256']})
        live=followup_records.get(rid,{})
        if live:
            apply_live_execution(run,live,followup_snapshot['captured_utc'])
            run['notes']='Native continuation retains its optimizer, scheduler horizon, positive LR floor and validation patience. Startup time-budget fault repaired; actual completed updates and queue state are recorded separately. Native historical worker augmentation RNG was not saved.'
        followups.append(dict(run_id=rid,model='Restormer',dataset='her2match',job=run['training_job'],state=run['state'],kind='Continue improving validation'))
    eqm_submission=read(audit_root/'eqm_continuation_submission.json',{})
    if eqm_submission.get('array_job_id'):
        captured=max(captured,eqm_submission['submitted_utc'])
        rid='finland__research_eqm_continuation_20261121'
        run=register(runs.get(rid) or fresh_run(rid,'eqm','finland',variant='Native EqM research continuation from 30,000 updates',parent_run='finland__research_eqm_20261017',compute_parent_run='finland__research_eqm_20261017'))
        run.update(state='Queued after Restormer / post-budget update check required',scheduler_status='PENDING',training_job=str(eqm_submission['array_job_id'])+'_0',training_dependency=eqm_submission['dependency'],source_inventory_sha256=eqm_submission['source_inventory_sha256'],status_as_of=eqm_submission['submitted_utc'],registered_only=True,last_step=0,parent_step=eqm_submission['parent_step'],new_optimizer_updates=0,training_complete=False,gate_passed=False,crop_size=256,notes='Original model, EMA, optimizer, constant learning rate, scheduler, saved RNG and sampler retained. Isolated operational budget extension; original fixed 256-field stopping monitor and separate full validation remain research-only and unranked.')
        receipt=source(audit_root/'eqm_continuation_submission.json','Native EqM research continuation submission')
        run['sources'].append({'file':receipt['file'],'sha256':receipt['sha256']})
        live=followup_records.get(rid,{})
        if live:apply_live_execution(run,live,followup_snapshot['captured_utc'])
        validation_path=cache/'followups/eqm/validation.jsonl'
        if validation_path.exists():
            checks=new_research_validation(validation_path.read_text(),run['parent_step'])
            if checks:
                binding=source(validation_path,'Actual post-budget EqM selection-monitor checks')
                protocol='EqM continuation · fixed 256-crop selection monitor'
                run['evaluations']=[e for e in run['evaluations'] if e.get('protocol')!=protocol]
                latest=checks[-1]
                run['evaluations'].append(dict(id=evaluation_id(rid,'finland','monitor',protocol,256),dataset='finland',marker='H&E',split='monitor',n=256,protocol=protocol,metrics=latest['metrics'],rankable=False,position=latest['step'],as_of=captured,evidence={'file':binding['file'],'sha256':binding['sha256']}))
                run['curves']={('val/'+key):[[c['step'],c['metrics'][key]] for c in checks if key in c['metrics']] for key in latest['metrics']}
                run['curve_counts']={key:len(points) for key,points in run['curves'].items()}
                run['curve_protocol']=dict(split='training selection monitor',n=256,stage='main',parent_history_run=run['parent_run'])
                write_csv(out/'downloads'/f'{rid}_curves.csv',['tag','step_or_epoch','value'],([tag,x,y] for tag,points in run['curves'].items() for x,y in points))
        followups.append(dict(run_id=rid,model='EqM',dataset='finland',job=run['training_job'],state=run['state'],kind='Continue improving research validation'))
    data['followup_campaigns']=followups
    public_audit=read(audit_root/'convergence_summary_public.json',{})
    if public_audit:
        captured=max(captured,public_audit['snapshot'])
        write(out/'data/training-audit.json',public_audit)
        data['training_audit']={k:v for k,v in public_audit.items() if k!='runs'}|{'download':'data/training-audit.json'}
        source(audit_root/'convergence_summary_public.json','Training convergence audit; validation-based decisions')
    capacity_candidates=[]
    for path in (workspace/'artifacts').glob('archives_resume_*/resource_capacity_public.json'):
        record=read(path,{})
        if record.get('snapshot'):capacity_candidates.append((record['snapshot'],path,record))
    if capacity_candidates:
        _,capacity_path,capacity=max(capacity_candidates,key=lambda entry:entry[0])
        captured=max(captured,capacity['snapshot'])
        write(out/'data/gpu-activity.json',capacity)
        data['gpu_activity']=copy.deepcopy(capacity)|{'download':'data/gpu-activity.json'}
        deferred_path=capacity_path.parent/'prepared/campaign.json'
        deferred=read(deferred_path,{})
        if deferred:
            public_deferred=dict(snapshot=capacity['snapshot'],submitted=False,run_records=deferred['task_count'],readiness_groups=deferred['readiness_groups'],runs=[dict(id=t['id'],model=t['model'],datasets=t['datasets'],saved_step=(t.get('parent_position') or {}).get('global_step'),gpus=t['per_task_gpus'],readiness=t['readiness'],selection_metric=t['scientific_selection_metric'],checkpoint_available=bool(t.get('parent_checkpoint_sha256'))) for t in deferred['tasks']])
            write(out/'data/deferred-training.json',public_deferred)
            data['gpu_activity']['blocked'].update(readiness_groups=deferred['readiness_groups'],plan_download='data/deferred-training.json')
            source(deferred_path,'Concrete deferred native training plans; no submissions without original data')
        write(out/'data/gpu-activity.json',data['gpu_activity'])
        source(capacity_path,'Actual shared GPU capacity and runnable training queue')
    if followup_path:
        receipt=source(followup_path,'Current follow-up gate, update and scheduler observations')
        for rid in followup_records:
            if rid in runs:runs[rid]['sources'].append({'file':receipt['file'],'sha256':receipt['sha256']})
    source(out/'data/swinir-protocol-review.json','SwinIR metric and data code equivalence review')
    attach_reviewed_protocols(runs,out/'data/protocol-compatibility.json',out/'data/swinir-protocol-review.json')
    # Preserve the original reconciled workbook rows as anchors; newly scored
    # rows are reconciled to the fresh canonical files instead of rewriting Excel.
    baseline_evals={(r['id'],e['id']):e for r in baseline['runs'] for e in r['evaluations'] if e.get('rankable')}
    for (rid,eid),old in baseline_evals.items():
        current_eval=next((e for e in runs[rid]['evaluations'] if e['id']==eid),None);assert current_eval,f'Removed original score: {rid}'
        for key,value in old['metrics'].items():assert current_eval['metrics'].get(key)==value,f'Changed original workbook-era score: {rid}/{key}'
    for (rid,eid),old in previous_evals.items():
        current_eval=next((e for e in runs[rid]['evaluations'] if e['id']==eid),None)
        assert current_eval and current_eval['metrics']==old['metrics'],f'Changed previously published score: {rid}/{eid}'
    keys=set(data['metrics'])
    for run in runs.values():
        for e in run['evaluations']:keys.update(e['metrics'])
    for key in keys-set(data['metrics']):data['metrics'][key]=dict(label=key.replace('_',' ').title(),direction='—',unit='recorded units',description='Additional recorded diagnostic. Read the source-specific protocol and metric coverage; no ranking direction has been assigned.')
    index_runs=[]
    for rid,run in runs.items():
        run['sources']=list({json.dumps(source,sort_keys=True):source for source in run.get('sources',[])}.values())
        if any(e.get('rankable') and e['split'] in {'val','test'} for e in run['evaluations']):run['state']='Evaluated'
        run['curve_counts']=run.get('curve_counts') or {k:len(v) for k,v in run.get('curves',{}).items()};run.setdefault('curve_axis','step')
        write(out/'data/runs'/f'{rid}.json',run)
        row={k:v for k,v in run.items() if k not in {'curves','gallery','distributions','configuration','live_status','research_validation_report'}}
        row.update(has_curves=bool(run.get('curves')),image_count=len(run.get('gallery',[])),distribution_splits=list(run.get('distributions',{})))
        index_runs.append(row)
    for model in models.values():
        mine=[r for r in index_runs if r['model']==model['id']];model['runs']=len(mine);model['evaluated_runs']=sum(any(e.get('rankable') and e['split']=='test' for e in r['evaluations']) for r in mine);model['datasets']=sorted({e['dataset'] for r in mine for e in r['evaluations'] if e.get('rankable')})
    now=datetime.now(timezone.utc).isoformat();data.update(snapshot=max(captured,campaign_snapshot.get('snapshot_utc',captured)),runs=index_runs,models=list(models.values()))
    audit=data['audit'];audit['original_workbook_snapshot']=baseline['snapshot'];audit['refresh_observed_utc']=captured;audit['original_ranked_rows_preserved']=len(baseline_evals)
    audit['fresh_result_rows_verified']=sum(c['new_evaluation'] for c in changes);audit['current_unstained_result_rows_verified']=sum(e.get('rankable') and r['task']=='unstained' for r in index_runs for e in r['evaluations'])
    audit['previous_published_ranked_rows_preserved']=len(previous_evals);audit['new_rows_since_previous_snapshot']=sum(c['new_since_previous_snapshot'] for c in changes)
    audit['unstained_result_rows_verified']=audit['current_unstained_result_rows_verified'];audit['new_result_reconciliation']='data/refresh-provenance.json'
    audit['notes']=[n for n in audit.get('notes',[]) if not n.startswith('Cluster evaluation')]+['Original workbook-era scored means were checked and retained unchanged. New result rows are reconciled independently to complete source reports and checkpoint/manifest hashes. Current unfinished runs remain unscored.']
    write(out/'data/index.json',data)
    write(out/'data/refresh-provenance.json',dict(schema='benchmark-live-refresh-v1',observed_utc=captured,exported_utc=now,original_snapshot=baseline['snapshot'],original_index_sha256=sha(baseline_path),original_ranked_rows_preserved=len(baseline_evals),original_score_fingerprints=[dict(run=rid,evaluation=eid,metric_keys=sorted(old['metrics']),metrics_sha256=hashlib.sha256(json.dumps(old['metrics'],sort_keys=True,separators=(',',':')).encode()).hexdigest()) for (rid,eid),old in baseline_evals.items()],new_result_rows=[c for c in changes if c['new_evaluation']],sources=sources))
    reconciliation=read(out/'data/reconciliation.json');reconciliation['audit']=audit;reconciliation['new_result_rows']=[c for c in changes if c['new_evaluation']];write(out/'data/reconciliation.json',reconciliation)
    metric_keys=sorted(keys)
    write_csv(out/'downloads/results.csv',['run_id','model','dataset','marker','split','protocol','tiles','rankable','training_hours']+metric_keys,([r['id'],r['name'],e['dataset'],e['marker'],e['split'],e['protocol'],e['n'],e['rankable'],r.get('training_hours')]+[e['metrics'].get(k) for k in metric_keys] for r in index_runs for e in r['evaluations']))
    return dict(runs=len(index_runs),datasets=len(data['datasets']),fresh_result_rows=audit['fresh_result_rows_verified'],preserved_ranked_rows=len(baseline_evals),snapshot=now)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2]);args=parser.parse_args();print(json.dumps(refresh(Path(__file__).resolve().parents[1],args.workspace.resolve()),indent=2))
