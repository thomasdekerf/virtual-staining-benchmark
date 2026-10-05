"""Create a static, traceable benchmark snapshot from workbook and run evidence.

No training, checkpoint selection, or evaluation is performed by this exporter.
Raw checkpoints, source datasets, cluster paths and patient identifiers stay local.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, re, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import openpyxl
from PIL import Image
import yaml

SITE = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=Path, default=SITE.parent)
parser.add_argument('--quick', action='store_true', help='Skip TensorBoard and distribution extraction for an initial preview')
parser.add_argument('--reuse-curves', action='store_true', help='Reuse exported scalar CSVs and previews; only use when raw TensorBoard logs are unchanged')
args = parser.parse_args()
ROOT = args.workspace.resolve()
OUT = SITE / 'dist'
for name in ['data/runs', 'assets', 'downloads']:
    (OUT/name).mkdir(parents=True, exist_ok=True)
warnings = []
workbook_path = ROOT / 'Virtual_Staining_Model_Priorities.xlsx'
wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)

def read_json(p, default=None):
    try: return json.loads(p.read_text())
    except (OSError, ValueError): return default

def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def clean(o):
    if isinstance(o, dict): return {str(k):clean(v) for k,v in o.items() if k is not None}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o): return None
    if isinstance(o, str):
        o = re.sub(r'/(?:sofia|user|mnt|media|home|Users)/[^\s\"\',;]+', '<local path>', o)
        return o
    if isinstance(o, (int,float,str,bool)) or o is None: return o
    return str(o)

def write_json(p, o):
    p.write_text(json.dumps(clean(o), separators=(',', ':'), ensure_ascii=False, allow_nan=False))

def sheet_rows(name, header_row):
    rows=list(wb[name].values)
    return [(i,dict(zip(rows[header_row-1],r))) for i,r in enumerate(rows[header_row:],header_row+1) if r and r[0] is not None]

names = {'pyramid':'Pyramid Pix2Pix','tdkstain':'TDKStain','asp':'ASP','pspstain':'PSPStain','histdist':'HistDiST','dvst':'D-VST','mcs_stain':'MCS-Stain','dgr':'DGR','hemit':'HEMIT','diffvs':'DiffVS','miphei_vit':'MiPHEI-ViT','unet':'U-Net','pix2pix':'Pix2Pix','cyclegan':'CycleGAN','cut':'CUT','bcistainer':'BCIStainer','pgvms':'PGVMS','unistainnet':'UNIStainNet','pd_unist':'PD-UniST','stainexpert':'StainExpert','cssp2p':'CSSP2P-GAN','implicitstainer':'ImplicitStainer','oda_gan':'ODA-GAN','nafnet':'NAFNet','restormer':'Restormer','hat':'HAT','palette':'Palette','i2imamba':'I2I-Mamba','flow':'Conditional flow','dit':'DiT','vae':'H&E VAE','stable':'STABLE','mmt':'Marker transport U-Net'}
aliases={'DGR (DTR repository)':'dgr','HEMIT dual-branch Pix2Pix':'hemit','DiffVS / Diffusion-FT':'diffvs','MIPHEI-ViT':'miphei_vit','U-Net regression baseline':'unet','Pix2Pix baseline':'pix2pix','CycleGAN baseline':'cyclegan','CUT baseline':'cut'}
def model_id(method):
    if method in aliases:return aliases[method]
    for k,v in names.items():
        if method.lower()==v.lower():return k
    return re.sub(r'[^a-z0-9]+','_',method.lower()).strip('_')

def infer_model(run):
    n=run.removeprefix('priority_')
    for k in sorted(names,key=len,reverse=True):
        if n.startswith(k+'_'):return k
    if 'hat' in n:return 'hat'
    if 'restormer' in n:return 'restormer'
    if 'pix2pix' in n:return 'pix2pix'
    if 'cycle_' in n:return 'cyclegan'
    if 'stable' in n:return 'stable'
    if n.startswith('panel_') or 'unet' in n or n.startswith('study_fusion') or n.startswith('study_refine'):return 'unet'
    return 'other'

models={}
for row,r in sheet_rows('Model priorities',9):
    mid=model_id(r['Method'])
    models[mid]={'id':mid,'name':r['Method'],'priority':r['Priority'],'task':r.get('Task'),'backbone':r.get('Backbone'),'family':r.get('Training family'),'pairing':r.get('Training pairing'),'notes':r.get('Protocol notes'),'recommendation':r.get('Recommendation'),'implementation':r.get('Implementation status'),'url':r.get('Author code'),'orion':r.get('ORION progress'),'breast':r.get('Breast IHC progress'),'source':{'sheet':'Model priorities','row':row}}

datasets=[
 {'id':'orion','name':'ORION CRC','task':'mif','direction':'Multiplex IF → H&E','tissue':'Human colorectal cancer','description':'17-channel multiplex fluorescence translated to paired H&E. Delivered models include regression, GAN, diffusion, transformer and marker-panel studies.','counts':{'train':295096,'val':12402,'test':10952},'notes':'Crop sizes, validation subsets and SSIM implementations differ between historical studies. Choose one protocol before comparing. Held-out tiles come from two slides.','source':'https://github.com/labsyspharm/orion-crc'},
 {'id':'BCI','name':'BCI','task':'ihc','direction':'H&E → IHC','tissue':'Human breast · HER2','markers':['HER2'],'description':'Weakly paired H&E and HER2 IHC images. Separate-dataset and pooled training are evaluated on the same frozen BCI test tiles.','notes':'Original test preserved. Validation uses 15% grade-stratified training tiles. Case identity and patient independence are unverified. Native 256-pixel crops.','source':'https://github.com/bupt-ai-cz/BCI'},
 {'id':'MIST','name':'MIST','task':'ihc','direction':'H&E → IHC','tissue':'Human breast · four markers','markers':['HER2','ER','PR','Ki67'],'description':'H&E to HER2, ER, PR and Ki67. A frozen source-group split replaces overlapping author splits.','notes':'70/15/15 source-group split across all markers. Source groups are not verified patients. Physical field of view varies.','source':'https://github.com/lifangda01/AdaptiveSupervisedPatchNCE'},
 {'id':'IHC4BC','name':'IHC4BC','task':'ihc','direction':'H&E → IHC','tissue':'Human breast · four markers','markers':['HER2','ER','PR','Ki67'],'description':'Marker-specific baselines and pooling with MIST (and BCI for HER2). All models share the same evaluation manifests.','notes':'70/15/15 source-case split, stratified by marker availability. Shared source identifiers stay together. Patient independence is unverified.','source':'https://ihc4bc.github.io/'},
 {'id':'finland','name':'Finland prostate','task':'unstained','direction':'Unstained brightfield → H&E','tissue':'Mouse anterior prostate','description':'39,965 registered 1024-pixel RGB tile pairs from 81 slides, at 0.353 µm/pixel. Native 256-pixel crops for evaluation.','counts':{'train':28807,'val':5979,'test':5179},'slides':{'train':57,'val':12,'test':12},'notes':'Slide-disjoint local split with seed 42. Animal identities unknown. This is not the paper’s published split.','source':'https://doi.org/10.23729/9ddc2fc5-9bdb-404c-be07-c9c9540a32de'},
 {'id':'skin','name':'DermaRepo skin','task':'unstained','direction':'Unstained brightfield → H&E','tissue':'Human skin','description':'1,985 registered tile pairs from 60 slides in 21 accession groups. Targets are chemical H&E. Nominal 0.5 µm/pixel.','counts':{'train':1585,'val':275,'test':125},'slides':{'train':44,'val':10,'test':6},'cases':{'train':15,'val':3,'test':3},'notes':'Case-disjoint split; patient identities unknown. 16 of 76 matching slides failed registration checks. Global registration leaves local deformation and focus differences.','source':'https://doi.org/10.17632/gxgg933ny3.1'}
]
breast=read_json(ROOT/'artifacts/breast_ihc/campaign/progress.json',{})
for ds in datasets:
    if ds['task']=='ihc':
        counts=breast.get('campaign',{}).get('protocol',{}).get('counts',{})
        ds['counts']={split:{m:values.get(ds['id']+'/'+m,0) for m in ds['markers']} for split,values in counts.items()}

runs={}
def new_run(rid,mid,dataset='orion',**kw):
    if rid not in runs:
        runs[rid]={'id':rid,'model':mid,'name':names.get(mid,mid),'variant':rid.split('_job')[0].replace('_',' '),'datasets':[dataset],'task':next((d['task'] for d in datasets if d['id']==dataset),'mif'),'marker':'H&E','state':'Saved run','seed':None,'evaluations':[],'sources':[],'curves':{},'gallery':[],'distributions':{},**kw}
    if mid not in models:
        models[mid]={'id':mid,'name':names.get(mid,mid),'priority':None,'family':'Historical experiment','notes':'Supplemental model family discovered in saved training runs.','source':{'type':'run inventory'}}
    return runs[rid]

def metrics(d):
    out={}
    for k,v in (d or {}).items():
        if finite(v) and k not in ['samples','n','count']:out['delta_e76' if k=='delta_e' else k]=v
    return out

def add_eval(run,dataset,split,vals,n,protocol,**kw):
    if not vals:return
    token=hashlib.sha256(f'{run["id"]}|{dataset}|{split}|{protocol}|{n}'.encode()).hexdigest()[:12]
    e={'id':token,'dataset':dataset,'split':split,'n':n,'protocol':protocol,'metrics':metrics(vals),'marker':run.get('marker','H&E'),'rankable':True,**kw}
    old=next((x for x in run['evaluations'] if x['id']==token),None)
    if old:old.update(e)
    else:run['evaluations'].append(e)

def box_protocol(n=10952,crop=256):return f'Box SSIM · {crop}px · n={n:,}'
def sk_protocol(n=10952,crop=256,legacy=False):return f'{"Legacy test" if legacy else "Matched FP32"} · skimage SSIM · {crop}px · n={n:,}'

# The workbook is an inventory, not the only source of results.
orion_sheet=[]
for row,r in sheet_rows('ORION runs',7):
    mid=model_id(r['Method']); checkpoint=r.get('Checkpoint')
    if checkpoint:rid=Path(checkpoint).parent.name
    else:
        cfg=r.get('Original / campaign configuration')
        rid=(f'priority_{Path(cfg).stem}_production_job1383267' if cfg else 'inventory_'+mid+'_'+str(row))
    run=new_run(rid,mid,variant=r.get('Variant'),state=r.get('Progress'),training_hours=r.get('Training allocation (h)'),gpu_hours=r.get('GPU allocation (h)'),position=r.get('Epoch / step'),selection=r.get('Checkpoint selection'),time_definition=r.get('Time definition'),notes=r.get('Notes / requirement'))
    run['sources'].append({'sheet':'ORION runs','row':row})
    orion_sheet.append((row,rid))
    if finite(r.get('PSNR (dB)')):
        proto=r.get('Metric protocol') or ''
        n=r.get('Tiles') or 0
        protocol=sk_protocol(n,legacy='legacy' in proto.lower()) if any(k in proto.lower() for k in ['skimage','scikit','7x7','7×7']) else box_protocol(n)
        add_eval(run,'orion',r.get('Split','val'),{'psnr':r.get('PSNR (dB)'),'ssim':r.get('SSIM'),'lpips':r.get('LPIPS'),'mae':r.get('MAE'),'delta_e76':r.get('DeltaE76')},n,protocol,evidence={'sheet':'ORION runs','row':row},selection=r.get('Checkpoint selection'))

for t in breast.get('tasks',[]):
    rid=t['id'];result=t.get('result') or {}
    run=new_run(rid,t['model'],t['datasets'][0],task='ihc',datasets=t['datasets'],marker=t['marker'],variant='Pooled datasets' if t['stage']=='pooled' else 'Separate dataset',stage=t['stage'],state=t['status'],seed=42,training_hours=result.get('training_hours'),gpu_hours=result.get('gpu_hours'),position=result.get('position'),selection=result.get('selection'),time_definition=result.get('time_definition'),checkpoint_sha256=result.get('checkpoint_sha256'),source_sha256=result.get('source_sha256'),evaluation_seconds=result.get('evaluation_seconds'),live_status=t.get('live_status'),train_tiles=t.get('train_tiles'))
    run['sources'].append({'file':'artifacts/breast_ihc/campaign/progress.json','record':rid})
    cfg=ROOT/'artifacts/breast_ihc/campaign'/t['config']
    if cfg.exists():run['configuration']=cfg.read_text()
    for split,s in result.get('full_splits',{}).items():
        for ds,v in s.get('per_dataset',{}).items():
            vals=metrics(v.get('metrics'));dab=v.get('dab',{})
            vals.update({'dab_od_mae':dab.get('mean_absolute_error'),'dab_correlation':dab.get('correlation'),'dab_pred_mean':dab.get('mean_pred'),'dab_target_mean':dab.get('mean_target')})
            add_eval(run,ds,split,vals,v['n'],box_protocol(v['n']),manifest_sha256=s.get('manifest_sha256'),metric_protocol=result.get('metric_protocol'),selection=result.get('selection'),dab_samples=dab.get('samples'),evidence={'file':'artifacts/breast_ihc/campaign/progress.json','record':rid})
    hp=ROOT/'artifacts/breast_ihc/campaign/histories'/f'{rid}.csv'
    if hp.exists():
        for point in csv.DictReader(hp.open()):
            if not point.get('step'):continue
            for key,val in point.items():
                if key in ['epoch','step'] or not val:continue
                run['curves'].setdefault('val/'+key,[]).append([float(point['step']),float(val)])

# Reconcile the dataset sheets cell-by-cell against production JSON.
reconciliation=[]
for sheet in ['BCI','MIST','IHC4BC']:
    for row,r in sheet_rows(sheet,8):
        rid=r.get('Run ID')
        if rid not in runs:continue
        run=runs[rid]
        run['sources'].append({'sheet':sheet,'row':row})
        e=next(x for x in run['evaluations'] if x['dataset']==sheet and x['split']==r['Split'])
        for col,key in [('PSNR (dB) ↑','psnr'),('SSIM ↑','ssim'),('LPIPS ↓','lpips'),('RGB MAE ↓','mae'),('DeltaE76 ↓','delta_e76'),('DAB OD MAE ↓','dab_od_mae'),('DAB correlation ↑','dab_correlation')]:
            a,b=r.get(col),e['metrics'].get(key)
            if finite(a) and (not finite(b) or abs(a-b)>1e-9):raise ValueError(f'Workbook mismatch: {sheet}!row {row}: {col}')
        reconciliation.append({'sheet':sheet,'row':row,'run':rid,'evaluation':e['id']})

for ds,folder,sheet in [('finland','unstained_finland','Unstained Finland'),('skin','unstained_dermarepo','Unstained Skin')]:
    campaign=read_json(ROOT/'artifacts'/folder/'campaign.json',{})
    progress=read_json(ROOT/'artifacts'/folder/'progress.json',{})
    tasks=campaign.get('tasks',[])
    for t in tasks:
        rid=ds+'__'+t['id'];p=progress.get('tasks',{}).get(t['id'],{})
        run=new_run(rid,t['model'],ds,variant='RGB adaptation · seed 42',state=p.get('status',t.get('status')),seed=t.get('seed'),training_hours=p.get('training_hours'),gpu_hours=p.get('gpu_hours'),notes=t.get('reason') or p.get('reason'),live_status=p.get('training_state'),selection='Minimum full-validation AlexNet LPIPS; 10 warmup checks, then 20 checks without improvement')
        run['sources'].append({'file':f'artifacts/{folder}/progress.json','record':t['id']})
        cfg=ROOT/'artifacts'/folder/t.get('config','missing')
        if cfg.exists():run['configuration']=cfg.read_text()
    for row,r in sheet_rows(sheet,9):
        directory=r.get('Run directory')
        if not directory:continue
        rid=ds+'__'+Path(directory).name
        if rid in runs:runs[rid]['sources'].append({'sheet':sheet,'row':row})

source_dirs={}
for base in [ROOT/'sofia_tensorboard',ROOT/'artifacts/priority_suite/existing_evidence']:
    if base.exists():
        for d in base.iterdir():
            if d.is_dir() and (d/'config.yml').exists():source_dirs.setdefault(d.name,[]).append(d)
for campaign in ['asp','pspstain','histdist','palette','pyramid_pix2pix']:
    base=ROOT/'artifacts'/campaign/'final_validation'
    for d in base.glob('*'):
        if d.is_dir() and (d/'config.yml').exists():source_dirs.setdefault(d.name,[]).append(d)
for ds in ['orion','finland','skin']:
    for d in (SITE/'.source-cache'/ds).glob('*'):
        if d.is_dir() and ((d/'config.yml').exists() or (d/'status.json').exists() or (d/'tensorboard').exists()):
            rid=d.name if ds=='orion' else ds+'__'+d.name
            source_dirs.setdefault(rid,[]).append(d)

for rid,dirs in source_dirs.items():
    if 'smoke' in rid or 'segmenter' in rid or 'profile' in rid:continue
    ds=rid.split('__')[0] if '__' in rid else 'orion'
    run=new_run(rid,infer_model(rid.split('__')[-1]),ds)
    for d in dirs:
        cfg=read_json(d/'run_metadata.json',{})
        if cfg.get('manifest_rows',{}).get('train'):run['train_tiles']=cfg['manifest_rows']['train']
        if (d/'config.yml').exists():
            run['configuration']=(d/'config.yml').read_text()
            try:
                c=yaml.safe_load(run['configuration']);run['seed']=c.get('seed',run['seed']);run['crop_size']=c.get('data',{}).get('crop_size',256)
            except Exception:pass
        status=read_json(d/'status.json',{})
        if status:
            run['live_status']=status
            run['last_step']=status.get('step',status.get('global_step'))
            if status.get('complete'):run['state']='Training completed'
            elif status.get('budget_pause'):run['state']='Paused for continuation'
            elif status.get('step',status.get('global_step',0)):run['state']='Training snapshot'
            if status.get('early_stop'):run['stop_reason']='Validation plateau'
            elif status.get('complete'):run['stop_reason']='Configured schedule completed'
            elif status.get('budget_pause'):run['stop_reason']='Wall-time pause'
        alloc=status.get('allocation',{})
        if alloc.get('seconds') and run.get('training_hours') is None:
            run['training_hours']=alloc['seconds']/3600;run['gpu_hours']=run['training_hours']*alloc.get('world_size',4);run['time_definition']='Recorded training-job allocation; setup, validation and checkpoints included. Parent and earlier segments may be excluded.'
        # Audited final summaries take precedence over latest training status.
        files=list(d.glob('final_evaluation.json'))+list(d.glob('*/summary.json'))
        for f in files:
            obj=read_json(f,{})
            if obj.get('full_splits'):
                for key in ['training_hours','gpu_hours','time_definition','checkpoint_sha256','source_sha256','selection','position','evaluation_seconds']:
                    if obj.get(key) is not None:run[key]=obj[key]
                complete=obj.get('all_literature_metrics_complete',True)
                for split,s in obj['full_splits'].items():
                    if not s.get('metrics'):continue
                    vals=metrics(s['metrics']);coverage={}
                    for group in ['image_metrics','sampled_stain_and_nuclei']:
                        for k,v in s.get('extended_metrics',{}).get(group,{}).items():
                            if isinstance(v,dict) and finite(v.get('mean')):vals[k]=v['mean'];coverage[k]=v.get('valid_tiles')
                    for detector,values in s.get('learned_nuclei',{}).items():
                        if isinstance(values,dict):
                            for k,v in values.items():
                                if finite(v):vals[detector+'_'+k]=v
                    proto=box_protocol(s.get('n',0)) if ds=='orion' else ('Full production evaluation' if complete else 'Partial production evaluation')
                    vals.update(metrics({k:v for k,v in s.get('distribution',{}).items() if k in ['fid_clean','kid_clean']}))
                    add_eval(run,ds,split,vals,s.get('n'),proto,rankable=complete,partial=not complete,coverage=coverage,group_macro=s.get('group_macro_bootstrap'),bootstrap_unit=s.get('bootstrap_unit'),manifest_sha256=s.get('manifest_sha256'),evidence={'file':str(f.relative_to(ROOT))},metric_protocol=obj.get('protocol'),checkpoint_sha256=obj.get('checkpoint_sha256'))
            elif finite(obj.get('psnr')):
                split=obj.get('split') or ('val' if 'validation' in f.parent.name else 'test')
                n=obj.get('samples',obj.get('n',0));crop=obj.get('crop_size',256)
                recorded=str(obj.get('metric_protocol','')).lower()
                matched='matched_v1' in recorded or any(x in f.parent.name for x in ['test_matched','registration_test','unpaired_test'])
                legacy=f.parent.name=='test' or 'legacy' in recorded
                proto=sk_protocol(n,crop,legacy=legacy and not matched) if matched or legacy or 'skimage' in recorded else box_protocol(n,crop)
                add_eval(run,ds,split,obj,n,proto,manifest_sha256=obj.get('manifest_sha256'),checkpoint_sha256=obj.get('checkpoint_sha256'),position=obj.get('selected_progress'),evidence={'file':str(f.relative_to(ROOT))})
                if obj.get('checkpoint_sha256'):run['checkpoint_sha256']=obj['checkpoint_sha256']
                if obj.get('selected_progress'):run['position']=obj['selected_progress']
        hp=d/'history.jsonl'
        if hp.exists():
            for line in hp.read_text().splitlines():
                try:p=json.loads(line)
                except ValueError:continue
                step=p.get('global_step',p.get('step',p.get('epoch')))
                if not finite(step):continue
                for group in ['train','validation']:
                    for k,v in p.get(group,{}).items():
                        if finite(v) and k!='samples':run['curves'].setdefault(('val/' if group=='validation' else 'train/')+k,[]).append([step,v])
        run['sources'].append({'file':str(d.relative_to(ROOT))})

# Priority suite production evidence includes both frozen evaluation splits.
for p in (ROOT/'artifacts/priority_suite/results').glob('*.json'):
    o=read_json(p,{})
    if not o.get('checkpoint'):continue
    rid=Path(o['checkpoint']).parent.name;run=new_run(rid,infer_model(rid))
    run.update({k:o.get(k) for k in ['training_hours','gpu_hours','time_definition','checkpoint_sha256','source_sha256','selection']})
    run['position']=o.get('position');run['state']='Evaluated'
    for split,s in o.get('full_splits',{}).items():add_eval(run,'orion',split,s.get('metrics'),s.get('n'),box_protocol(s.get('n')),manifest_sha256=s.get('manifest_sha256'),evidence={'file':str(p.relative_to(ROOT))})

# Historical scheduler allocation and selected validation are distinct evidence.
overview=read_json(ROOT/'docs/JOB_OVERVIEW.json',{})
for item in overview.get('catalog',[]):
    rid=item['run']
    if rid not in runs:continue
    run=runs[rid];run['variant']=item.get('variant',run['variant']);run['purpose']=item.get('purpose');run['training_job']=item.get('job')
    job=overview.get('jobs',{}).get(item.get('job'),{})
    if job.get('elapsed') and run.get('training_hours') is None:
        elapsed=job['elapsed'];days=0
        try:
            if '-' in elapsed:day,elapsed=elapsed.split('-',1);days=int(day)
            parts=[int(x) for x in elapsed.split(':')]
            if len(parts)==3:
                seconds=days*86400+parts[0]*3600+parts[1]*60+parts[2]
                run['training_hours']=seconds/3600
                run['time_definition']='Historical Slurm training-job elapsed time. Includes setup, validation and I/O; excludes separate pretraining, parent runs and other continuation allocations.'
        except ValueError:pass
    saved=overview.get('runs',{}).get(rid,{})
    selected=saved.get('selected_validation') or {}
    if selected.get('validation',{}).get('psnr') is not None and saved.get('has_best'):
        config=yaml.safe_load(run.get('configuration','{}')) or {}
        dcfg=config.get('data',{});n=selected['validation'].get('samples',dcfg.get('max_val_samples',12402));crop=dcfg.get('val_crop_size',dcfg.get('crop_size',256))
        proto=box_protocol(n,crop)
        if not any(e['split']=='val' and e['protocol']==proto for e in run['evaluations']):
            add_eval(run,'orion','val',selected['validation'],n,proto,position=selected.get('step'),evidence={'file':'docs/JOB_OVERVIEW.json','record':rid},as_of=overview.get('captured_at_utc'),selection='Saved selected validation checkpoint in historical run snapshot')
    test=saved.get('test') or {}
    if run.get('gpu_hours') is None and run.get('training_hours') is not None and test.get('world_size'):
        run['gpu_hours']=run['training_hours']*test['world_size']
for run in runs.values():
    if isinstance(run.get('variant'),str):run['variant']=re.sub(r'\s+\((?:val|test)\)$','',run['variant']).removeprefix('Existing: ')

def export_pair(run, encoded, step, split='val', label='Saved training preview', selected=False):
    im=Image.open(io.BytesIO(encoded)).convert('RGB')
    # torchvision make_grid(nrow=4, padding=2): targets row, predictions row.
    h=(im.height-6)//2;w=(im.width-10)//4
    if h!=w or w<32:return
    if any(im['label'].startswith(label) for im in run['gallery']):return
    for i in range(4):
        gt=im.crop((2+i*(w+2),2,2+i*(w+2)+w,2+h))
        pred=im.crop((2+i*(w+2),4+h,2+i*(w+2)+w,4+2*h))
        prefix=f'assets/{run["id"]}_{"selected_"+split if selected else step}_{i}'
        gt.save(OUT/(prefix+'_gt.webp'),lossless=True);pred.save(OUT/(prefix+'_pred.webp'),lossless=True)
        run['gallery'].append({'gt':prefix+'_gt.webp','pred':prefix+'_pred.webp','label':f'{label} · field {i+1}','step':step,'split':split,'checkpoint':'selected final checkpoint' if selected else 'preview checkpoint; may differ from the selected final model'})

if args.reuse_curves:
    for rid,run in runs.items():
        old=read_json(OUT/'data/runs'/f'{rid}.json',{})
        scalar_file=OUT/'downloads'/f'{rid}_curves.csv'
        if scalar_file.exists():
            curves={}
            for point in csv.DictReader(scalar_file.open()):curves.setdefault(point['tag'],[]).append([float(point['step_or_epoch']),float(point['value'])])
            run['curves'].update(curves)
        run['gallery']=old.get('gallery',[])

for rid,dirs in source_dirs.items():
    if rid not in runs:continue
    for d in reversed(dirs):
        for split in ['test','val']:
            f=d/f'final_{split}.png'
            if f.exists():export_pair(runs[rid],f.read_bytes(),None,split,f'Selected {split} checkpoint',selected=True)

if not args.quick and not args.reuse_curves:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for number,(rid,dirs) in enumerate(source_dirs.items(),1):
        if rid not in runs:continue
        run=runs[rid]
        # Prefer the synced cluster evidence, otherwise use saved local logs.
        event_dirs=[d/'tensorboard' for d in dirs if (d/'tensorboard').exists() and list((d/'tensorboard').glob('events.out.tfevents.*'))]
        if not event_dirs:continue
        try:
            a=EventAccumulator(str(event_dirs[-1]),size_guidance={'scalars':0,'images':2},purge_orphaned_data=True);a.Reload()
            for tag in a.Tags().get('scalars',[]):
                if tag.startswith(('train/','val/','train_interval/','system/')):
                    points=[[e.step,e.value] for e in a.Scalars(tag) if finite(e.value)]
                    if points:run['curves'][tag]=points
            tag='val/targets_then_predictions'
            if tag in a.Tags().get('images',[]):
                e=max(a.Images(tag),key=lambda x:x.step);export_pair(run,e.encoded_image_string,e.step)
            if number%10==0:print(f'Extracted TensorBoard for {number}/{len(source_dirs)} runs',flush=True)
        except Exception as exc:warnings.append({'run':rid,'type':'tensorboard','reason':str(exc)[:120]})

# Latest validation stays separate from selected checkpoint results.
for run in runs.values():
    state=run.get('live_status') or {}
    vals=state.get('metrics') or {}
    if vals and finite(vals.get('psnr')) and run['task']!='ihc':
        n=vals.get('samples',state.get('validation_samples'))
        coverage=f'n={n:,}' if n is not None else 'tile count not recorded'
        add_eval(run,run['datasets'][0],'monitor',vals,n,f'Latest training validation · {run.get("crop_size",256)}px · {coverage}',rankable=False,evidence={'type':'training status'},position=state.get('step'))
    for tag,points in list(run['curves'].items()):
        # Last saved value per step after resumed/duplicated evidence.
        unique={p[0]:p[1] for p in points}
        run['curves'][tag]=[[x,unique[x]] for x in sorted(unique)]
    if any(e.get('rankable') and e['split'] in ['val','test'] for e in run['evaluations']):run['state']='Evaluated'
    if run['task']=='unstained' and any(e.get('partial') for e in run['evaluations']):run['state']='Partial evaluation'

# The 24-patch pathology pilot is a distinct experiment, never a full-test rank.
pilot_root=ROOT/'artifacts/pathology_evaluation/pilot_20260921'
pilot=read_json(pilot_root/'pilot.json',{})
pilot_definitions={}
pilot_mapping={'PSNR':'psnr','SSIM (7px)':'ssim','LPIPS':'lpips','RGB MAE':'mae','RGB RMSE':'rmse','MS-SSIM':'ms_ssim','RGB Pearson':'pearson_rgb','CIE76 colour error':'delta_e76','CIEDE2000':'delta_e2000','Hematoxylin MAE':'hematoxylin_od_mae','Eosin MAE':'eosin_od_mae','Hematoxylin histogram W1':'hematoxylin_hist_w1','Eosin histogram W1':'eosin_hist_w1'}
pilot_values={mid:{} for mid in ['nafnet','implicitstainer','mcs_stain']}
pilot_coverage={mid:{} for mid in pilot_values}
for row,r in sheet_rows('Evaluation pilot',8):
    label=r.get('Metric')
    if not label or label=='Metric':continue
    key=pilot_mapping.get(label,re.sub(r'[^a-z0-9]+','_',label.lower().replace('2×','2x')).strip('_'))
    pilot_definitions[key]=(label,'↑' if r.get('Direction')=='Higher' else '↓',r.get('Unit'))
    coverage=str(r.get('Valid patches (N / I / M)','')).split(' / ')
    for i,(mid,col) in enumerate([('nafnet','NAFNet'),('implicitstainer','ImplicitStainer'),('mcs_stain','MCS-Stain')]):
        if finite(r.get(col)):pilot_values[mid][key]=r[col]
        if i<len(coverage):
            try:pilot_coverage[mid][key]=int(coverage[i])
            except ValueError:pass
for mid,vals in pilot_values.items():
    source=pilot.get('models',{}).get(mid,{})
    rid=Path(source['run']).name if source.get('run') else None
    if rid in runs:add_eval(runs[rid],'orion','pilot',vals,24,'Pathology pilot · 24 seeded test patches',rankable=False,coverage=pilot_coverage[mid],evidence={'sheet':'Evaluation pilot','rows':'9–64'})
if not args.quick and (pilot_root/'images.npz').exists():
    with np.load(pilot_root/'images.npz') as ims:
        for mid,key in [('nafnet','nafnet'),('implicitstainer','implicit'),('mcs_stain','mcs')]:
            candidates=[r for r in runs.values() if r['model']==mid and 'orion_production' in r['id']]
            if not candidates:continue
            run=next((r for r in candidates if '_reference_' not in r['id']),candidates[0])
            image_key=next((k for k in ims.files if k==key or k.startswith(key)),None)
            real_key=next((k for k in ims.files if k in ['real','gt','target','he']),None)
            if image_key and real_key and not any(im['split']=='pilot' for im in run['gallery']):
                for i in range(min(24,len(ims[image_key]))):
                    prefix=f'assets/pilot_{mid}_{i}'
                    for part,k in [('gt',real_key),('pred',image_key)]:Image.fromarray(np.uint8(np.clip(np.round(ims[k][i]*255),0,255))).save(OUT/(prefix+'_'+part+'.webp'),lossless=True)
                    run['gallery'].append({'gt':prefix+'_gt.webp','pred':prefix+'_pred.webp','label':f'Pathology pilot · patch {pilot.get("indices",list(range(24)))[i]}','split':'pilot','checkpoint':'Selected best checkpoint; 24 seeded test patches, not full-test image coverage'})

def distribution(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return None
    lo,hi=np.quantile(a,[0,1]);sd=np.std(a);band=max(1.06*sd*len(a)**(-.2),float(hi-lo)/200,1e-8)
    grid=np.linspace(lo,hi,81);density=np.zeros(81)
    for chunk in np.array_split(a,max(1,math.ceil(len(a)/500))):density+=np.exp(-.5*((grid[:,None]-chunk[None,:])/band)**2).sum(axis=1)
    density/=max(density.max(),1e-12)
    q=np.quantile(a,[0,.05,.25,.5,.75,.95,1])
    return {'n':len(a),'mean':float(a.mean()),'sd':float(sd),'quantiles':[float(v) for v in q],'density':[[float(x),float(y)] for x,y in zip(grid,density)],'method':'Gaussian KDE; Silverman bandwidth, minimum range/200; full finite observations; width normalized within each violin'}

if not args.quick:
    for rid,dirs in source_dirs.items():
        if rid not in runs:continue
        run=runs[rid];seen=set()
        for d in reversed(dirs):
            files=sorted(list(d.rglob('per_tile.csv'))+list(d.rglob('*_per_tile.csv')),key=lambda f:('test_matched' not in str(f),str(f)))
            for f in files:
                split='val' if '/val/' in str(f) or f.name.startswith('val_') else 'test'
                if split in seen:continue
                records=list(csv.DictReader(f.open()));seen.add(split)
                keys=[k for k in (records[0] if records else {}) if k not in ['index','path','slide','dataset','marker','case','source']]
                dist={}
                for k in keys:
                    values=[]
                    for row in records:
                        try:v=float(row[k])
                        except (TypeError,ValueError):continue
                        if finite(v):values.append(v)
                    stat=distribution(values)
                    if stat:dist['delta_e76' if k=='delta_e' else k]=stat
                if dist:
                    run['distributions'][split]=dist
                    run.setdefault('distribution_evidence',{})[split]={'file':str(f.relative_to(ROOT)),'tiles':len(records),'note':'Full finite per-tile observations from this file. Check the evaluation protocol when comparing with other results.'}
                    # Numeric per-tile export uses index only, omitting source identifiers.
                    destination=OUT/'downloads'/f'{rid}_{split}_per_tile.csv'
                    with destination.open('w',newline='') as fp:
                        writer=csv.writer(fp);writer.writerow(['tile_index']+list(dist))
                        for i,row in enumerate(records):writer.writerow([i]+[row.get('delta_e' if k=='delta_e76' else k) for k in dist])

metric_defs={
 'psnr':('PSNR','↑','dB','Peak signal-to-noise ratio from RGB pixel error, averaged per tile. Higher means less pixel error. Sensitive to registration; does not establish cell fidelity.'),
 'ssim':('SSIM','↑','score','Structural similarity of luminance, contrast and local structure. Windowing differs between box and scikit-image protocols. Compare within the selected protocol.'),
 'lpips':('LPIPS','↓','distance','AlexNet Learned Perceptual Image Patch Similarity. Lower means more similar learned visual features. A natural-image perceptual measure, not a validated pathology endpoint.'),
 'mae':('RGB MAE','↓','RGB [0–1]','Mean absolute error between paired RGB pixels on a 0–1 scale. Lower means closer pixel values; alignment and stain intensity affect it.'),
 'delta_e76':('ΔE76','↓','CIELAB','CIE 1976 Euclidean colour difference in Lab space. Lower means closer colour. This is not CIEDE2000.'),
 'delta_e2000':('ΔE2000','↓','CIEDE2000','CIEDE2000 perceptual colour difference. Measured on the fixed stain-analysis subset where configured. Lower means closer colour.'),
 'dab_od_mae':('DAB OD MAE','↓','optical density','Absolute difference in mean DAB optical density between prediction and target. Descriptive stain diagnostic; not receptor status or cell positivity.'),
 'dab_correlation':('DAB correlation','↑','Pearson r','Correlation of predicted and target DAB optical-density summaries across tiles. A cohort diagnostic, not a per-cell score.'),
 'dab_pred_mean':('Predicted DAB OD','—','optical density','Mean predicted DAB optical density. Interpret beside the target mean; it has no universal best direction.'),
 'dab_target_mean':('Target DAB OD','—','optical density','Mean target DAB optical density. A reference summary, not a model-performance ranking.'),
 'rmse':('RMSE','↓','RGB [0–1]','Root mean squared RGB error. Larger local errors receive more weight than in MAE.'),
 'mse':('MSE','↓','RGB²','Mean squared RGB error on a 0–1 scale.'),
 'ms_ssim':('MS-SSIM','↑','score','Structural similarity at multiple image scales. Higher is better. Scale weights and implementation belong to the recorded protocol.'),
 'pearson_rgb':('RGB Pearson','↑','Pearson r','Paired RGB intensity correlation, averaged over valid channels. Constant channels can be undefined; valid coverage is reported.'),
 'dists':('DISTS','↓','distance','Deep Image Structure and Texture Similarity. Lower indicates greater similarity of learned structure and texture. Not a diagnostic validation.'),
 'fid_clean':('FID clean','↓','distance','Fréchet Inception Distance between generated and real image feature distributions under clean-fid preprocessing. Unpaired distribution similarity, not paired tissue fidelity. Small cohorts can bias estimates.'),
 'kid_clean':('KID clean','↓','raw MMD²','Kernel Inception Distance under clean-fid preprocessing. Raw unbiased MMD², not multiplied by 1000; values can be negative. Compare only identical sampling protocols.'),
 'scm':('SCM','↑','score','Structural component of the configured Gaussian SSIM calculation. Emphasizes paired local structure; distinct from complete SSIM.'),
 'ssim_gaussian':('SSIM Gaussian','↑','score','SSIM using the recorded Gaussian-window protocol. Keep separate from box-window and scikit-image SSIM.'),
 'ssim_skimage':('SSIM skimage','↑','score','Scikit-image SSIM under the recorded evaluator settings. Keep separate from box-window SSIM.'),
 'hematoxylin_od_mae':('H stain MAE','↓','optical density','Paired hematoxylin optical-density error after fixed colour deconvolution. Lower indicates closer stain intensity.'),
 'eosin_od_mae':('E stain MAE','↓','optical density','Paired eosin optical-density error after fixed colour deconvolution. Lower indicates closer stain intensity.'),
 'nuclei_f1':('Nuclei F1','↑','score','Harmonic mean of precision and recall for spatially matched automatic nuclei. Reference labels are detector outputs on real H&E, not manual annotations.'),
 'nuclei_precision':('Nuclei precision','↑','score','Fraction of predicted automatic nuclei that match real-H&E detections within the fixed matching radius.'),
 'nuclei_recall':('Nuclei recall','↑','score','Fraction of real-H&E automatic detections recovered at matching locations in the prediction.'),
 'nuclei_mask_dice':('Nuclei Dice','↑','score','Dice overlap of automatic nuclear foreground masks. Real-H&E detector masks are pseudo-reference labels.'),
 'nuclei_mask_iou':('Nuclei IoU','↑','score','Intersection over union of automatic nuclear foreground masks. Higher means more spatial overlap.'),
 'nuclei_count_abs_error':('Nuclei count error','↓','nuclei/tile','Absolute difference in automatic nuclear counts per tile. Correct counts can still accompany incorrect locations.'),
 'nuclei_area_wasserstein_px2':('Nuclei area W₁','↓','pixel²','Wasserstein distance between detected nuclear area distributions. Pixel units depend on tissue resolution.'),
 'nuclei_eccentricity_wasserstein':('Nuclei shape W₁','↓','distance','Wasserstein distance between detected nuclear eccentricity distributions. Lower means closer shape distributions.'),
 'loss':('Objective loss','↓','recipe-specific','The objective used by this training recipe. Different recipes, weights and objectives have different scales; compare its trend within a run.'),
 'training_hours':('Training time','↓','hours','Recorded training-job wall time, including setup, validation and I/O as specified per run. Parent training and continuation segments may be excluded. A compute comparison, not an accuracy score.'),
 'gpu_hours':('GPU allocation','↓','GPU hours','Wall time multiplied by allocated GPUs. Allocated compute, not measured kernel utilization. Follow each run’s recorded time definition.')
}
all_metric_keys=set(metric_defs)
for run in runs.values():
    for e in run['evaluations']:all_metric_keys.update(e['metrics'])
for key in sorted(all_metric_keys):
    if key in metric_defs:continue
    label=key.replace('_',' ').replace('hoptimus','H-Optimus').replace('uni ','UNI ').replace('conch','CONCH').title()
    direction='—';desc='Additional recorded diagnostic. Read its run protocol and sample coverage before interpreting or comparing.'
    if 'cosine' in key:
        direction='↓' if 'distance' in key else '↑';desc='Paired frozen pathology-encoder embedding similarity. Cosine distance is 1 − cosine similarity. Representation agreement can remain high despite missing or displaced nuclei; not a diagnostic score.'
    elif 'normalized_l2' in key:direction='↓';desc='Euclidean distance between normalized paired encoder embeddings. Mathematically linked to cosine similarity; not independent evidence.'
    elif 'phv' in key:direction='↓';desc='Layer-wise ResNet-50 perceptual histology value: fraction of channels whose spatial feature means differ by more than 0.01. The legacy input-range convention and standard RGB preprocessing are separate protocols.'
    elif 'hist_w1' in key:direction='↓';desc='Wasserstein distance between fixed deconvolved stain-intensity histograms. Lower means closer stain distributions; spatial location is not measured.'
    elif 'count_pearson' in key:direction='↑';desc='Pearson correlation between automatic nuclear counts in predicted and real-H&E patches. Count correlation does not establish correct cell locations.'
    elif 'area_w1' in key:direction='↓';desc='Wasserstein distance between automatic nuclear area distributions, measured in original crop pixel². Depends on detector and tissue resolution.'
    elif 'shape_w1' in key:direction='↓';desc='Wasserstein distance between the eccentricity distributions of automatically detected nuclei. Lower means more similar detected shapes.'
    elif any(s in key for s in ['dice','iou','f1','precision','recall']):direction='↑';desc='Spatial agreement of automatic nuclear detections or masks under the named detector protocol. Real-H&E detections are pseudo-reference labels, not manual ground truth.'
    elif any(s in key for s in ['error','distance','mae','rmse','mse','wasserstein']):direction='↓'
    if key in pilot_definitions:
        label,direction,unit=pilot_definitions[key]
        if 'micro_f1' in key:desc='F1 from pooled true-positive, false-positive and false-negative automatic nuclear matches across all pilot patches. Real-H&E detector outputs are pseudo-reference labels. Typed F1 also requires matching predicted nuclear class.'
    else:unit='recorded units'
    metric_defs[key]=(label,direction,unit,desc)
glossary={k:{'label':v[0],'direction':v[1],'unit':v[2],'description':v[3]} for k,v in metric_defs.items()}

index_runs=[]
for run in runs.values():
    rid=run['id']
    # Full scalar history is downloadable; plot series are deterministically reduced.
    full_curves=run['curves']
    if full_curves:
        with (OUT/'downloads'/f'{rid}_curves.csv').open('w',newline='') as fp:
            writer=csv.writer(fp);writer.writerow(['tag','step_or_epoch','value'])
            for tag,points in full_curves.items():
                for p in points:writer.writerow([tag,*p])
    run['curve_counts']={tag:len(p) for tag,p in full_curves.items()}
    run['curves']={tag:(p if len(p)<=700 else [p[i] for i in sorted(set(np.linspace(0,len(p)-1,700,dtype=int)))]) for tag,p in full_curves.items()}
    run['curve_axis']='epoch' if rid in ['unet_l1_job1320937','gated_unet_l1_job1320938','pix2pix_job1320940','gated_unet_structural_job1320939','gated_pix2pix_structural_job1320936'] else 'step'
    run['gallery'].sort(key=lambda im:('selected final' not in im['checkpoint'],im['split']=='pilot'))
    write_json(OUT/'data/runs'/f'{rid}.json',run)
    index_runs.append({k:v for k,v in run.items() if k not in ['curves','gallery','distributions','configuration','live_status']})
    index_runs[-1].update({'has_curves':bool(run['curves']),'image_count':len(run['gallery']),'distribution_splits':list(run['distributions'])})
for model in models.values():
    mine=[r for r in index_runs if r['model']==model['id']]
    model['runs']=len(mine);model['evaluated_runs']=sum(any(e['rankable'] and e['split']=='test' for e in r['evaluations']) for r in mine)
    model['datasets']=sorted({d for r in mine for d in r['datasets'] if any(e['dataset']==d and e['rankable'] for e in r['evaluations'])})

snapshot=datetime.now(timezone.utc).isoformat()
audit={'workbook_sha256':hashlib.sha256(workbook_path.read_bytes()).hexdigest(),'workbook_sheets':wb.sheetnames,'model_priority_rows':len(sheet_rows('Model priorities',9)),'orion_inventory_rows':len(orion_sheet),'breast_runs':len(breast.get('tasks',[])),'dataset_result_rows_verified':len(reconciliation),'verified_cells':len(reconciliation)*7,'workbook_orion_rows_accounted':all(rid in runs for _,rid in orion_sheet),'warnings':warnings,'notes':['Validation and test entries refer to one training run; compute is stored once.','All 39 model-priority rows are retained, including deferred and out-of-scope methods.','Unstained pending and excluded methods retain their reason without fabricated results.','Cluster evaluation evidence may be newer than the workbook; partial production evaluations are excluded from ranking.','Some breast image/per-tile archives are not locally accessible; curves and complete means are preserved.']}
write_json(OUT/'data/index.json',{'snapshot':snapshot,'workbook_date':'2026-10-05','runs':index_runs,'datasets':datasets,'models':list(models.values()),'metrics':glossary,'audit':audit})
write_json(OUT/'data/reconciliation.json',{'audit':audit,'dataset_rows':reconciliation,'orion_rows':[{'sheet':'ORION runs','row':row,'run':rid} for row,rid in orion_sheet]})
with (OUT/'downloads/results.csv').open('w',newline='') as fp:
    writer=csv.writer(fp);metric_keys=sorted(all_metric_keys)
    writer.writerow(['run_id','model','dataset','marker','split','protocol','tiles','rankable','training_hours']+metric_keys)
    for r in index_runs:
        for e in r['evaluations']:writer.writerow([r['id'],r['name'],e['dataset'],e['marker'],e['split'],e['protocol'],e['n'],e['rankable'],r.get('training_hours')]+[e['metrics'].get(k) for k in metric_keys])
print(json.dumps({'runs':len(runs),'model_families':len(models),'datasets':len(datasets),'evaluations':sum(len(r['evaluations']) for r in runs.values()),'runs_with_curves':sum(bool(r['curves']) for r in runs.values()),'runs_with_images':sum(bool(r['gallery']) for r in runs.values()),'runs_with_distributions':sum(bool(r['distributions']) for r in runs.values()),'audit':audit},indent=2),flush=True)
