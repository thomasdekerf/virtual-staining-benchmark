"""Validate scientific snapshot consistency and static references before deployment."""
import hashlib, json, math, re
from followup_results import verify_gpu_snapshot
from publication_validation import validate_publication_scope
from datetime import datetime
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'dist'
index=json.loads((root/'data/index.json').read_text())
publication_validation=validate_publication_scope(root,index)
ids=[r['id'] for r in index['runs']]
assert len(ids)==len(set(ids)), 'Duplicate run IDs'
assert {d['id'] for d in index['datasets']}=={'orion','BCI','MIST','IHC4BC','finland','skin','her2match','hemit_dataset'}, 'Missing registered dataset'
assert index['audit']['model_priority_rows']==39, 'Workbook model inventory changed; reconcile coverage'
assert index['audit']['workbook_orion_rows_accounted'], 'Unaccounted workbook ORION rows'
assert index['audit']['breast_runs']==130, 'Incomplete breast campaign'
assert index['audit']['dataset_result_rows_verified']==360, 'Incomplete workbook reconciliation'
evaluations=images=curves=0
run_details={}
for entry in index['runs']:
    run=json.loads((root/'data/runs'/f'{entry["id"]}.json').read_text())
    run_details[run['id']]=run
    assert run['id']==entry['id']
    eids=[e['id'] for e in run['evaluations']]
    assert len(eids)==len(set(eids))
    for e in run['evaluations']:
        assert e['dataset'] in [d['id'] for d in index['datasets']]
        assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in e['metrics'].values())
        if e['partial'] if 'partial' in e else False: assert not e['rankable'], 'Partial evaluation was ranked'
        if e['split'] in ['monitor','pilot']: assert not e['rankable'], 'Provisional evaluation was ranked'
        if e['rankable']: assert e['n'] and e['n']>0
        if run['task']=='unstained' and e['rankable']:
            assert e['split'] in ['val','test'] and not e.get('partial')
            assert e['checkpoint_sha256'] and e['manifest_sha256']
            assert e['evidence']['file'].endswith('/complete.json'), 'Unstained rank lacks completion evidence'
            assert set(e['nuclei_protocol'])=={'stardist','stardist2x','hovernet','cellpose_sam'}
            for detector in e['nuclei_protocol']:
                assert detector+'_nuclei_f1' in e['metrics'], 'Missing nuclei result'
                precision=detector+'_nuclei_precision'
                if e.get('coverage',{}).get(precision)==0:
                    assert precision not in e['metrics'], 'Undefined precision was replaced by a score'
        assert all(k in index['metrics'] for k in e['metrics']), 'Metric missing from glossary'
        evaluations+=1
    for tag,points in run['curves'].items():
        assert all(math.isfinite(x) and math.isfinite(y) for x,y in points)
        assert all(points[i][0]<points[i+1][0] for i in range(len(points)-1)), 'Curve steps not monotonic'
        assert (root/'downloads'/f'{run["id"]}_curves.csv').exists()
        curves+=1
    for image in run['gallery']:
        assert (root/image['gt']).is_file() and (root/image['pred']).is_file(), 'Broken paired image reference'
        images+=1
    for split,stats in run['distributions'].items():
        assert (root/'downloads'/f'{run["id"]}_{split}_per_tile.csv').is_file()
        if run['task']=='unstained':
            evaluation=next((e for e in run['evaluations'] if e['split']==split),None)
            if evaluation:
                assert stats['psnr']['n']==evaluation['n'], 'Subset replaced full-split distributions'
        for metric,s in stats.items():
            assert s['n']>0 and s['quantiles']==sorted(s['quantiles'])
            assert all(math.isfinite(x) and 0<=d<=1 for x,d in s['density'])

unstained_rows=sum(e['rankable'] and r['task']=='unstained' for r in index['runs'] for e in r['evaluations'])
assert unstained_rows==index['audit']['unstained_result_rows_verified'], 'Unreconciled unstained scores'

# Numerical H&E-to-mIF is a separate task from ORION's reverse translation.
from hemit_dataset_results import CHANNELS, COUNTS, FINAL_MODELS, MODELS
hemit=[r for r in index['runs'] if 'hemit_dataset' in r['datasets']]
assert len(hemit)==len(MODELS) and {r['model'] for r in hemit}==set(MODELS)
assert {r['model'] for r in hemit if r['evaluation_complete']}==set(FINAL_MODELS)
for entry in hemit:
    r=run_details[entry['id']]
    assert r['task']=='he_to_mif' and r['target_channels']==list(CHANNELS)
    assert r['normalization']['physical_intensity_calibration'] is False
    assert r['normalization']['canonical_reindex']==[2,1,0]
    assert not r['gallery'] and not r['distributions']
    assert r['curves'] and r['curve_protocol']['inherited_parent_deduplicated']
    assert r['clinical_score_evaluation'] is False
    if r['model'] not in FINAL_MODELS:
        assert not r['evaluations'] and not r['training_complete']
    else:
        assert r['native_adequacy_verified'] and len(r['evaluations'])==2
        if r['curve_axis']=='epoch':assert r['selected_completed_epoch']==r['position']['epoch']+1
    for e in r['evaluations']:
        assert e['rankable'] and e['n']==COUNTS[e['split']] and e['target_space']=='mif'
        assert all(k.startswith('mif_') for k in e['metrics'])
        assert e['checkpoint_sha256']==r['checkpoint_sha256'] and e['manifest_sha256']
        for channel in CHANNELS:
            row=e['fluorescence_channels'][channel];key=f'mif_{channel}_pcc'
            assert row['pcc_total_tiles']==e['n']
            assert row['pcc_defined_tiles']+row['pcc_undefined_tiles']==e['n']
            assert e['coverage'][key]==row['pcc_defined_tiles']
            if key in e['metrics']:assert e['metrics'][key]==row['pcc_mean_defined_tiles']
assert all(not row['active_objective'] and row['state']=='user_deferred' for row in index['deferred_datasets'])
extension=run_details['continuation__bbdm_finland_HE_20261125']
assert extension['parent_run']=='balance__bbdm_finland_HE' and extension['gate_passed']
assert not extension['evaluations'] and not extension['training_complete']
assert extension['new_optimizer_updates']==extension['last_step']-100000>0
assert all(x>100000 for points in extension['curves'].values() for x,y in points)
bci_new=[r for r in index['runs'] if r['id'].startswith('bci_published__')]
assert len(bci_new)==6 and sum(bool(r['evaluation_complete']) for r in bci_new)==5
for r in bci_new:
    assert r['datasets']==['BCI'] and r['task']=='ihc' and r['marker']=='HER2' and r['gate_passed']
    for e in r['evaluations']:
        if e.get('rankable'):
            assert e['n']=={'val':585,'test':977}[e['split']]
            assert e.get('comparison_protocol_review_sha256'), 'New BCI result lacks common-code equivalence review'

# Independent published-reference anchors catch protocol mixing and rounded values.
unet=json.loads((root/'data/runs/unet_l1_job1320937.json').read_text())
matched=next(e for e in unet['evaluations'] if e['split']=='test' and e['protocol'].startswith('Matched FP32'))
assert abs(matched['metrics']['psnr']-23.7402681356641)<1e-10
assert not any(e['split']=='test' and e['protocol'].startswith('Box SSIM') for e in unet['evaluations'])
pilot=next(e for e in json.loads((root/'data/runs/priority_implicitstainer_orion_production_job1383267.json').read_text())['evaluations'] if e['split']=='pilot')
assert abs(pilot['metrics']['psnr']-22.67256396994569)<1e-10 and pilot['n']==24

# The overview must validate from checked-in public data only: CI has no private
# training workspace, SSH access, evaluator checkpoints or activity receipts.
for filename in ['index.html','app.js','overview.js','styles.css','overview.css',
                 'profile-data.js','learning-explorer.js','learning-explorer.css',
                 'data-laboratory.js','data-laboratory.css']:
    assert (root/filename).is_file(), f'Missing overview asset: {filename}'
html=(root/'index.html').read_text()
for reference in re.findall(r'(?:src|href)=[\"\']([^\"\']+)[\"\']',html):
    if reference.startswith(('http:','https:','data:','#')): continue
    assert (root/reference.split('?')[0]).is_file(), f'Broken static reference: {reference}'
assert re.search(r"from\s+['\"]\./overview\.js(?:[?#][^'\"]*)?['\"]",(root/'app.js').read_text()), 'Overview module is not imported'
assert 'overview.css' in html, 'Overview stylesheet is not loaded'
for stylesheet in ['learning-explorer.css','data-laboratory.css']:
    assert stylesheet in html, f'Explorer stylesheet is not loaded: {stylesheet}'
for module in root.glob('*.js'):
    for reference in re.findall(r"(?:from\s+|import\s*)['\"](\./[^'\"]+)['\"]",module.read_text()):
        resolved=re.split(r'[?#]',reference,maxsplit=1)[0]
        assert (module.parent/resolved).is_file(), f'Broken JavaScript import in {module.name}: {reference}'
assert (root.parent/'tools/test_explorer_semantics.mjs').is_file(), 'Missing explorer semantic tests'

summary_path=root/'data/visual-summary.json'
summary=json.loads(summary_path.read_text())
assert summary['schema']=='virtual-staining-visual-summary-v1'
assert summary['snapshot']['benchmark']==index['snapshot'], 'Overview uses a different historical snapshot'
assert datetime.fromisoformat(summary['snapshot']['activity'])>=datetime.fromisoformat(index['snapshot'])
assert not re.search(r'/(?:Users|sofia|user|home|mnt)/',summary_path.read_text()), 'Private filesystem path in public chart data'
entries=index['runs']
if index.get('gpu_activity'):
    verify_gpu_snapshot(index['gpu_activity'])
    assert summary.get('gpu_activity')==index['gpu_activity'], 'GPU overview uses stale execution data'
    assert json.loads((root/'data/gpu-activity.json').read_text())==index['gpu_activity'], 'GPU download differs from the shown snapshot'
ranked=lambda r: any(e.get('rankable') for e in r['evaluations'])
tested=lambda r: any(e.get('rankable') and e['split']=='test' for e in r['evaluations'])
scale=summary['scale']
expected={'run_records':len(entries),'catalog_models':len(index['models']),
    'model_families_with_runs':len({r['model'] for r in entries}),
    'evaluation_records':sum(len(r['evaluations']) for r in entries),
    'rankable_evaluations':sum(bool(e.get('rankable')) for r in entries for e in r['evaluations']),
    'evaluated_runs':sum(ranked(r) for r in entries),'tested_runs':sum(tested(r) for r in entries),
    'scalar_series':sum(len(r.get('curve_counts',{})) for r in entries),
    'scalar_points':sum(sum(r.get('curve_counts',{}).values()) for r in entries),
    'paired_fields':sum(r.get('image_count',0) for r in entries),
    'datasets_published':len(index['datasets']),'her2match_tasks':len(summary['activity']['her2match'])}
for key,value in expected.items(): assert scale[key]==value, f'Inflated or stale overview count: {key}'
compute=[r['gpu_hours'] for r in entries if isinstance(r.get('gpu_hours'),(int,float)) and math.isfinite(r['gpu_hours'])]
assert scale['compute_runs']==len(compute) and math.isclose(scale['recorded_gpu_hours'],math.fsum(compute),rel_tol=1e-12)
dataset_ids={d['id'] for d in summary['datasets']}
assert len(dataset_ids)==len(summary['datasets'])==scale['datasets_registered']
assert dataset_ids=={d['id'] for d in index['datasets']}
her2=next(d for d in summary['datasets'] if d['id']=='her2match')
assert sum(her2['counts'].values())==her2['pairs']==21172
her2_records=[r for r in entries if 'her2match' in r['datasets']]
assert her2['rankable_evaluations']==sum(bool(e.get('rankable')) and e['dataset']=='her2match' for r in her2_records for e in r['evaluations'])
assert her2['evaluated_runs']==sum(any(e.get('rankable') and e['dataset']=='her2match' for e in r['evaluations']) for r in her2_records)
for row in her2_records:
    for e in row['evaluations']:
        if e.get('rankable'):
            assert e['checkpoint_sha256'] and e['manifest_sha256'] and e['evidence']['file'].endswith('/final_evaluation.json')
            assert e['n']=={'val':3582,'test':5980}[e['split']]
assert {d['id'] for d in summary['coverage']['datasets']}==dataset_ids
for row in summary['coverage']['rows']:
    assert len(row['cells'])==len(dataset_ids) and {c['dataset'] for c in row['cells']}==dataset_ids
    for cell in row['cells']:
        assert 0<=cell['tested_runs']<=cell['evaluated_runs']<=cell['runs']
        if True:
            records=[r for r in entries if r['model']==row['id'] and cell['dataset'] in r['datasets']]
            scored=[r for r in records if any(e.get('rankable') and e['dataset']==cell['dataset'] for e in r['evaluations'])]
            assert cell['runs']==len(records) and cell['evaluated_runs']==len(scored), 'Coverage cell mixes scopes'

cohort=summary['cohort']
assert cohort['models']==len(cohort['runs']) and len({r['id'] for r in cohort['runs']})==len(cohort['runs'])
assert set(cohort['default_runs'])<={r['id'] for r in cohort['runs']}
for metric in cohort['metrics']:
    values=[r['values'][metric['key']] for r in cohort['runs']]
    assert all(math.isfinite(v) for v in values)
    assert (metric['min'],metric['max'])==(min(values),max(values)), 'Radar normalization cohort changed'
    assert metric['direction'] in ['higher','lower'] and metric['coverage_n']==cohort['n']
    for row in cohort['runs']:
        evaluation=next(e for e in run_details[row['id']]['evaluations'] if e['dataset']==cohort['dataset'] and e['split']==cohort['split'] and e['protocol']==cohort['protocol'])
        assert evaluation['rankable'] and evaluation['n']==cohort['n']
        assert evaluation['metric_protocol']==cohort['metric_protocol']
        assert evaluation.get('coverage',{}).get(metric['key'],evaluation['n'])==cohort['n'], 'Radar axis uses a subset of the cohort'
        assert evaluation['manifest_sha256']==cohort['manifest_sha256'] and evaluation['checkpoint_sha256']==row['checkpoint_sha256']
        value=row['values'][metric['key']]
        assert value==evaluation['metrics'][metric['key']] and row['coverage'][metric['key']]==cohort['n']
        low,high=metric['min'],metric['max']
        normalized=.5 if low==high else (value-low)/(high-low)
        if low!=high and metric['direction']=='lower': normalized=1-normalized
        assert math.isclose(row['normalized'][metric['key']],normalized,abs_tol=1e-12), 'Radar direction/scale incorrect'
for series in summary['trajectories']['series']:
    detail=run_details[series['id']]
    trajectory=summary['trajectories']
    assert any(e['dataset']==trajectory['dataset'] and e['split']==trajectory['split'] and e['marker']==trajectory['marker'] and e['n']==trajectory['n'] and e['protocol']==trajectory['protocol'] for e in detail['evaluations']), 'Learning curves mix evaluation cohorts'
    assert series['points']==detail['curves']['val/lpips'], 'Learning plot fabricates or drops measurements'
    assert series['recorded_points']==detail['curve_counts']['val/lpips']==len(series['points'])
    assert series['axis']=='optimizer updates' and all(p[0]>0 for p in series['points'])
    assert all(series['points'][i][0]<series['points'][i+1][0] for i in range(len(series['points'])-1))
for row in summary['activity']['her2match']:
    assert row['step']>=0
    actual=next((r for r in entries if r['id']=='her2match__'+row['id']+'_seed42'),None)
    assert row['metrics_rankable']==bool(actual and any(e.get('rankable') and e['split']=='test' for e in actual['evaluations']))
    assert 'metrics' not in row, 'Provisional HER2Match values entered scored chart data'
for dataset in summary['datasets']:
    mine=[r for r in entries if dataset['id'] in r['datasets']]
    tested={r['model'] for r in mine if any(e.get('rankable') and e['dataset']==dataset['id'] and e['split']=='test' for e in r['evaluations'])}
    trained={r['model'] for r in mine if any(e.get('rankable') for e in r['evaluations']) or sum(r.get('curve_counts',{}).values()) or (r.get('last_step') or 0)>0}
    assert dataset['tested_model_families']==len(tested) and dataset['trained_model_families']==len(trained), 'Model-family counts were inflated by variants'
proof=json.loads((root/'data/refresh-provenance.json').read_text())
assert proof['original_ranked_rows_preserved']==index['audit']['original_ranked_rows_preserved']
assert len(proof['original_score_fingerprints'])==proof['original_ranked_rows_preserved']
for binding in proof['original_score_fingerprints']:
    e=next(e for e in run_details[binding['run']]['evaluations'] if e['id']==binding['evaluation'])
    values={k:e['metrics'][k] for k in binding['metric_keys']}
    assert hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()==binding['metrics_sha256'], 'Original scored result changed'
for task in (summary.get('coverage_balance') or {}).get('tasks',[]):
    registered=run_details.get('balance__'+task['id'])
    if registered:
        scored=[e for e in registered['evaluations'] if e.get('rankable')]
        assert task['evaluated']==any(e['split']=='test' for e in scored), 'Coverage task and result disagree'
        assert task['last_step']==registered.get('last_step',0) and task['gate_passed']==registered.get('gate_passed',False)
        if scored:
            assert task['gate_passed'] and task['training_complete'], 'Coverage result lacks native resumed training proof'
            assert {e['split'] for e in scored}=={'val','test'}
            assert all(e['checkpoint_sha256'] and e['manifest_sha256'] and e['evidence']['sha256'] for e in scored)
        else:
            assert not task['evaluated'], 'Unfinished coverage task acquired invented scores'
for source in summary['provenance']:
    assert re.fullmatch(r'[0-9a-f]{64}',source['sha256']), 'Malformed evidence fingerprint'
    if source.get('href'):
        path=root/source['href']
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256'], 'Changed public chart source'
# Submitted/queued training can fill a coverage entry, but never a scored cell.
if index.get('her2match_campaign'):
    from her2_campaigns import TARGET_MODELS
    campaign=index['her2match_campaign']
    assert campaign==summary.get('her2match_campaign'), 'HER2 campaign overview is stale'
    assert campaign==json.loads((root/'data/her2match-campaign.json').read_text()), 'HER2 campaign download differs'
    assert set(campaign['target_models'])==set(TARGET_MODELS) and len(campaign['rows'])==len(TARGET_MODELS)
    assert len({row['model'] for row in campaign['rows']})==len(TARGET_MODELS)
    assert sum(campaign['counts'].values())==len(TARGET_MODELS) and campaign['clinical_score_evaluation'] is False
    for row in campaign['rows']:
        mine=[run for run in entries if run['model']==row['model'] and 'her2match' in run['datasets']]
        tested=any(e.get('rankable') and e['dataset']=='her2match' and e['split']=='test' for run in mine for e in run['evaluations'])
        assert row['test_score_available']==tested and (row['status']=='evaluated')==tested
        if row['run_id']:assert row['run_id'] in run_details
        if row['status']=='blocked':assert row['reason'] and not tested
    for run in entries:
        if run['id'].startswith(('her2match_complete__','continuation__','architecture__')) and any(e.get('rankable') for e in run['evaluations']):
            assert run['gate_passed'] and run['training_complete'], 'New campaign result bypassed native gate'
            assert run['native_gate_evidence']['world_size']==4
            assert run['native_gate_evidence']['actual_tensor_update'] and run['native_gate_evidence']['positive_learning_rate']
            assert re.fullmatch('[0-9a-f]{64}',run['native_gate_evidence']['sha256'])
        if run.get('evaluation_reuses_parent'):
            assert not any(e.get('rankable') for e in run['evaluations']), 'Unchanged selected parent was ranked twice'
            parent=run_details[run['evaluation_reference_run']]
            assert any(e.get('rankable') and e.get('checkpoint_sha256')==run['checkpoint_sha256'] for e in parent['evaluations'])

# The matched gallery is independently bound to selected checkpoints and tissue fields.
from validate_prediction_gallery import validate_gallery
gallery_validation=validate_gallery(json.loads((root/'data/prediction-gallery.json').read_text()),root,index)
for filename in ['prediction-gallery.js','prediction-gallery.css']:
    assert (root/filename).is_file(), 'Missing gallery interface asset'
print(json.dumps({'runs':len(ids),'evaluations':evaluations,'scalar_series':curves,'paired_fields':images,'matched_gallery_fields':gallery_validation['fields'],'matched_gallery_predictions':gallery_validation['predictions'],'workbook_cells_verified':index['audit']['verified_cells'],'overview_cohort_runs':cohort['models'],'overview_coverage_cells':sum(len(r['cells']) for r in summary['coverage']['rows']),'published_models':publication_validation['models'],'validation':'passed'}))
