"""Validate scientific snapshot consistency and static references before deployment."""
import json, math
from pathlib import Path

root=Path(__file__).resolve().parents[1]/'dist'
index=json.loads((root/'data/index.json').read_text())
ids=[r['id'] for r in index['runs']]
assert len(ids)==len(set(ids)), 'Duplicate run IDs'
assert len(index['datasets'])==6, 'Missing dataset'
assert index['audit']['model_priority_rows']==39, 'Workbook model inventory changed; reconcile coverage'
assert index['audit']['workbook_orion_rows_accounted'], 'Unaccounted workbook ORION rows'
assert index['audit']['breast_runs']==130, 'Incomplete breast campaign'
assert index['audit']['dataset_result_rows_verified']==360, 'Incomplete workbook reconciliation'
evaluations=images=curves=0
for entry in index['runs']:
    run=json.loads((root/'data/runs'/f'{entry["id"]}.json').read_text())
    assert run['id']==entry['id']
    eids=[e['id'] for e in run['evaluations']]
    assert len(eids)==len(set(eids))
    for e in run['evaluations']:
        assert e['dataset'] in [d['id'] for d in index['datasets']]
        assert all(isinstance(v,(int,float)) and math.isfinite(v) for v in e['metrics'].values())
        if e['partial'] if 'partial' in e else False: assert not e['rankable'], 'Partial evaluation was ranked'
        if e['split'] in ['monitor','pilot']: assert not e['rankable'], 'Provisional evaluation was ranked'
        if e['rankable']: assert e['n'] and e['n']>0
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
        for metric,s in stats.items():
            assert s['n']>0 and s['quantiles']==sorted(s['quantiles'])
            assert all(math.isfinite(x) and 0<=d<=1 for x,d in s['density'])

# Independent published-reference anchors catch protocol mixing and rounded values.
unet=json.loads((root/'data/runs/unet_l1_job1320937.json').read_text())
matched=next(e for e in unet['evaluations'] if e['split']=='test' and e['protocol'].startswith('Matched FP32'))
assert abs(matched['metrics']['psnr']-23.7402681356641)<1e-10
assert not any(e['split']=='test' and e['protocol'].startswith('Box SSIM') for e in unet['evaluations'])
pilot=next(e for e in json.loads((root/'data/runs/priority_nafnet_orion_production_job1383267.json').read_text())['evaluations'] if e['split']=='pilot')
assert abs(pilot['metrics']['psnr']-23.186044288262085)<1e-10 and pilot['n']==24
print(json.dumps({'runs':len(ids),'evaluations':evaluations,'scalar_series':curves,'paired_fields':images,'workbook_cells_verified':index['audit']['verified_cells'],'validation':'passed'}))
