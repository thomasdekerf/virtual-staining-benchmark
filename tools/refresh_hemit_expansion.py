"""Refresh only HEMIT execution; retain every published evaluation verbatim."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

from architecture_catalog import create_run
from hemit_expansion_results import integrate
from publication_policy import apply_policy
from refresh_live_results import write


def refresh(site,workspace):
    out=site/'dist'
    published=json.loads(subprocess.check_output(['git','-C',str(site),'show','HEAD:dist/data/index.json']))
    data=json.loads((out/'data/index.json').read_text())
    original_ids={r['id'] for r in data['runs']}
    runs={r['id']:(json.loads((out/'data/runs'/f'{r["id"]}.json').read_text()) if 'hemit_dataset'in r['datasets'] else r)for r in data['runs']}
    sources=[]
    def source(path,label):
        p=Path(path);row=dict(file=str(p.relative_to(workspace)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),label=label)
        sources.append(row);return row
    def register(run):runs[run['id']]=run;return run
    names={r['id']:r['name']for r in data['models']}
    def fresh_run(rid,model,dataset,**kw):return create_run(rid,model,dataset,names,**kw)
    campaign=integrate(site,workspace,data,runs,register,fresh_run,source,write)
    if campaign is None:raise ValueError('Complete46-family audit and fresh staged HEMIT evidence required')
    preserved=0
    for run in published['runs']:
        require=next((r for r in runs.values()if r['id']==run['id']),None)
        if require is None:raise ValueError('Removed a published run: '+run['id'])
        for e in run['evaluations']:
            if not e.get('rankable'):continue
            current=next((v for v in require['evaluations']if v['id']==e['id']),None)
            if current!=e:raise ValueError('Changed a published evaluation payload: '+run['id']+'/'+e['id'])
            preserved+=1
    rows=[]
    for rid,run in runs.items():
        if 'hemit_dataset'in run['datasets']:
            write(out/'data/runs'/f'{rid}.json',run)
            row={k:v for k,v in run.items()if k not in {'curves','gallery','distributions','configuration','live_status','research_validation_report'}}
            row.update(has_curves=bool(run.get('curves')),image_count=len(run.get('gallery',[])),distribution_splits=list(run.get('distributions',{})))
        else:row=run
        rows.append(row)
    data.update(runs=rows,snapshot=campaign['snapshot'])
    for model in data['models']:
        mine=[r for r in rows if r['model']==model['id']]
        model.update(runs=len(mine),evaluated_runs=sum(any(e.get('rankable')and e['split']=='test'for e in r['evaluations'])for r in mine),
                     datasets=sorted({e['dataset']for r in mine for e in r['evaluations']if e.get('rankable')}))
    data['audit']['refresh_observed_utc']=campaign['snapshot']
    write(out/'data/index.json',data)
    scope=apply_policy(out)
    if scope['run_records']!=len(rows):raise ValueError('New HEMIT submission lacks explicit publication-policy review')
    report=dict(schema='hemit-execution-refresh-v1',captured_utc=campaign['snapshot'],exported_utc=datetime.now(timezone.utc).isoformat(),
        all_previously_published_rankable_evaluation_payloads_preserved=preserved,new_scores=0,
        additional_submitted_run_records=sorted(set(runs)-original_ids),catalog_methods_accounted=len(campaign['rows']),sources=sources)
    write(out/'data/hemit-expansion-reconciliation.json',report)
    print(json.dumps(dict(preserved_rankable_evaluations=preserved,catalog_methods=len(campaign['rows']),submitted_models=campaign['model_families'],counts=campaign['counts'])))


if __name__=='__main__':
    site=Path(__file__).resolve().parents[1]
    refresh(site,site.parent)
