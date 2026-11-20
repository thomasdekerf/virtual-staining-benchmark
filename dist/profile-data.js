// Pure cohort/normalization helpers shared by the interactive profile explorer.
const finite = value => typeof value === 'number' && Number.isFinite(value);
const preferredMetrics = ['psnr','ssim','lpips','dists','conch_cosine','mae','delta_e76'];

export function buildProfileCohorts(index) {
  const groups = new Map();
  for (const run of index.runs) for (const evaluation of run.evaluations) {
    if (!evaluation.rankable || evaluation.partial || ['pilot','monitor'].includes(evaluation.split)) continue;
    const identity = [evaluation.dataset,evaluation.marker || run.marker,evaluation.split,
      evaluation.protocol,evaluation.n,evaluation.comparison_protocol || evaluation.metric_protocol || '',evaluation.crop_size || run.crop_size || '',
      evaluation.manifest_sha256 || ''];
    const id = JSON.stringify(identity);
    if (!groups.has(id)) {
      const name = index.datasets.find(d=>d.id===evaluation.dataset)?.name || evaluation.dataset;
      groups.set(id,{id,dataset:evaluation.dataset,dataset_name:name,marker:identity[1],split:evaluation.split,
        protocol:evaluation.protocol,n:evaluation.n,metric_protocol:evaluation.comparison_protocol || evaluation.metric_protocol,
        manifest_sha256:evaluation.manifest_sha256,crop_size:identity[6],entries:[],
        label:`${name} · ${identity[1]} · ${evaluation.split} · ${evaluation.protocol} · ${Number(evaluation.n).toLocaleString('en-GB')} tiles`});
    }
    const group=groups.get(id);
    if(!group.entries.some(entry=>entry.run.id===run.id))group.entries.push({run,evaluation});
  }
  return [...groups.values()].filter(g=>g.entries.length>=2).sort((a,b)=>
    (a.split==='test'?0:1)-(b.split==='test'?0:1) || a.label.localeCompare(b.label));
}

export function materializeProfileCohort(index, group, metricKeys) {
  const keys=Object.keys(index.metrics).filter(key=>{
    if(!['↑','↓'].includes(index.metrics[key].direction))return false;
    return group.entries.every(({evaluation:e})=>finite(e.metrics[key]) && (e.coverage?.[key]??e.n)===group.n);
  });
  const availableMetrics=keys.map(key=>({key,label:index.metrics[key].label,
    direction:index.metrics[key].direction==='↑'?'higher':'lower',unit:index.metrics[key].unit||'',
    min:Math.min(...group.entries.map(e=>e.evaluation.metrics[key])),
    max:Math.max(...group.entries.map(e=>e.evaluation.metrics[key])),coverage_n:group.n}));
  const requested=metricKeys||preferredMetrics.filter(key=>keys.includes(key)).slice(0,5);
  const metrics=requested.map(key=>availableMetrics.find(m=>m.key===key)).filter(Boolean);
  const runs=group.entries.map(({run,evaluation})=>{
    const values=Object.fromEntries(availableMetrics.map(m=>[m.key,evaluation.metrics[m.key]]));
    const normalized=Object.fromEntries(availableMetrics.map(m=>[m.key,m.min===m.max ? .5 :
      m.direction==='higher'?(values[m.key]-m.min)/(m.max-m.min):(m.max-values[m.key])/(m.max-m.min)]));
    return {id:run.id,name:run.name,label:run.name+' · '+run.variant,variant:run.variant,model:run.model,
      values,normalized,href:`#/run/${encodeURIComponent(run.id)}?eval=${encodeURIComponent(evaluation.id)}`,
      coverage:Object.fromEntries(availableMetrics.map(m=>[m.key,evaluation.coverage?.[m.key]??evaluation.n])),
      training_hours:run.training_hours,gpu_hours:run.gpu_hours,checkpoint_sha256:evaluation.checkpoint_sha256};
  }).sort((a,b)=>a.name.localeCompare(b.name)||a.id.localeCompare(b.id));
  const defaults=['finland__unet_seed42','finland__nafnet_seed42','finland__restormer_seed42'].filter(id=>runs.some(r=>r.id===id));
  return {...group,entries:undefined,metrics,available_metrics:availableMetrics,runs,models:runs.length,
    default_runs:defaults.length?defaults:runs.slice(0,3).map(r=>r.id)};
}
