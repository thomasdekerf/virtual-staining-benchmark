const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const format=value=>finite(value)?Math.abs(value)>=1000?value.toLocaleString('en-GB',{maximumFractionDigits:1}):value.toFixed(Math.abs(value)>=10?2:4):'—';
const count=value=>Number(value||0).toLocaleString('en-GB');
const directionSign=direction=>['↑','higher'].includes(direction)?1:['↓','lower'].includes(direction)?-1:null;
const $=(s,root)=>root.querySelector(s);
const $$=(s,root)=>[...root.querySelectorAll(s)];
const states=new WeakMap();

/** A comparison group retains metric protocol, sample size and manifest identity. */
export function cohortKey(e,run=null){return JSON.stringify([e.dataset||'',e.marker||'',e.split||'',e.protocol,e.n,e.comparison_protocol||e.metric_protocol||'',e.manifest_sha256||'',e.crop_size??run?.crop_size??'']);}

export function cohortRows(index,filters={}){
  const candidates=index.runs.flatMap(run=>(run.evaluations||[]).filter(e=>e.rankable&&!e.partial&&
    (!filters.dataset||e.dataset===filters.dataset)&&(!filters.marker||(e.marker||run.marker)===filters.marker)&&
    (!filters.split||e.split===filters.split)&&(!filters.protocol||e.protocol===filters.protocol))
    .map(e=>({run,evaluation:e})));
  const groups=new Map();
  for(const row of candidates){const key=cohortKey(row.evaluation,row.run);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}
  const key=filters.cohort_key||[...groups].sort((a,b)=>b[1].length-a[1].length||a[0].localeCompare(b[0]))[0]?.[0];
  return (groups.get(key)||[]).map(({run,evaluation})=>({id:run.id,model:run.model,name:run.name,
    label:run.name+(run.variant?' · '+run.variant:''),run,evaluation,
    values:{...evaluation.metrics,
      training_hours:run.compute_complete===false||run.compute_parent_run?undefined:run.training_hours,
      gpu_hours:run.compute_complete===false||run.compute_parent_run?undefined:run.gpu_hours}}));
}

export function normalizeValue(value,min,max,direction){
  if(!finite(value)||!finite(min)||!finite(max)||directionSign(direction)===null)return null;
  if(min===max)return .5;
  const normalized=(value-min)/(max-min);
  return directionSign(direction)===1?normalized:1-normalized;
}

export function paretoFront(rows,xKey,yKey,xDirection,yDirection){
  const sx=directionSign(xDirection),sy=directionSign(yDirection);
  if(sx===null||sy===null)return [];
  const valid=rows.filter(r=>finite(r.values[xKey])&&finite(r.values[yKey]));
  return valid.filter(a=>!valid.some(b=>b!==a&&sx*b.values[xKey]>=sx*a.values[xKey]&&sy*b.values[yKey]>=sy*a.values[yKey]&&
    (sx*b.values[xKey]>sx*a.values[xKey]||sy*b.values[yKey]>sy*a.values[yKey])));
}

export function pearsonPairs(rows,xKey,yKey){
  const pairs=rows.map(r=>[r.values[xKey],r.values[yKey]]).filter(p=>p.every(finite));
  const n=pairs.length;
  if(n<3)return {n,r:null};
  const mx=pairs.reduce((a,p)=>a+p[0],0)/n,my=pairs.reduce((a,p)=>a+p[1],0)/n;
  let xx=0,yy=0,xy=0;
  for(const [x,y] of pairs){xx+=(x-mx)**2;yy+=(y-my)**2;xy+=(x-mx)*(y-my);}
  return {n,r:xx===0||yy===0?null:Math.max(-1,Math.min(1,xy/Math.sqrt(xx*yy)))};
}

const metadata=(index,key)=>key==='training_hours'?{label:'Training wall time',unit:'hours',direction:'↓'}:
  key==='gpu_hours'?{label:'Recorded GPU allocation',unit:'GPU hours',direction:'↓'}:
  index.metrics[key]||{label:key.replaceAll('_',' '),unit:'recorded units',direction:'—'};
const option=(value,label,current)=>`<option value="${esc(value)}" ${value===current?'selected':''}>${esc(label)}</option>`;
const runHref=row=>`#/run/${encodeURIComponent(row.id)}?eval=${encodeURIComponent(row.evaluation.id)}`;
const modes={scatter:['The trade-off explorer','Plot any two measured dimensions. Follow quality, compute and the compromises between them.'],
  heatmap:['Every model. Every selected metric.','A direction-aligned matrix makes patterns across many models easier to compare.'],
  parallel:['Follow the metric profile','Trace many models across the same axes without stacking radar polygons.'],
  correlation:['Which metrics move together?','Correlations across recorded run means reveal agreement and disagreement between metrics.']};

function availableGroups(index,state){
  const groups=new Map();
  for(const run of index.runs)for(const e of run.evaluations||[]){
    if(!e.rankable||e.partial||e.dataset!==state.dataset||(e.marker||run.marker)!==state.marker||e.split!==state.split)continue;
    const key=cohortKey(e,run);if(!groups.has(key))groups.set(key,{key,e,crop:e.crop_size??run.crop_size,n:0});groups.get(key).n++;
  }
  return [...groups.values()].sort((a,b)=>(a.e.protocol==='Full production evaluation'?-1:0)-(b.e.protocol==='Full production evaluation'?-1:0)||b.n-a.n||a.key.localeCompare(b.key));
}

function prepare(index,state){
  const datasets=index.datasets.filter(d=>index.runs.some(r=>r.evaluations.some(e=>e.rankable&&e.dataset===d.id)));
  if(!datasets.some(d=>d.id===state.dataset))state.dataset=datasets[0]?.id;
  const markers=[...new Set(index.runs.flatMap(r=>r.evaluations.filter(e=>e.rankable&&e.dataset===state.dataset).map(e=>e.marker||r.marker)))].sort();
  if(!markers.includes(state.marker))state.marker=markers[0];
  const splits=['test','val'].filter(s=>index.runs.some(r=>r.evaluations.some(e=>e.rankable&&e.dataset===state.dataset&&(e.marker||r.marker)===state.marker&&e.split===s)));
  if(!splits.includes(state.split))state.split=splits[0];
  const groups=availableGroups(index,state);
  if(!groups.some(g=>g.key===state.cohort_key))state.cohort_key=groups[0]?.key;
  const rows=cohortRows(index,state);
  const keys=[...new Set(rows.flatMap(r=>Object.keys(r.values).filter(k=>finite(r.values[k]))))].sort((a,b)=>metadata(index,a).label.localeCompare(metadata(index,b).label));
  if(!keys.includes(state.x))state.x=keys.includes('psnr')?'psnr':keys[0];
  if(!keys.includes(state.y))state.y=keys.includes('lpips')?'lpips':keys[1]||keys[0];
  return {datasets,markers,splits,groups,rows,keys};
}

function controls(index,state,context){
  return `<div class="lab-filters"><label class="field">Dataset<select data-lab="dataset">${context.datasets.map(d=>option(d.id,d.name,state.dataset)).join('')}</select></label>
  <label class="field">Target stain<select data-lab="marker">${context.markers.map(m=>option(m,m,state.marker)).join('')}</select></label>
  <label class="field">Evaluation split<select data-lab="split">${context.splits.map(s=>option(s,s==='test'?'Held-out test':'Validation',state.split)).join('')}</select></label>
  <label class="field lab-protocol">Matched comparison group<select data-lab="cohort_key">${context.groups.map(g=>option(g.key,`${g.e.protocol} · ${count(g.e.n)} tiles${g.crop?' · '+g.crop+'px':''} · ${g.n} run records`,state.cohort_key)).join('')}</select></label></div>
  <p class="lab-cohort-note"><strong>${context.rows.length} comparable run records.</strong> Dataset, stain, split, metric protocol, recorded sample count, crop size and manifest are held fixed. Training recipes and compute budgets may differ. Metric-specific finite tile coverage can differ; inspect individual results.</p>`;
}

function axisLabel(index,key){const m=metadata(index,key);return `${m.label} ${m.direction}${m.unit?' · '+m.unit:''}`;}
const color=(index,i)=>`hsl(${(i*137.508+225)%360} 58% 46%)`;
function rowColor(index,rows,row,state){
  if(state.color==='recipe')return row.run.datasets.length>1?'#b83475':'#4159dc';
  const models=[...new Set(rows.map(r=>r.model))].sort();return color(index,models.indexOf(row.model));
}

function scatter(index,state,rows,keys){
  const valid=rows.filter(r=>finite(r.values[state.x])&&finite(r.values[state.y]));
  const logValid=valid.length&&valid.every(r=>r.values[state.x]>0);
  if(state.log&&!logValid)state.log=false;
  const xmeta=metadata(index,state.x),ymeta=metadata(index,state.y);
  const canPareto=directionSign(xmeta.direction)!==null&&directionSign(ymeta.direction)!==null&&state.x!==state.y;
  const frontier=canPareto?paretoFront(valid,state.x,state.y,xmeta.direction,ymeta.direction):[];
  const frontierIds=new Set(frontier.map(r=>r.id));
  const narrow=typeof matchMedia!=='undefined'&&matchMedia('(max-width:700px)').matches;
  const w=narrow?470:980,h=narrow?420:460,left=82,right=30,top=35,bottom=80;
  let graph='<div class="empty">No run has both selected measurements. Choose another axis.</div>';
  if(valid.length){
    const transform=v=>state.log?Math.log10(v):v;
    const xs=valid.map(r=>transform(r.values[state.x])),ys=valid.map(r=>r.values[state.y]);
    const extent=values=>{let a=Math.min(...values),b=Math.max(...values);const margin=(b-a||Math.max(Math.abs(a)*.1,.01))*.08;return [a-margin,b+margin];};
    const [xmin,xmax]=extent(xs),[ymin,ymax]=extent(ys);
    const x=v=>left+(transform(v)-xmin)/(xmax-xmin)*(w-left-right),y=v=>h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom);
    const maxGpu=Math.max(...valid.map(r=>finite(r.values.gpu_hours)?r.values.gpu_hours:0),1);
    const ticks=Array.from({length:5},(_,i)=>{
      const tx=xmin+(xmax-xmin)*i/4,ty=ymin+(ymax-ymin)*i/4;
      return `<line x1="${left}" x2="${w-right}" y1="${y(ty)}" y2="${y(ty)}"/><text x="${left-12}" y="${y(ty)+4}" text-anchor="end">${format(ty)}</text><text x="${left+(w-left-right)*i/4}" y="${h-bottom+27}" text-anchor="middle">${format(state.log?10**tx:tx)}</text>`;
    }).join('');
    const boundary=state.pareto&&canPareto&&frontier.length>1?`<polyline points="${[...frontier].sort((a,b)=>a.values[state.x]-b.values[state.x]).map(r=>x(r.values[state.x])+','+y(r.values[state.y])).join(' ')}" fill="none" stroke="#17253e" stroke-width="1.5" stroke-dasharray="5 5"/>`:'';
    graph=`<svg class="lab-chart lab-scatter" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(xmeta.label)} versus ${esc(ymeta.label)} for ${valid.length} matched run records"><title>Measured model trade-offs</title>${ticks}${boundary}${valid.map(row=>{
      const radius=state.size&&finite(row.values.gpu_hours)?Math.max(3,16*Math.sqrt(Math.max(0,row.values.gpu_hours)/maxGpu)):6;
      const better=state.pareto&&canPareto&&frontierIds.has(row.id);
      const title=`${row.label}; ${xmeta.label}: ${format(row.values[state.x])}; ${ymeta.label}: ${format(row.values[state.y])}${state.size?'; recorded GPU hours: '+format(row.values.gpu_hours):''}${better?'; nondominated among shown records':''}`;
      return `<a class="lab-point" href="${runHref(row)}" data-run="${esc(row.id)}" aria-label="${esc(title)}"><circle cx="${x(row.values[state.x])}" cy="${y(row.values[state.y])}" r="${radius}" fill="${rowColor(index,rows,row,state)}" fill-opacity=".78" stroke="${better?'#17253e':'white'}" stroke-width="${better?2.5:1.5}"><title>${esc(title)}</title></circle>${state.labels?`<text class="lab-point-label" x="${x(row.values[state.x])+radius+4}" y="${y(row.values[state.y])-radius-3}">${esc(row.name)}</text>`:''}</a>`;
    }).join('')}<text class="lab-axis-label" x="${(left+w-right)/2}" y="${h-20}" text-anchor="middle">${esc(xmeta.label)} ${xmeta.direction}${state.log?' (log scale)':''}</text><text class="lab-axis-label" x="${left}" y="18">${esc(ymeta.label)} ${ymeta.direction}</text></svg>`;
  }
  return `<div class="lab-presets"><span>Try a question</span><button data-preset="quality">Pixel error vs perception</button><button data-preset="compute">Quality vs compute</button><button data-preset="embedding">Structure vs embedding</button><button data-preset="nuclei">Perception vs nuclei</button></div>
    <div class="lab-axis-controls"><label class="field">Horizontal axis<select data-lab="x">${keys.map(k=>option(k,axisLabel(index,k),state.x)).join('')}</select></label><label class="field">Vertical axis<select data-lab="y">${keys.map(k=>option(k,axisLabel(index,k),state.y)).join('')}</select></label><label class="field">Colour<select data-lab="color">${option('family','Model family',state.color)}${option('recipe','Separate / pooled training',state.color)}</select></label></div>
    <div class="lab-switches"><label><input type="checkbox" data-lab-check="labels" ${state.labels?'checked':''}>Label all points</label><label><input type="checkbox" data-lab-check="size" ${state.size?'checked':''}>Bubble size indicates recorded GPU hours</label><label><input type="checkbox" data-lab-check="log" ${state.log?'checked':''} ${!logValid?'disabled':''}>Log horizontal axis</label><label><input type="checkbox" data-lab-check="pareto" ${state.pareto?'checked':''} ${!canPareto?'disabled':''}>Show Pareto frontier</label></div>
    <div class="lab-plot-wrap">${graph}</div><div class="lab-hover" id="lab-hover" role="status" aria-live="polite">Hover or focus a point to inspect it. Click to open its complete run.</div>
    <div class="lab-chart-notes"><span><strong>${valid.length}</strong> plotted / ${rows.length} cohort records; ${rows.length-valid.length} missing an axis measurement</span>${state.pareto&&canPareto?`<span><strong>${frontier.length}</strong> nondominated records among the plotted points</span>`:''}${state.size?'<span>Bubble area follows recorded allocation above a 3 px visibility floor; missing allocation uses the standard point size.</span>':''}</div>
    <p class="visual-caption">Arrows show the recorded better direction. A Pareto record has no other plotted record at least as good on both axes and strictly better on one; this is a numerical trade-off, not a significance test. Connecting frontier points does not imply attainable intermediate models. Compute axes and bubble sizes exclude incomplete allocation histories and continuations whose parent cost is missing. Recorded time remains available on the run page. ${state.labels?'Labels can overlap in a dense view; focus individual points for exact identification.':''}</p>
    ${resultTable(index,valid,[state.x,state.y],state)}`;
}

function chosenKeys(state,keys){return state.metrics.filter(k=>keys.includes(k));}
function metricPicker(index,state,keys){
  return `<details class="lab-metric-picker"><summary>${chosenKeys(state,keys).length} selected metric axes · choose measurements</summary><div class="lab-picker-toolbar"><input id="lab-metric-search" type="search" placeholder="Find a metric…" aria-label="Find a metric axis"><button data-metrics="suggested">Suggested metrics</button><button data-metrics="all">All available</button><button data-metrics="clear">Clear</button></div><div class="lab-metric-options">${keys.map(k=>`<label data-metric-option="${esc(k+' '+metadata(index,k).label).toLowerCase()}"><input type="checkbox" data-lab-metric="${esc(k)}" ${state.metrics.includes(k)?'checked':''}>${esc(axisLabel(index,k))}</label>`).join('')}</div></details>`;
}
function normalized(rows,key,index){
  const vals=rows.map(r=>r.values[key]).filter(finite);
  const min=Math.min(...vals),max=Math.max(...vals),direction=metadata(index,key).direction;
  return row=>normalizeValue(row.values[key],min,max,direction);
}
function sortedRows(rows,state,index){
  return [...rows].sort((a,b)=>{
    if(state.sort==='name')return a.name.localeCompare(b.name)||a.label.localeCompare(b.label);
    const x=a.values[state.sort],y=b.values[state.sort];
    if(!finite(x))return finite(y)?1:0;if(!finite(y))return -1;
    return (directionSign(metadata(index,state.sort).direction)===1?y-x:x-y)||a.label.localeCompare(b.label);
  });
}
function sortControl(index,state,keys){return `<label class="field lab-sort">Order run records<select data-lab="sort">${option('name','Model name',state.sort)}${keys.map(k=>option(k,axisLabel(index,k),state.sort)).join('')}</select></label>`;}
function heatmap(index,state,rows,keys){
  const metrics=chosenKeys(state,keys),ordered=sortedRows(rows,state,index),scales=new Map(metrics.map(k=>[k,normalized(rows,k,index)]));
  if(!metrics.length)return metricPicker(index,state,keys)+'<div class="empty">Choose a metric to build the matrix.</div>';
  return `${metricPicker(index,state,keys)}<div class="lab-matrix-toolbar">${sortControl(index,state,keys)}<span class="lab-heat-legend">Weaker in cohort <i></i> Stronger in cohort</span></div>
    <div class="lab-matrix-scroll"><table class="lab-heatmap"><caption class="sr-only">Raw metric values, coloured by direction-aligned normalized values within this cohort.</caption><thead><tr><th>Model / recipe</th>${metrics.map(k=>`<th>${esc(metadata(index,k).label)} ${metadata(index,k).direction}<small>${esc(metadata(index,k).unit||'')}</small></th>`).join('')}</tr></thead><tbody>${ordered.map(row=>`<tr><th scope="row"><a href="${runHref(row)}">${esc(row.name)}</a><small>${esc(row.run.variant||row.id)}</small></th>${metrics.map(k=>{
      const value=row.values[k],score=scales.get(k)(row),background=score===null?'#f0f3f8':`hsl(231 55% ${96-score*45}%)`,foreground=score!==null&&score>.58?'white':'#263752';
      return `<td style="background:${background};color:${foreground}" title="${esc(row.label+'; '+metadata(index,k).label+': '+format(value)+'; '+(score===null?'unmeasured or no better direction':Math.round(score*100)+' / 100 relative within cohort')+'; finite tile coverage '+(row.evaluation.coverage?.[k]??row.evaluation.n))}">${format(value)}</td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div><p class="visual-caption">Numbers remain in original units. Colour scales separately within each metric, across all records in the selected cohort; changing row order does not change the scale. Missing values and metrics without a defined better direction stay neutral. This is not an overall score.</p>`;
}

function parallel(index,state,rows,keys){
  const metrics=chosenKeys(state,keys),scales=new Map(metrics.map(k=>[k,normalized(rows,k,index)]));
  const directional=metrics.filter(k=>directionSign(metadata(index,k).direction)!==null);
  if(directional.length<2)return metricPicker(index,state,keys)+'<div class="empty">Choose at least two metrics with a defined better direction.</div>';
  const w=Math.max(760,directional.length*155),h=425,left=85,right=85,top=75,bottom=65;
  const x=i=>left+i/(directional.length-1)*(w-left-right),y=v=>h-bottom-v*(h-top-bottom);
  const chart=`<svg class="lab-chart lab-parallel" style="min-width:${w}px" viewBox="0 0 ${w} ${h}" role="img" aria-label="Direction-aligned metric profiles across ${rows.length} matched run records"><title>Parallel metric profiles</title>${[0,.25,.5,.75,1].map(v=>`<text x="${left-28}" y="${y(v)+4}" text-anchor="end">${Math.round(v*100)}</text>`).join('')}${directional.map((k,i)=>{
    const label=metadata(index,k).label;
    return `<line x1="${x(i)}" x2="${x(i)}" y1="${top}" y2="${h-bottom}"/><text class="lab-axis-label" x="${x(i)}" y="${top-30}" text-anchor="middle">${esc(label.length>18?label.slice(0,16)+'…':label)} ${metadata(index,k).direction}<title>${esc(label)}</title></text>`;
  }).join('')}${rows.map(row=>{
    let segments=[],segment=[];
    directional.forEach((k,i)=>{const value=scales.get(k)(row);if(value===null){if(segment.length)segments.push(segment);segment=[];}else segment.push([x(i),y(value)]);});if(segment.length)segments.push(segment);
    return `<a class="lab-series" href="${runHref(row)}" data-run="${esc(row.id)}" aria-label="${esc(row.label)}"><title>${esc(row.label)}</title>${segments.map(s=>s.length>1?`<polyline points="${s.map(p=>p.join(',')).join(' ')}" fill="none" stroke="${rowColor(index,rows,row,state)}" stroke-width="2"/>`:`<circle cx="${s[0][0]}" cy="${s[0][1]}" r="4" fill="${rowColor(index,rows,row,state)}"/>`).join('')}</a>`;
  }).join('')}</svg>`;
  return `${metricPicker(index,state,keys)}<div class="lab-parallel-legend"><span>Each line is one run record. Focus or hover to isolate a profile; click to inspect.</span><span>0 = weaker · 100 = stronger within this cohort</span></div><div class="lab-plot-wrap">${chart}</div><div class="lab-hover" id="lab-hover" role="status" aria-live="polite">Trace a line across the axes to inspect a model’s trade-offs.</div><p class="visual-caption">Each axis uses fixed min/max scaling over the selected cohort, aligned so higher means better. Gaps remain gaps; missing measurements are not connected or replaced with zero. ${metrics.length-directional.length} selected axes without a better direction are omitted. Scroll horizontally for a large metric set. Lines do not represent trajectories or changes over time.</p>${resultTable(index,rows,directional,state)}`;
}

function correlation(index,state,rows,keys){
  const metrics=chosenKeys(state,keys);
  if(!metrics.length)return metricPicker(index,state,keys)+'<div class="empty">Choose metrics to compare their recorded run-level associations.</div>';
  return `${metricPicker(index,state,keys)}<div class="lab-correlation-key"><span>Negative</span><i></i><span>Positive</span><span>NA = fewer than three pairs or a constant metric</span></div><div class="lab-matrix-scroll"><table class="lab-correlation"><caption class="sr-only">Pearson correlations across run means, with pairwise number of run records.</caption><thead><tr><th>Metric</th>${metrics.map(k=>`<th>${esc(metadata(index,k).label)}<small>${metadata(index,k).direction}</small></th>`).join('')}</tr></thead><tbody>${metrics.map(x=>`<tr><th scope="row">${esc(metadata(index,x).label)} ${metadata(index,x).direction}</th>${metrics.map(y=>{
      const result=pearsonPairs(rows,x,y),r=result.r,bg=r===null?'#f0f3f8':r<0?`hsl(335 52% ${97-Math.abs(r)*43}%)`:`hsl(231 55% ${97-r*43}%)`,fg=r!==null&&Math.abs(r)>.65?'white':'#263752';
      return `<td style="background:${bg};color:${fg}" title="${esc(metadata(index,x).label+' versus '+metadata(index,y).label+'; Pearson r '+(r===null?'undefined':r.toFixed(6))+'; '+result.n+' paired run records')}"><strong>${r===null?'NA':r.toFixed(2)}</strong><small>n=${result.n}</small></td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div><p class="visual-caption">Pearson r is calculated from the raw recorded run means, using finite pairs independently for every cell. Arrows retain each metric’s better direction; raw correlation signs are not flipped. These are associations across delivered recipes in one cohort, not tile-level correlations, independent repeat-seed uncertainty or causal evidence. Pooled and separate training recipes may be represented.</p>`;
}

function resultTable(index,rows,keys,state){
  return `<details class="lab-exact-table"><summary>Inspect ${rows.length} exact records and export the plotted values</summary><div class="lab-table-tools"><button id="lab-export">Download CSV</button><span>Missing values are left blank.</span></div><div class="lab-matrix-scroll"><table><thead><tr><th>Model / training recipe</th>${keys.map(k=>`<th>${esc(metadata(index,k).label)} ${metadata(index,k).direction}<small>${esc(metadata(index,k).unit||'')}</small></th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr><th scope="row"><a href="${runHref(row)}">${esc(row.name)}</a><small>${esc(row.run.variant||row.id)}</small></th>${keys.map(k=>`<td class="number">${format(row.values[k])}<small>${['gpu_hours','training_hours'].includes(k)?'recorded allocation':count(row.evaluation.coverage?.[k]??row.evaluation.n)+' finite/reference tiles'}</small></td>`).join('')}</tr>`).join('')}</tbody></table></div></details>`;
}

function exportCsv(index,rows,keys){
  const cell=value=>'"'+String(value??'').replaceAll('"','""')+'"';
  const header=['run_id','model','recipe','dataset','stain','split','protocol','reference_tiles',...keys];
  const lines=rows.map(r=>[r.id,r.name,r.run.variant,r.evaluation.dataset,r.evaluation.marker||r.run.marker,r.evaluation.split,r.evaluation.protocol,r.evaluation.n,...keys.map(k=>finite(r.values[k])?r.values[k]:'')]);
  const blob=new Blob([[header,...lines].map(row=>row.map(cell).join(',')).join('\n')],{type:'text/csv;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='virtual-staining-chart-records.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

/** Render a self-contained laboratory inside the overview's owned chart panel. */
export function renderDataLaboratory(container,index,mode='scatter'){
  if(!modes[mode])mode='scatter';
  let state=states.get(index);
  if(!state){state={dataset:'finland',marker:'H&E',split:'test',cohort_key:'',x:'psnr',y:'lpips',color:'family',size:false,labels:false,log:false,pareto:true,sort:'lpips',metrics:['psnr','ssim','lpips','dists','conch_cosine']};states.set(index,state);}
  const draw=()=>{
    const context=prepare(index,state),[title,description]=modes[mode];
    const content=mode==='scatter'?scatter(index,state,context.rows,context.keys):mode==='heatmap'?heatmap(index,state,context.rows,context.keys):mode==='parallel'?parallel(index,state,context.rows,context.keys):correlation(index,state,context.rows,context.keys);
    container.innerHTML=`<section class="visual-panel data-laboratory"><div class="visual-panel-heading"><div><div class="eyebrow">Data laboratory / ${esc(mode)}</div><h2>${title}</h2><p>${description}</p></div><span class="chart-badge">Measured values · matched cohorts</span></div>${controls(index,state,context)}${content}</section>`;
    $$('[data-lab]',container).forEach(el=>el.onchange=()=>{state[el.dataset.lab]=el.value;if(['dataset','marker','split'].includes(el.dataset.lab))state.cohort_key='';draw();});
    $$('[data-lab-check]',container).forEach(el=>el.onchange=()=>{state[el.dataset.labCheck]=el.checked;draw();});
    $$('[data-preset]',container).forEach(el=>el.onclick=()=>{
      const pair={quality:['psnr','lpips'],compute:['gpu_hours','lpips'],embedding:['ssim','conch_cosine'],nuclei:['lpips',context.keys.includes('hovernet_nuclei_f1')?'hovernet_nuclei_f1':'nuclei_f1']}[el.dataset.preset];
      if(pair.every(k=>context.keys.includes(k))){[state.x,state.y]=pair;state.log=false;draw();}else $('#lab-hover',container).textContent='This comparison needs measurements that are absent in the chosen cohort. Choose another dataset or preset.';
    });
    $$('[data-lab-metric]',container).forEach(el=>el.onchange=()=>{state.metrics=el.checked?[...state.metrics,el.dataset.labMetric]:state.metrics.filter(k=>k!==el.dataset.labMetric);draw();const details=$('.lab-metric-picker',container);if(details)details.open=true;});
    $$('[data-metrics]',container).forEach(el=>el.onclick=()=>{state.metrics=el.dataset.metrics==='all'?[...context.keys]:el.dataset.metrics==='clear'?[]:['psnr','ssim','lpips','dists','conch_cosine'].filter(k=>context.keys.includes(k));draw();const details=$('.lab-metric-picker',container);if(details)details.open=true;});
    const search=$('#lab-metric-search',container);if(search)search.oninput=()=>{const q=search.value.toLowerCase();$$('[data-metric-option]',container).forEach(el=>el.hidden=!el.dataset.metricOption.includes(q));};
    $$('.lab-point,.lab-series',container).forEach(el=>{
      const inspect=()=>{const row=context.rows.find(r=>r.id===el.dataset.run);if(!row)return;const keys=mode==='scatter'?[state.x,state.y]:state.metrics;$('#lab-hover',container).textContent=row.label+' · '+keys.map(k=>metadata(index,k).label+': '+format(row.values[k])).join(' · ');};
      el.onpointerenter=inspect;el.onfocus=inspect;
    });
    const download=$('#lab-export',container);if(download)download.onclick=()=>{const keys=mode==='scatter'?[state.x,state.y]:chosenKeys(state,context.keys);const rows=mode==='scatter'?context.rows.filter(r=>keys.every(k=>finite(r.values[k]))):context.rows;exportCsv(index,rows,keys);};
  };
  draw();
  const media=typeof matchMedia==='undefined'?null:matchMedia('(max-width:700px)');
  if(media)media.addEventListener('change',draw);
  return ()=>{if(media)media.removeEventListener('change',draw);};
}
