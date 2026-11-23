import {buildProfileCohorts, materializeProfileCohort} from './profile-data.js?v=20261120-completed-results';
const esc = value => String(value ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const number = value => Number(value || 0).toLocaleString('en-GB', {maximumFractionDigits: 0});
const date = value => value ? new Date(value).toLocaleDateString('en-GB', {day:'numeric',month:'short',year:'numeric',timeZone:'UTC'}) : 'Not recorded';
const colors = ['#4159dc','#b83475','#147b71','#a95a17','#7844b3','#24749c','#b13e35','#64791b','#745f4a','#9c347e','#30754a','#4b68a0','#965c00','#8a4d62','#006f7e','#5c5b87','#8b5024','#5a713d','#a34451','#3c7283'];
const dash = ['', '8 4', '3 4','12 3 2 3','6 3 2 3'];
let snapshotPromise;
let activeSummary;
let selectedProfiles, focusedMetric = 'lpips', zoomDots = true;
let profileGroups, profileGroupId, profileAxisKeys, radarMode='overlay', activeIndex;
let explorerCleanup, renderSequence=0;
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function cellStyle(cell) {
  if (cell.evaluated_runs) return 'scored';
  if (cell.status && /train|queue|submitted|running|paused/i.test(cell.status)) return 'active';
  return cell.runs ? 'recorded' : 'absent';
}

function miniature(data) {
  return `<div class="atlas-mini" role="group" aria-label="Interactive experiment landscape" aria-describedby="landscape-help">${data.coverage.datasets.map((dataset,di) => `<div class="atlas-mini-row"><span>${esc(dataset.short_name || dataset.name || dataset.id)}</span><div>${data.coverage.rows.map((row,mi) => {
    const cell = row.cells.find(c => c.dataset === dataset.id) || {};
    const label=`${row.name} · ${dataset.name || dataset.id} · ${cell.evaluated_runs||0} scored of ${cell.runs||0} run records · ${cell.active_runs||0} unfinished · ${cell.status||'No recorded work'}${cell.quality_warning?' · Quality warning: '+cell.quality_warning:''}`;
    return `<a class="atlas-mini-cell ${cellStyle(cell)} ${cell.quality_warning?'quality-warning':''}" href="#/benchmark?dataset=${encodeURIComponent(dataset.id)}&model=${encodeURIComponent(row.id)}" tabindex="${di===0&&mi===0?'0':'-1'}" data-landscape-dataset="${esc(dataset.id)}" data-landscape-model="${esc(row.id)}" data-landscape-row="${di}" data-landscape-column="${mi}" title="${esc(label)}" aria-label="${esc(label)}"></a>`;
  }).join('')}</div></div>`).join('')}</div><div class="landscape-legend"><span><i class="scored"></i>Scored evaluation</span><span><i class="active"></i>Training / pending scores</span><span><i class="recorded"></i>Other recorded work</span><span><i class="absent"></i>No record</span><span><i class="quality-warning"></i>Quality issue</span></div><p id="landscape-help" class="landscape-help">Hover or tap a rectangle to identify it. Arrow keys move through the grid; Enter opens matching runs.</p><div class="landscape-inspector" id="landscape-inspector" role="status" aria-live="polite"><strong>Explore a model and dataset</strong><span>Select a rectangle to see its model, dataset and saved evaluation status.</span></div>`;
}

function scaleCard(value, label, note) {
  return `<div class="scope-stat"><strong>${number(value)}</strong><span>${label}</span><small>${note}</small></div>`;
}

function header(data, layout) {
  const s = data.scale;
  return `<section class="research-hero" aria-labelledby="research-title">
    <div class="research-hero-copy"><div class="research-kicker">Virtual staining / Research atlas</div>
      <h1 id="research-title">Virtual staining,<br>measured at scale.</h1>
      <p>Published model families across tissue datasets, with conventional baselines and dataset-specific adaptations. Explore the results and follow each method back to its paper and benchmark evidence.</p>
      <div class="hero-links"><a class="hero-primary" href="#/benchmark">Explore the results <span aria-hidden="true">↗</span></a><a href="#/gallery">Compare tissue predictions</a><a href="#activity">Latest campaign activity <span aria-hidden="true">↓</span></a></div>
    </div><div class="research-hero-evidence"><div class="hero-evidence-top"><span>THE EXPERIMENT LANDSCAPE</span><span>${number(data.coverage.rows.length)} × ${number(data.coverage.datasets.length)}</span></div>
      ${miniature(data)}<p>Each column is a model family; each row is a dataset. Colours describe evaluation availability, rather than prediction quality.</p>
    </div>
  </section>
  <div class="scope-stats" aria-label="Research scale">
    ${scaleCard(s.run_records, 'run records', 'Includes scored and unscored records')}
    ${scaleCard(s.model_families_with_runs, 'model families represented', `${number(s.catalog_models)} catalogued methods · includes queued work`)}
    ${scaleCard(s.datasets_registered, 'datasets represented', `${number(s.datasets_published)} with published result records`)}
    ${scaleCard(s.rankable_evaluations, 'scored evaluation rows', 'Dataset / stain / split / protocol results')}
    ${scaleCard(s.scalar_points, 'recorded curve points', `${number(s.scalar_series)} scalar histories`)}
  </div>
  <div class="visual-heading"><div><div class="eyebrow">The visual laboratory</div><h2>One evidence base. Seven ways to explore it.</h2></div><p>Results: ${date(data.snapshot.benchmark)}<br>Activity: ${date(data.snapshot.activity)} · saved snapshots</p></div>
  <nav class="visual-tabs" aria-label="Visual examples">
    ${[['atlas','01','Coverage atlas','Breadth & completeness'],['curves','02','Learning explorer','Recorded histories'],['radar','03','Radar profiles','Overlay or individual grids'],['scatter','04','Scatter explorer','Quality & compute trade-offs'],['heatmap','05','Metric heatmap','Models × quality measures'],['parallel','06','Parallel coordinates','Follow each metric profile'],['correlation','07','Metric relationships','Agreement & disagreement']].map(([id,n,title,subtitle]) => `<a ${layout===id?'aria-current="page"':''} class="visual-tab ${layout===id?'active':''}" href="#/overview${id==='atlas'?'':'/'+id}"><span class="view-index">${n}</span><span><strong>${title}</strong><small>${subtitle}</small></span>${id==='atlas'?'<span class="recommended-view">Start here</span>':''}</a>`).join('')}
  </nav>`;
}

function familyCount(data,dataset,key='tested_model_families'){
  if(typeof dataset[key]==='number')return dataset[key];
  return data.coverage.rows.filter(row=>row.cells.some(c=>c.dataset===dataset.id&&(key==='tested_model_families'?c.tested_runs>0:c.evaluated_runs>0))).length;
}

function balancePlan(data){
  const plan=data.coverage_balance;
  if(!plan)return '';
  const tasks=plan.tasks||[],total=plan.task_count||tasks.length,done=tasks.filter(t=>t.evaluated).length;
  const failed=tasks.filter(t=>t.failed&&!t.evaluated).length;
  return `<section class="visual-panel balance-plan"><div class="visual-panel-heading"><div><div class="eyebrow">A focused expansion</div><h2>Closing the coverage gap</h2><p>${number(total)} new runs add complementary model families across the breast and unstained datasets.</p></div><span class="chart-badge">${number(done)} / ${number(total)} fully evaluated</span></div><div class="balance-progress"><progress max="${total}" value="${done}" aria-label="New runs with full validation and test results">${done} of ${total}</progress><p><strong>${number(done)} completed</strong> · ${number(total-done-failed)} still in progress${failed?' · '+number(failed)+' need attention':''}<span>Full validation and test results are required to count as completed.</span></p></div><div class="table-wrap"><table><caption class="sr-only">Current tested model families and selected new training additions.</caption><thead><tr><th>Dataset</th><th class="number">Tested families now</th><th>Added families in this campaign</th><th class="number">Planned benchmark families</th></tr></thead><tbody>${(plan.datasets||[]).map(row=>{
    const dataset=data.datasets.find(d=>d.id===row.id)||row;
    const added=row.added_models||row.models||[];
    const names=added.map(m=>typeof m==='object'?m.name||m.id:plan.models?.find(model=>model.id===m)?.name||data.coverage.rows.find(model=>model.id===m)?.name||m);
    return `<tr><th scope="row">${esc(dataset.name||row.id)}</th><td class="number">${number(familyCount(data,dataset))}</td><td>${esc(names.join(', '))}${row.marker?`<small>${esc(row.marker)} anchor comparison</small>`:''}</td><td class="number">${number(row.target_model_families||row.target_families||familyCount(data,dataset)+added.length)}</td></tr>`;
  }).join('')}</tbody></table></div><details class="balance-task-details"><summary>Inspect all ${number(total)} new runs</summary><div class="table-wrap"><table><caption class="sr-only">Saved progress of each run in the coverage expansion.</caption><thead><tr><th>Model / dataset</th><th>Status</th><th class="number">Latest saved update</th><th>Evidence</th></tr></thead><tbody>${tasks.map(task=>{
    const model=plan.models?.find(m=>m.id===task.model)?.name||task.model;
    const dataset=(task.datasets||[]).map(id=>data.datasets.find(d=>d.id===id)?.short_name||data.datasets.find(d=>d.id===id)?.name||id).join(', ');
    const state=task.evaluated?'Full validation + test complete':task.state||'Queued';
    return `<tr><th scope="row"><a href="#/run/${encodeURIComponent('balance__'+task.id)}">${esc(model)}</a><small>${esc(dataset)} · ${esc(task.marker)}</small></th><td>${esc(state)}</td><td class="number">${task.last_step>0?number(task.last_step):'—'}</td><td>${task.evaluated?'Selected checkpoint evaluated':task.gate_passed?'Native training check passed':'Training check pending'}</td></tr>`;
  }).join('')}</tbody></table></div></details><p class="visual-caption">Progress observed ${date(plan.snapshot||data.snapshot.activity)}${plan.job_id?' · training array '+esc(plan.job_id):''}. Targets describe the model families covered once the selected runs finish. Original breast additions target HER2; the other marker and pooling combinations are not duplicated.</p></section>`;
}

function coverage(data) {
  const datasets = data.coverage.datasets;
  const rows = data.coverage.rows;
  return `<div class="atlas-layout"><section class="visual-panel coverage-panel"><div class="visual-panel-heading"><div><div class="eyebrow">Example 01 / The scope of the programme</div><h2>The coverage atlas</h2><p>Model families × datasets. Numbers count run records, including separate stains and training recipes.</p></div><span class="chart-badge">${number(rows.length)} families · ${number(datasets.length)} datasets</span></div>
    <div class="coverage-legend"><span><i class="scored"></i>Has scored evaluation</span><span><i class="recorded"></i>Other recorded work</span><span><i class="active"></i>Training / pending scores</span><span><i class="absent"></i>No record</span><span><i class="quality-warning"></i>Quality issue</span></div>
    <div class="coverage-scroll"><table class="coverage-table"><caption class="sr-only">Number of run records per model family and dataset. A pooled run can occur in several columns.</caption><thead><tr><th>Model family</th>${datasets.map(d=>`<th scope="col">${esc(d.short_name || d.name || d.id)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr><th scope="row">${esc(row.name)}</th>${datasets.map(d=>{
      const c=row.cells.find(c=>c.dataset===d.id)||{};
      const style=cellStyle(c), runs=c.runs||0;
      const label=`${row.name}, ${d.name || d.id}: ${runs} run records, ${c.evaluated_runs||0} with scored evaluations; ${c.status||'no record'}`;
      return `<td><a class="coverage-cell ${style}" href="${'#/benchmark?dataset='+encodeURIComponent(d.id)+'&model='+encodeURIComponent(row.id)}" aria-label="${esc(label)}" title="${esc(label)}"><span>${runs || '—'}</span><span class="cell-detail">${style==='scored'?'scored':style==='active'?'pending':style==='recorded'?'recorded':''}</span></a></td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>
    <p class="visual-caption">A record is not a claim that training or evaluation finished. Pooled runs appear in more than one dataset column; column totals must not be added as distinct runs.</p>
    </section><aside class="atlas-aside"><section class="visual-panel"><div class="eyebrow">Distinct families with held-out tests</div><h3>Model breadth by dataset</h3><p class="family-count-note">Each family counts once. Recipe variants do not increase these bars.</p><div class="dataset-bars">${data.datasets.map(d=>{
      const max=Math.max(...data.datasets.map(x=>familyCount(data,x)),1),counted=familyCount(data,d);
      return `<a href="${'#/benchmark?dataset='+encodeURIComponent(d.id)}" class="dataset-bar-row"><span>${esc(d.short_name || d.name)}</span><strong>${number(counted)}</strong><div class="dataset-bar-track"><i style="width:${100*counted/max}%"></i></div><small>${typeof d.trained_model_families==='number'?number(d.trained_model_families)+' trained families · ':''}${number(d.runs)} run records</small></a>`;
    }).join('')}</div></section>
    <section class="view-verdict"><span class="verdict-number">01</span><h3>Best first impression</h3><p>The atlas makes breadth and gaps visible together. Use it as the opening page; follow it with curves and metric profiles.</p><div><strong>Shows</strong> coverage, variants, missing results</div><div><strong>Doesn’t show</strong> which model is best</div></section>
    <section class="visual-panel evidence-totals"><h3>More than a leaderboard</h3><p><strong>${number(data.scale.evaluated_runs)}</strong> run records with a scored evaluation</p><p><strong>${number(data.scale.paired_fields)}</strong> retained paired tissue previews</p><p><strong>${number(data.scale.recorded_gpu_hours)}</strong> recorded GPU hours</p><small>Compute is recorded for ${number(data.scale.compute_runs)} runs only. Missing and parent allocations are not estimated.</small></section></aside></div>`;
}

function profileRuns(data) {
  return data.cohort.runs.filter(r=>selectedProfiles.has(r.id));
}

const profileIndex=(run,data)=>Math.max(0,data.cohort.runs.findIndex(r=>r.id===run.id));
const profileColor=(run,data)=>colors[profileIndex(run,data)%colors.length];
const profileDash=(run,data)=>dash[profileIndex(run,data)%dash.length];
const profileLabel=(run,data)=>data.cohort.runs.filter(r=>r.name===run.name).length>1?run.label:run.name;

function profileLegend(data) {
  return `<div class="profile-legend">${profileRuns(data).map(r=>`<button type="button" data-profile-focus="${esc(r.id)}" title="${esc(profileLabel(r,data))}" style="--series-color:${profileColor(r,data)}"><i class="series-line series-${profileIndex(r,data)%3}"></i>${esc(profileLabel(r,data))}</button>`).join('')}</div>`;
}

function radarPlot(data, explicitRuns=null, compact=false) {
  const runs=explicitRuns||profileRuns(data), metrics=data.cohort.metrics, n=metrics.length;
  if(n<3)return '<div class="empty">Choose at least three metric axes for the radar.</div>';
  if(!runs.length)return '<div class="empty">Select models to draw their profiles.</div>';
  const w=compact?410:650,h=compact?330:440,cx=w/2,cy=compact?163:216,r=compact?99:142;
  const point=(i,v)=>[cx+Math.cos(-Math.PI/2+2*Math.PI*i/n)*r*v,cy+Math.sin(-Math.PI/2+2*Math.PI*i/n)*r*v];
  const poly=v=>metrics.map((_,i)=>point(i,v).join(',')).join(' ');
  return `<svg class="research-chart radar-chart ${compact?'compact-radar':''}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Relative metric profiles on a fixed matched cohort. Outward means better on each axis; no overall score is calculated."><title>${compact?esc(runs[0].name)+' metric profile':'Relative metric trade-offs'}</title><desc>Each metric is scaled against all scored runs in the fixed cohort. Polygon area is not a model ranking.</desc>${[.25,.5,.75,1].map(v=>`<polygon points="${poly(v)}" fill="none" stroke="#dce3ef"/><text x="${cx+5}" y="${cy-r*v-5}">${Math.round(v*100)}</text>`).join('')}${metrics.map((m,i)=>{
    const end=point(i,1),label=point(i,1.23), anchor=Math.abs(label[0]-cx)<12?'middle':label[0]>cx?'start':'end';
    const text=compact&&m.key==='conch_cosine'?'CONCH':m.label.length>19?m.label.slice(0,17)+'…':m.label;
    return `<line x1="${cx}" x2="${end[0]}" y1="${cy}" y2="${end[1]}"/><text class="radar-label" x="${label[0]}" y="${label[1]+5}" text-anchor="${anchor}">${esc(text)} ${m.direction==='higher'?'↑':'↓'}<title>${esc(m.label)}</title></text>`;
  }).join('')}${runs.map((run,i)=>{
    const color=profileColor(run,data);
    return `<g data-profile-series="${esc(run.id)}"><polygon points="${metrics.map((m,k)=>point(k,run.normalized[m.key]).join(',')).join(' ')}" fill="${color}" fill-opacity="${runs.length>5?'.015':'.045'}" stroke="${color}" stroke-width="2.3" stroke-dasharray="${profileDash(run,data)}"/>${metrics.map((m,k)=>{const p=point(k,run.normalized[m.key]);return `<circle cx="${p[0]}" cy="${p[1]}" r="${compact?3:4}" fill="${color}"><title>${esc(profileLabel(run,data))} · ${esc(m.label)}: ${run.values[m.key].toFixed(5)} ${esc(m.unit||'')} · relative ${Math.round(run.normalized[m.key]*100)}</title></circle>`}).join('')}</g>`;
  }).join('')}</svg>`;
}

function radarDisplay(data){
  if(radarMode==='overlay')return radarPlot(data);
  return `<div class="radar-multiples">${profileRuns(data).map(run=>`<article><h4><span class="profile-key" style="--series-color:${profileColor(run,data)}"></span><a href="${esc(run.href||'#/run/'+run.id)}">${esc(profileLabel(run,data))}</a></h4>${radarPlot(data,[run],true)}</article>`).join('')}</div>`;
}

function metricDots(data) {
  const m=data.cohort.metrics.find(m=>m.key===focusedMetric)||data.cohort.metrics[0];
  const runs=profileRuns(data);
  const narrow=matchMedia('(max-width:700px)').matches;
  const w=narrow?450:650,h=Math.max(170,80+runs.length*48),left=narrow?122:130,right=narrow?70:95,top=36,bottom=40;
  let min=m.min,max=m.max;
  if(zoomDots&&runs.length){const values=runs.map(r=>r.values[m.key]);min=Math.min(...values);max=Math.max(...values);const margin=(max-min||Math.max(Math.abs(min)*.01,.001))*.18;min-=margin;max+=margin;}
  if(min===max){min-=.01;max+=.01;}
  const x=v=>left+(v-min)/(max-min)*(w-left-right);
  return `<svg class="research-chart dot-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Raw ${esc(m.label)} scores of the selected models. ${m.direction==='higher'?'Higher':'Lower'} is better."><title>Exact ${esc(m.label)} comparison</title>${Array.from({length:4},(_,i)=>{const v=min+(max-min)*i/3;return `<line x1="${x(v)}" x2="${x(v)}" y1="${top-8}" y2="${h-bottom}"/><text x="${x(v)}" y="${h-15}" text-anchor="middle">${v.toFixed(3)}</text>`;}).join('')}${runs.map((run,i)=>{
    const y=top+12+i*48,v=run.values[m.key];
    const label=narrow&&run.name.length>11?run.name.slice(0,10)+'…':run.name;
    return `<text x="${left-15}" y="${y+5}" text-anchor="end">${esc(label)}<title>${esc(profileLabel(run,data))}</title></text><circle cx="${x(v)}" cy="${y}" r="6" fill="${profileColor(run,data)}"><title>${esc(profileLabel(run,data))}: ${v.toFixed(6)}</title></circle><text x="${w-right+12}" y="${y+5}">${v.toFixed(4)}</text>`;
  }).join('')}</svg>`;
}

function rawTable(data) {
  return `<div class="table-wrap profile-values"><table><caption class="sr-only">Unnormalized metric values behind the radar</caption><thead><tr><th>Selected model / recipe</th>${data.cohort.metrics.map(m=>`<th>${esc(m.label)} ${m.direction==='higher'?'↑':'↓'}</th>`).join('')}</tr></thead><tbody>${profileRuns(data).map(r=>`<tr><th scope="row"><span class="profile-key" style="--series-color:${profileColor(r,data)}"></span><a href="${esc(r.href||'#/run/'+r.id)}">${esc(profileLabel(r,data))}</a></th>${data.cohort.metrics.map(m=>`<td class="number">${r.values[m.key].toFixed(4)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function prepareProfiles(data,index) {
  if(!profileGroups)profileGroups=buildProfileCohorts(index);
  if(!profileGroupId)profileGroupId=(profileGroups.find(g=>g.dataset==='finland'&&g.split==='test')||profileGroups[0]).id;
  const group=profileGroups.find(g=>g.id===profileGroupId)||profileGroups[0];
  const cohort=materializeProfileCohort(index,group,profileAxisKeys);
  if(!selectedProfiles)selectedProfiles=new Set(cohort.default_runs);
  if(!cohort.metrics.some(m=>m.key===focusedMetric))focusedMetric=cohort.metrics[0]?.key;
  return {...data,cohort};
}

function radarView(data) {
  const c=data.cohort;
  return `<section class="visual-panel"><div class="visual-panel-heading"><div><div class="eyebrow">03 / Metric profiles without a model cap</div><h2>Explore metric profiles</h2><p>${esc(c.dataset_name||c.dataset)} · ${esc(c.marker||'H&E')} · ${esc(c.split)} · ${number(c.n)} tiles per run · ${esc(c.protocol)}</p></div><span class="chart-badge">${c.runs.length} eligible runs · ${c.available_metrics.length} common metrics</span></div>
    <div class="profile-controls"><label class="field">Comparison cohort<select id="profile-cohort">${profileGroups.map(g=>`<option value="${esc(g.id)}" ${g.id===profileGroupId?'selected':''}>${esc(g.label)} (${g.entries.length} runs)</option>`).join('')}</select></label><label class="field">Radar arrangement<select id="radar-display"><option value="overlay" ${radarMode==='overlay'?'selected':''}>Overlay all selected profiles</option><option value="multiples" ${radarMode==='multiples'?'selected':''}>One radar per run / shared scales</option></select></label></div>
    <details class="profile-axis-picker"><summary>Choose radar axes (${c.metrics.length} of ${c.available_metrics.length} available)</summary><div>${c.available_metrics.map(m=>`<label><input type="checkbox" data-profile-axis="${esc(m.key)}" ${c.metrics.some(x=>x.key===m.key)?'checked':''}>${esc(m.label)} ${m.direction==='higher'?'↑':'↓'}</label>`).join('')}</div><p>Only metrics with finite values and full tile coverage in every run of this cohort are available. Choose at least three axes.</p></details>
    <div class="profile-picker"><div class="profile-select-tools"><strong id="profile-count">${selectedProfiles.size} of ${c.runs.length} selected</strong><button type="button" id="profiles-all">Select all ${c.runs.length}</button><button type="button" id="profiles-clear">Clear</button><label class="field">Find a run<input type="search" id="profile-search" placeholder="Model or recipe…"></label></div><div class="profile-model-choices">${c.runs.map(r=>`<label data-profile-choice><input type="checkbox" data-profile="${esc(r.id)}" ${selectedProfiles.has(r.id)?'checked':''}><span class="profile-key" style="--series-color:${profileColor(r,data)}"></span>${esc(profileLabel(r,data))}</label>`).join('')}</div><p id="profile-feedback" role="status" aria-live="polite">No selection limit. Hover or focus a legend label to isolate its line; switch to individual radars for a large selection.</p></div>
    <div class="profile-charts ${radarMode==='multiples'?'profile-grid-mode':''}" id="profile-charts"><div><h3>Relative metric profile</h3><p class="small muted">Outward is better on each axis · scale 0–100</p><div id="profile-legend">${profileLegend(data)}</div><div id="radar-plot">${radarDisplay(data)}</div></div><div class="profile-exact"><h3>Check the actual difference</h3><label class="field">Metric in original units<select id="profile-metric">${c.metrics.map(m=>`<option value="${esc(m.key)}" ${focusedMetric===m.key?'selected':''}>${esc(m.label)} · ${m.direction==='higher'?'higher':'lower'} is better${m.unit?' · '+esc(m.unit):''}</option>`).join('')}</select></label><label class="dot-zoom"><input type="checkbox" id="dot-zoom" ${zoomDots?'checked':''}>Zoom dot-plot axis to selected values</label><div id="metric-dots">${metricDots(data)}</div><div class="normalization-note"><strong>Fixed reference scales.</strong><p>Every radar uses the minimum and maximum across all ${number(c.runs.length)} runs in this cohort, even when only one is selected. The dot plot can zoom; its labelled axis then changes with the selection. Polygon area is not an overall quality score.</p></div></div></div>
    <div id="profile-values">${rawTable(data)}</div>
    <p class="visual-caption">Higher-is-better = (value − minimum) / range; lower-is-better = (maximum − value) / range. Constant axes sit at 50. These single-seed numerical differences do not establish statistical superiority.</p>
  </section><div class="view-verdict horizontal"><span class="verdict-number">03</span><div><h3>Choose the arrangement that answers your question</h3><p>Overlay profiles to see intersections. Individual radars preserve the same reference scales while making every run inspectable. The metric heatmap and parallel coordinates offer additional views of larger selections.</p></div><div class="verdict-limit"><strong>Reference cohort stays fixed</strong><span>Dataset, stain, split, sample count, crop, manifest and metric protocol define the comparison.</span></div></div>`;
}

function her2matchCampaign(data){
  const campaign=data.her2match_campaign;if(!campaign)return '';
  const labels={evaluated:'Evaluated',training:'Training',queued:'Queued',blocked:'Blocked',prepared:'Preparing'};
  return `<section class="visual-panel her2-completion" aria-labelledby="her2-completion-title"><div class="visual-panel-heading"><div><div class="eyebrow">HER2Match / 30 landscape families</div><h2 id="her2-completion-title">${number(campaign.counts.evaluated)} evaluated. ${number(campaign.target_model_families)} families accounted for.</h2><p>Original author case splits · observed ${esc(new Date(campaign.snapshot).toLocaleString('en-GB',{timeZone:'Europe/Brussels',day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'}))} Brussels time</p></div><a href="${esc(campaign.download)}" download>Download the HER2Match campaign</a></div><div class="her2-completion-counts">${Object.entries(labels).map(([key,label])=>`<span class="her2-status-${key}"><strong>${number(campaign.counts[key])}</strong> ${label}</span>`).join('')}</div>${campaign.quality_warning_count?`<p class="her2-quality-label"><strong>${number(campaign.quality_warning_count)} confirmed colour-collapse warnings.</strong> Complete evaluation does not make those models useful training outcomes.</p>`:''}<p class="visual-caption">Evaluated means that the selected checkpoint has complete validation and test results; it does not certify useful prediction quality. Training, queued and blocked methods have no invented scores. Existing scores remain available while a model continues training.</p><div class="her2-completion-grid">${campaign.rows.map(row=>{const tag=row.run_id?'a':'article';return `<${tag} ${row.run_id?`href="#/run/${encodeURIComponent(row.run_id)}"`:''} class="her2-completion-card her2-status-${esc(row.status)} ${row.quality_warning?'has-quality-warning':''}" data-her2-model="${esc(row.model)}" data-her2-status="${esc(row.status)}"><strong>${esc(row.name)}</strong><span>${esc(labels[row.status]||row.status)}</span>${row.quality_warning?'<span class="her2-quality-label">Output collapse</span>':''}<small>${row.reason?esc(row.reason):row.status==='evaluated'?'Full selected-checkpoint val/test results':row.step?number(row.step)+' saved production updates':row.gate_passed?'Native update/resume check passed':esc(row.scientific_status||'Original preparation pending')}</small>${row.job&&row.status!=='evaluated'?`<small>Job ${esc(row.job)}</small>`:''}</${tag}>`;}).join('')}</div>${campaign.rows.filter(row=>row.quality_warning).map(row=>`<p class="her2-quality-note" role="note"><strong>${esc(row.name)} quality warning.</strong> ${esc(row.quality_warning)}${row.repair_jobs?.length?' Repair job '+esc(row.repair_jobs.join(', '))+'.':''}</p>`).join('')}${campaign.rows.some(row=>row.reason)?`<p class="training-archive-note"><strong>BCIStainer requires genuine crop-level HER2 grades.</strong> Released WSI scores are metadata; they cannot be substituted for the original classifier/comparator labels. This method remains blocked pending those labels.</p>`:''}${campaign.external_validation?`<p class="visual-caption"><strong>Separate external validation:</strong> official BCIStainer author EMA checkpoint · job ${esc(campaign.external_validation.job_id)} · ${esc(campaign.external_validation.status)}. This transfers an externally trained generator; it does not train BCIStainer on HER2Match or complete the in-domain task.</p>`:''}<p class="visual-caption">${esc(campaign.note)} Clinical HER2 grading is not evaluated. This is a saved execution snapshot; the page does not poll the cluster.</p></section>`;
}

function gpuActivity(data){
  const activity=data.gpu_activity;if(!activity)return '';
  const partition=activity.partition||{},total=partition.schedulable_gpus||1;
  const training=activity.own_training_gpus||0,preparing=activity.own_preparation_gpus||0,evaluating=activity.own_evaluation_gpus||0,own=training+preparing+evaluating;
  const other=Math.max(0,(partition.allocated_gpus||0)-own),free=partition.free_schedulable_gpus||0;
  const label=model=>data.coverage.rows.find(row=>row.id===model)?.name||model;
  const preparation=activity.preparation_jobs||[];
  return `<section class="visual-panel gpu-activity"><div class="visual-panel-heading"><div><div class="eyebrow">Current cluster snapshot</div><h2>Running now. Ready to run next.</h2><p>${number(training)} H200 GPUs allocated to published models across ${number(activity.own_running_training_jobs)} jobs · observed ${esc(new Date(activity.snapshot).toLocaleString('en-GB',{timeZone:'Europe/Brussels',day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'}))} Brussels time</p>${evaluating?`<p>${number(evaluating)} additional H200 GPUs running held-out evaluation.</p>`:''}${preparation.length?`<p>${number(preparing)} additional H200 GPUs allocated to original TRAIN-data preparation.</p>`:''}</div><a href="${esc(activity.download||'data/gpu-activity.json')}" download>Download the capacity snapshot</a></div><div class="gpu-capacity-bar" role="img" aria-label="${own} GPUs allocated to included published methods (${training} training, ${preparing} data preparation, ${evaluating} evaluation), ${other} allocated to other workloads, ${free} free out of ${total} schedulable GPUs"><i class="gpu-own" style="width:${100*own/total}%"></i><i class="gpu-other" style="width:${100*other/total}%"></i><i class="gpu-free" style="width:${100*free/total}%"></i></div><div class="gpu-capacity-legend"><span><i class="gpu-own"></i>${number(own)} for published methods</span><span><i class="gpu-other"></i>${number(other)} other workloads</span><span><i class="gpu-free"></i>${number(free)} free</span></div><p class="gpu-capacity-note">Training status covers only the included published methods. ${number(total)} schedulable GPUs in the shared partition. The largest free group on one node is ${number(partition.largest_free_gpus_per_node)} GPUs. Four-GPU runs wait until four are available together. Independent ready jobs are eligible to run concurrently.</p><div class="table-wrap"><table><caption class="sr-only">Included published-method training jobs and saved optimizer progress.</caption><thead><tr><th>Model / dataset</th><th>Status</th><th class="number">GPUs / job</th><th class="number">Saved update</th><th>Queue / progress</th></tr></thead><tbody>${(activity.jobs||[]).map(job=>`<tr><th scope="row">${esc(label(job.model))}<small>${esc(job.dataset)} · ${esc(job.job)}</small></th><td>${esc(job.status)}${job.evaluation_recovered?'<small>Evaluation recovered; training complete</small>':''}${job.kind?`<small>${esc(job.kind)}</small>`:''}</td><td class="number">${number(job.gpus)}</td><td class="number">${number(job.step)}</td><td>${job.status==='RUNNING'?'Allocated and running':job.status==='COMPLETED'?'Allocation completed':job.status==='PENDING'&&job.gpus>(partition.largest_free_gpus_per_node||0)?'Waiting for a suitable GPU allocation':esc(job.reason||'Queued')}${job.new_optimizer_updates!=null&&job.step>job.new_optimizer_updates?`<small>${number(job.new_optimizer_updates)} new continuation updates</small>`:''}</td></tr>`).join('')}</tbody></table></div>${preparation.length?`<div class="gpu-preparation"><h3>TDKStain original data preparation</h3><p>Original Cellpose2.2.3 nuclei/cyto2 ensembles prepare supervision on all11,610 TRAIN fields. Preparation GPUs are counted separately from model training. The one-field probe is never a complete production map set; no validation or test images enter this preparation.</p><div class="table-wrap"><table><thead><tr><th>Preparation stage</th><th>Status</th><th>GPUs / job</th><th>Cached TRAIN fields</th><th>Dependency</th></tr></thead><tbody>${preparation.map(job=>`<tr><th>${job.stage==='probe'?'One-field native probe':'Full original TRAIN maps'}<small>${esc(job.job)}</small></th><td>${esc(job.status)}${job.probe_passed?'<small>Probe passed; production set incomplete</small>':job.complete?'<small>Full source-bound map set verified</small>':''}</td><td>${number(job.gpus)}</td><td>${number(job.cached_map_files)} / ${number(job.required_rows)}</td><td>${esc(job.dependency||'Independent')}</td></tr>`).join('')}</tbody></table></div></div>`:''}${activity.blocked?.run_records?`<p class="training-archive-note"><strong>${number(activity.blocked.run_records)} additional run records remain blocked.</strong> ${esc(activity.blocked.reason)}. These plans retain their separately recorded archive blocker; their original data/checkpoints must be restored before they can run.</p>${activity.blocked.readiness_groups?`<div class="gpu-deferred-groups">${Object.entries(activity.blocked.readiness_groups).map(([key,value])=>`<span><strong>${number(value)}</strong> ${esc(({CHECKPOINT_AND_NATIVE_SOURCE_VERIFIED_WAITING_DATA:'Verified checkpoint and source; waiting for data',ORIGINAL_SOURCE_BINDING_REVIEW_REQUIRED:'Historical source verification still required',ZERO_LR_REQUIRES_EXPLICIT_NEW_LR_STAGE:'Zero learning rate; new stage required',ARCHIVED_PARENT_CHECKPOINT_UNAVAILABLE:'Archived parent checkpoint required'})[key]||key)}</span>`).join('')}</div><a class="gpu-deferred-download" href="${esc(activity.blocked.plan_download)}" download>Download all ${number(activity.blocked.run_records)} prepared run plans</a>`:''}`:''}<p class="gpu-capacity-note">${activity.limits?.artificial_two_new_allocation_cap_removed?'The previous voluntary concurrency cap has been removed. ':''}Per-run GPU counts and global training batches retain their original recipes. This is a saved scheduler snapshot; the page does not poll the cluster.</p></section>`;
}

function trainingAudit(data){
  const audit=data.training_audit;if(!audit)return '';
  const counts=audit.counts||{},blocked=audit.blocked||{};
  const blockedCount=blocked.total_run_records||((blocked.orion_runs||0)+(blocked.archived_original_breast_runs||0));
  return `<section class="visual-panel training-audit"><div class="visual-panel-heading"><div><div class="eyebrow">Training completeness</div><h2>Which runs still need more training?</h2><p>${number(audit.run_records)} run records reviewed · ${number(audit.runs_with_training_evidence)} with training evidence · ${date(audit.snapshot)}</p></div><a href="${esc(audit.download||'data/training-audit.json')}" download>Download the run-by-run audit</a></div><div class="training-audit-counts"><div><strong>${number(counts.validation_patience_exhausted)}</strong><span>Stopped by validation patience</span></div><div><strong>${number(counts.hard_budget_with_sustained_validation_gain)}</strong><span>Budget ended while validation improved</span></div><div><strong>${number(counts.hard_budget_without_sustained_tail_gain)}</strong><span>Budget ended without sustained recent gain</span></div></div><p class="training-audit-note">${esc(audit.decision_metric)} Counts describe included run records and dataset-specific training variants.</p><div class="training-followups">${(data.followup_campaigns||[]).map(task=>`<article><span>${esc(task.kind)}</span><h3><a href="#/run/${encodeURIComponent(task.run_id)}">${esc(task.model)}</a> · ${esc(data.datasets.find(d=>d.id===task.dataset)?.name||task.dataset)}</h3><p>${esc(task.state)}</p><small>Job ${esc(task.job)}</small></article>`).join('')}</div>${blockedCount?`<p class="training-archive-note"><strong>${number(blockedCount)} further flagged records await archive access.</strong> ${esc(blocked.reason)}</p>`:''}<details><summary>How to interpret the audit</summary><ul>${(audit.limitations||[]).map(note=>`<li>${esc(note)}</li>`).join('')}</ul><a href="#/overview/curves">Inspect the recorded validation curves</a></details></section>`;
}

function activities(data) {
  const activity=data.activity;
  return `<section class="campaign-section" id="activity"><div class="visual-heading"><div><div class="eyebrow">Work continuing beyond the published scores</div><h2>The next chapter is already running.</h2></div><p>Activity observed ${date(activity.observed_utc || data.snapshot.activity)}<br>Saved evidence; this page does not poll the cluster.</p></div>
    <div class="campaign-grid"><article class="visual-panel"><div class="visual-panel-heading"><div><h3>Original HER2Match campaign</h3><p>${activity.her2match.length} initial models · 21,172 paired images · author case splits</p></div><span class="chart-badge">New dataset</span></div><div class="campaign-runs">${activity.her2match.map(r=>{
      const progress=r.max_steps?Math.min(100,100*(r.step||0)/r.max_steps):0;
      return `<div class="campaign-run"><div><strong>${esc(r.name)}</strong><span>${esc(r.state)}</span></div><div class="campaign-progress"><i style="width:${progress}%"></i></div><small>${r.step?number(r.step)+' optimizer updates':r.production_started?'Production started':'Awaiting production'}${r.max_steps?' / '+number(r.max_steps):''}</small></div>`;
    }).join('')}</div><p class="visual-caption">Progress bars show the recorded update budget where available. Queued and unfinished runs carry no final scores.</p></article>
    <div><article class="visual-panel"><h3>Recoveries, with the evidence retained</h3><div class="recovery-list">${activity.recoveries.map(r=>`<div><span class="recovery-state ${r.complete?'done':''}" aria-hidden="true">${r.complete?'✓':'↗'}</span><div><strong>${esc(r.name)}</strong><span>${esc(r.state)}${r.step?' · '+number(r.step)+' updates':''}</span><p>${esc(r.note)}</p></div></div>`).join('')}</div></article>
    <article class="visual-panel metric-readiness"><h3>Pathology-aware evaluators</h3>${activity.evaluators.map(e=>`<div><strong>${esc(e.name)}</strong><span>${esc(e.state)}</span><p>${esc(e.note)}</p></div>`).join('')}</article></div></div>
    </section><details class="overview-provenance"><summary>How these visual examples are grounded</summary><p>The coverage counts and scores come from verified evaluation evidence. Campaign cards show separately dated execution progress. A queued or unfinished run enters scored comparisons only after its complete held-out evaluation is verified.</p><ul>${(data.notes||[]).map(n=>`<li>${esc(n)}</li>`).join('')}</ul><a href="data/visual-summary.json" download>Download the chart data and evidence fingerprints</a></details>`;
}

function bind(main, data, layout) {
  const landscape=$('.atlas-mini',main),inspector=$('#landscape-inspector',main);
  if(landscape&&inspector){
    const cells=$$('[data-landscape-model]',landscape);
    const inspect=element=>{
      const row=data.coverage.rows.find(r=>r.id===element.dataset.landscapeModel);
      const dataset=data.coverage.datasets.find(d=>d.id===element.dataset.landscapeDataset);
      const cell=row?.cells.find(c=>c.dataset===dataset?.id)||{},style=cellStyle(cell);
      const state=style==='scored'?'Scored evaluation available':style==='active'?'Training recorded or awaiting scores':style==='recorded'?'Recorded work; no scored evaluation':'No run recorded';
      cells.forEach(c=>{c.classList.toggle('landscape-column-active',c.dataset.landscapeModel===row?.id);c.classList.toggle('landscape-cell-active',c===element);c.tabIndex=c===element?0:-1;});
      inspector.innerHTML=`<strong>${esc(row?.name)} <span>· ${esc(dataset?.name||dataset?.id)}</span></strong><span>${esc(state)} · ${number(cell.runs)} run records · ${number(cell.evaluated_runs)} scored · ${number(cell.tested_runs)} with tests${cell.active_runs?' · '+number(cell.active_runs)+' unfinished':''}</span><small>${esc(cell.status||'No recorded work')}.${cell.quality_warning?' Quality warning: '+esc(cell.quality_warning):''} <a href="${esc(element.getAttribute('href'))}">Open matching runs</a></small>`;
    };
    cells.forEach(element=>{
      let pointerType;
      element.onpointerenter=()=>inspect(element);element.onfocus=()=>inspect(element);
      element.onpointerdown=event=>{pointerType=event.pointerType;};
      element.onclick=event=>{if(event.detail>0&&(pointerType==='touch'||event.pointerType==='touch')){event.preventDefault();inspect(element);}pointerType=null;};
      element.onkeydown=event=>{
        const delta={ArrowLeft:[0,-1],ArrowRight:[0,1],ArrowUp:[-1,0],ArrowDown:[1,0]}[event.key];
        if(!delta)return;event.preventDefault();
        const r=Math.max(0,Math.min(data.coverage.datasets.length-1,Number(element.dataset.landscapeRow)+delta[0]));
        const c=Math.max(0,Math.min(data.coverage.rows.length-1,Number(element.dataset.landscapeColumn)+delta[1]));
        cells.find(x=>Number(x.dataset.landscapeRow)===r&&Number(x.dataset.landscapeColumn)===c)?.focus();
      };
    });
  }
  const activityLink=$('a[href="#activity"]',main);
  if(activityLink)activityLink.onclick=e=>{e.preventDefault();$('#activity',main).scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});};
  $$('a.coverage-cell[href="#activity"],.dataset-bar-row[href="#activity"]',main).forEach(a=>a.onclick=e=>{e.preventDefault();$('#activity',main).scrollIntoView({behavior:'auto'});});
  if(layout==='radar'){
    const focus=id=>$$('[data-profile-series]',main).forEach(el=>el.style.opacity=!id||el.dataset.profileSeries===id?'1':'.1');
    const bindFocus=()=>$$('[data-profile-focus]',main).forEach(button=>{
      button.onpointerenter=()=>focus(button.dataset.profileFocus);button.onpointerleave=()=>focus(null);
      button.onfocus=()=>focus(button.dataset.profileFocus);button.onblur=()=>focus(null);
    });
    const redraw=()=>{
      $('#profile-count',main).textContent=`${selectedProfiles.size} of ${data.cohort.runs.length} selected`;
      $('#profile-feedback',main).textContent=selectedProfiles.size?'No selection limit. Hover or focus a legend label to isolate its line.':'Select models to draw their profiles.';
      $('#radar-plot',main).innerHTML=radarDisplay(data);$('#metric-dots',main).innerHTML=metricDots(data);
      $('#profile-values',main).innerHTML=rawTable(data);$('#profile-legend',main).innerHTML=profileLegend(data);bindFocus();
    };
    const rebuild=()=>{
      const next=prepareProfiles(data,activeIndex);activeSummary=next;
      $('#visual-example',main).innerHTML=radarView(next);bind(main,next,'radar');
    };
    $$('[data-profile]',main).forEach(input=>input.onchange=()=>{
      input.checked?selectedProfiles.add(input.dataset.profile):selectedProfiles.delete(input.dataset.profile);redraw();
    });
    $('#profiles-all',main).onclick=()=>{selectedProfiles=new Set(data.cohort.runs.map(r=>r.id));$$('[data-profile]',main).forEach(el=>el.checked=true);redraw();};
    $('#profiles-clear',main).onclick=()=>{selectedProfiles.clear();$$('[data-profile]',main).forEach(el=>el.checked=false);redraw();};
    $('#profile-search',main).oninput=e=>{const term=e.target.value.toLowerCase();$$('[data-profile-choice]',main).forEach(label=>label.hidden=!label.textContent.toLowerCase().includes(term));};
    $('#profile-cohort',main).onchange=e=>{profileGroupId=e.target.value;selectedProfiles=null;profileAxisKeys=null;rebuild();};
    $('#radar-display',main).onchange=e=>{radarMode=e.target.value;$('#profile-charts',main).classList.toggle('profile-grid-mode',radarMode==='multiples');redraw();};
    $$('[data-profile-axis]',main).forEach(input=>input.onchange=()=>{
      const keys=$$('[data-profile-axis]:checked',main).map(el=>el.dataset.profileAxis);
      if(keys.length<3){input.checked=true;$('#profile-feedback',main).textContent='Choose at least three axes to form a radar profile.';return;}
      profileAxisKeys=keys;rebuild();
    });
    $('#profile-metric',main).onchange=e=>{focusedMetric=e.target.value;$('#metric-dots',main).innerHTML=metricDots(data);};
    $('#dot-zoom',main).onchange=e=>{zoomDots=e.target.checked;$('#metric-dots',main).innerHTML=metricDots(data);};
    bindFocus();
  }
}

export function disposeOverview(){
  renderSequence++;
  if(typeof explorerCleanup==='function')explorerCleanup();
  explorerCleanup=null;
}

export async function renderOverview(main, layout='atlas', index) {
  const sequence=++renderSequence;
  if(!['atlas','curves','radar','scatter','heatmap','parallel','correlation'].includes(layout))layout='atlas';
  if(!snapshotPromise)snapshotPromise=fetch('data/visual-summary.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('The visual summary could not load.');return r.json();});
  let data=await snapshotPromise;
  if(!index)index=await fetch('data/index.json').then(r=>r.json());
  activeIndex=index;
  if(sequence!==renderSequence)return;
  if(!location.hash.startsWith('#/overview')&&location.hash&&location.hash!=='#activity')return;
  if(layout==='radar')data=prepareProfiles(data,index);
  activeSummary=data;
  const content=layout==='radar'?radarView(data):layout==='atlas'?coverage(data):'<div class="loading">Loading the visual explorer…</div>';
  main.innerHTML=`<div class="research-overview">${header(data,layout)}<section class="visual-example" id="visual-example" aria-label="${layout} explorer">${content}</section>${her2matchCampaign(data)}${gpuActivity(data)}${balancePlan(data)}${trainingAudit(data)}${activities(data)}</div>`;
  bind(main,data,layout);
  const container=$('#visual-example',main);
  if(layout==='curves'){
    const module=await import('./learning-explorer.js');
    if(sequence!==renderSequence)return;
    const cleanup=await module.renderLearningExplorer(container,index);
    if(sequence===renderSequence)explorerCleanup=cleanup;else if(typeof cleanup==='function')cleanup();
  }else if(['scatter','heatmap','parallel','correlation'].includes(layout)){
    const module=await import('./data-laboratory.js?v=20261120-completed-results');
    if(sequence!==renderSequence)return;
    const cleanup=await module.renderDataLaboratory(container,index,layout);
    if(sequence===renderSequence)explorerCleanup=cleanup;else if(typeof cleanup==='function')cleanup();
  }
}

matchMedia('(max-width:700px)').addEventListener('change',()=>{
  if(activeSummary&&$('#metric-dots'))$('#metric-dots').innerHTML=metricDots(activeSummary);
});
