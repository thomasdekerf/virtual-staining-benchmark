const esc = value => String(value ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const number = value => Number(value || 0).toLocaleString('en-GB', {maximumFractionDigits: 0});
const date = value => value ? new Date(value).toLocaleDateString('en-GB', {day:'numeric',month:'short',year:'numeric',timeZone:'UTC'}) : 'Not recorded';
const colors = ['#4159dc', '#b83475', '#147b71'];
const dash = ['', '8 4', '3 4'];
let snapshotPromise;
let activeSummary;
let selectedProfiles, selectedLines, focusedMetric = 'lpips', zoomDots = true;
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function cellStyle(cell) {
  if (cell.evaluated_runs) return 'scored';
  if (cell.status && /train|queue|submitted|running|paused/i.test(cell.status)) return 'active';
  return cell.runs ? 'recorded' : 'absent';
}

function miniature(data) {
  return `<div class="atlas-mini" aria-hidden="true">${data.coverage.datasets.map(dataset => `<div class="atlas-mini-row"><span>${esc(dataset.short_name || dataset.name || dataset.id)}</span><div>${data.coverage.rows.map(row => {
    const cell = row.cells.find(c => c.dataset === dataset.id) || {};
    return `<i class="${cellStyle(cell)}"></i>`;
  }).join('')}</div></div>`).join('')}</div>`;
}

function scaleCard(value, label, note) {
  return `<div class="scope-stat"><strong>${number(value)}</strong><span>${label}</span><small>${note}</small></div>`;
}

function header(data, layout) {
  const s = data.scale;
  return `<section class="research-hero" aria-labelledby="research-title">
    <div class="research-hero-copy"><div class="research-kicker">Virtual staining / Research atlas</div>
      <h1 id="research-title">Virtual staining,<br>measured at scale.</h1>
      <p>A view across model families, tissue datasets and training recipes. Explore the breadth of the experiments, then follow each result back to its evidence.</p>
      <div class="hero-links"><a class="hero-primary" href="#/benchmark">Explore the results <span aria-hidden="true">↗</span></a><a href="#activity">Latest campaign activity <span aria-hidden="true">↓</span></a></div>
    </div><div class="research-hero-evidence"><div class="hero-evidence-top"><span>THE EXPERIMENT LANDSCAPE</span><span>${number(data.coverage.rows.length)} × ${number(data.coverage.datasets.length)}</span></div>
      ${miniature(data)}<p>Each column is a model family.<br>Each row is a dataset. Filled cells have recorded work.</p>
    </div>
  </section>
  <div class="scope-stats" aria-label="Research scale">
    ${scaleCard(s.run_records, 'run records', 'Includes scored and unscored records')}
    ${scaleCard(s.model_families_with_runs, 'historical model families', `${number(s.catalog_models)} catalogued methods · BBDM now added`)}
    ${scaleCard(s.datasets_registered, 'datasets represented', `${number(s.datasets_published)} with published result records + HER2Match`)}
    ${scaleCard(s.rankable_evaluations, 'scored evaluation rows', 'Dataset / stain / split / protocol results')}
    ${scaleCard(s.scalar_points, 'recorded curve points', `${number(s.scalar_series)} scalar histories`)}
  </div>
  <div class="visual-heading"><div><div class="eyebrow">Three ways into the evidence</div><h2>Choose a view. See a different part of the work.</h2></div><p>Results: ${date(data.snapshot.benchmark)}<br>Activity: ${date(data.snapshot.activity)} · saved snapshots</p></div>
  <nav class="visual-tabs" aria-label="Visual examples">
    ${[['atlas','01','Coverage atlas','Breadth & completeness'],['curves','02','Learning curves','Progress & convergence'],['radar','03','Metric profiles','Strengths & trade-offs']].map(([id,n,title,subtitle]) => `<a ${layout===id?'aria-current="page"':''} class="visual-tab ${layout===id?'active':''}" href="#/overview${id==='atlas'?'':'/'+id}"><span class="view-index">${n}</span><span><strong>${title}</strong><small>${subtitle}</small></span>${id==='atlas'?'<span class="recommended-view">Start here</span>':''}</a>`).join('')}
  </nav>`;
}

function coverage(data) {
  const datasets = data.coverage.datasets;
  const rows = data.coverage.rows;
  return `<div class="atlas-layout"><section class="visual-panel coverage-panel"><div class="visual-panel-heading"><div><div class="eyebrow">Example 01 / The scope of the programme</div><h2>The coverage atlas</h2><p>Model families × datasets. Numbers count run records, including separate stains and training recipes.</p></div><span class="chart-badge">${number(rows.length)} families · ${number(datasets.length)} datasets</span></div>
    <div class="coverage-legend"><span><i class="scored"></i>Has scored evaluation</span><span><i class="recorded"></i>Recorded, unscored</span><span><i class="active"></i>Current campaign</span><span><i class="absent"></i>No record</span></div>
    <div class="coverage-scroll"><table class="coverage-table"><caption class="sr-only">Number of run records per model family and dataset. A pooled run can occur in several columns.</caption><thead><tr><th>Model family</th>${datasets.map(d=>`<th scope="col">${esc(d.short_name || d.name || d.id)}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr><th scope="row">${esc(row.name)}</th>${datasets.map(d=>{
      const c=row.cells.find(c=>c.dataset===d.id)||{};
      const style=cellStyle(c), runs=c.runs||0;
      const label=`${row.name}, ${d.name || d.id}: ${runs} run records, ${c.evaluated_runs||0} with scored evaluations${style==='active'?', current campaign':''}`;
      return `<td><a class="coverage-cell ${style}" href="${d.id==='her2match'?'#activity':'#/benchmark?dataset='+encodeURIComponent(d.id)+'&model='+encodeURIComponent(row.id)}" aria-label="${esc(label)}"><span>${runs || '—'}</span><span class="cell-detail">${style==='scored'?'scored':style==='active'?'campaign':style==='recorded'?'recorded':''}</span></a></td>`;
    }).join('')}</tr>`).join('')}</tbody></table></div>
    <p class="visual-caption">A record is not a claim that training or evaluation finished. Pooled runs appear in more than one dataset column; column totals must not be added as distinct runs.</p>
    </section><aside class="atlas-aside"><section class="visual-panel"><div class="eyebrow">Published snapshot</div><h3>How the work is distributed</h3><div class="dataset-bars">${data.datasets.map(d=>{
      const max=Math.max(...data.datasets.map(x=>x.runs||0),1);
      return `<a href="${d.id==='her2match'?'#activity':'#/benchmark?dataset='+encodeURIComponent(d.id)}" class="dataset-bar-row"><span>${esc(d.short_name || d.name)}</span><strong>${number(d.runs)}</strong><div class="dataset-bar-track"><i style="width:${100*(d.runs||0)/max}%"></i></div><small>${number(d.evaluated_runs)} records with scores</small></a>`;
    }).join('')}</div></section>
    <section class="view-verdict"><span class="verdict-number">01</span><h3>Best first impression</h3><p>The atlas makes breadth and gaps visible together. Use it as the opening page; follow it with curves and metric profiles.</p><div><strong>Shows</strong> coverage, variants, missing results</div><div><strong>Doesn’t show</strong> which model is best</div></section>
    <section class="visual-panel evidence-totals"><h3>More than a leaderboard</h3><p><strong>${number(data.scale.evaluated_runs)}</strong> run records with a scored evaluation</p><p><strong>${number(data.scale.paired_fields)}</strong> retained paired tissue previews</p><p><strong>${number(data.scale.recorded_gpu_hours)}</strong> recorded GPU hours</p><small>Compute is recorded for ${number(data.scale.compute_runs)} runs only. Missing and parent allocations are not estimated.</small></section></aside></div>`;
}

function axes({w,h,left,right,top,bottom,xmin,xmax,ymin,ymax,xlabel,ylabel,yformat}) {
  const x=v=>left+(v-xmin)/(xmax-xmin)*(w-left-right);
  const y=v=>h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom);
  return {x,y,markup:`${Array.from({length:5},(_,i)=>{
    const v=ymin+(ymax-ymin)*i/4;
    return `<line x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-12}" y="${y(v)+4}" text-anchor="end">${yformat(v)}</text>`;
  }).join('')}${Array.from({length:5},(_,i)=>{
    const v=xmin+(xmax-xmin)*i/4;
    return `<text x="${x(v)}" y="${h-bottom+25}" text-anchor="middle">${number(v)}</text>`;
  }).join('')}<text class="axis-label" x="${(left+w-right)/2}" y="${h-9}" text-anchor="middle">${esc(xlabel)}</text><text class="axis-label" x="${left}" y="${top-17}">${esc(ylabel)}</text>`};
}

function learningPlot(data, selected) {
  const series=data.trajectories.series.filter(s=>selected.has(s.id));
  if(!series.length)return '<div class="empty">Select at least one model to show its measured trajectory.</div>';
  const all=series.flatMap(s=>s.points), xmax=Math.max(...all.map(p=>p[0])), xmin=Math.min(...all.map(p=>p[0]));
  const low=Math.min(...all.map(p=>p[1])), high=Math.max(...all.map(p=>p[1])), pad=(high-low||.1)*.1;
  const narrow=matchMedia('(max-width:700px)').matches;
  const w=narrow?450:880,h=narrow?365:410,left=76,right=24,top=48,bottom=64;
  const ax=axes({w,h,left,right,top,bottom,xmin:Math.min(0,xmin),xmax:xmax||1,ymin:Math.max(0,low-pad),ymax:high+pad,xlabel:data.trajectories.x_label,ylabel:data.trajectories.y_label,yformat:v=>v.toFixed(3)});
  return `<svg class="research-chart learning-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Recorded validation LPIPS over optimizer updates. Lower is better. No smoothing or extrapolation."><title>Validation LPIPS learning curves</title>${ax.markup}${series.map(s=>{
    const index=data.trajectories.series.findIndex(x=>x.id===s.id)%colors.length;
    const selected=s.points.find(p=>p[0]===s.selected_step), marker=selected?`<polygon points="${[[0,-5],[5,0],[0,5],[-5,0]].map(([dx,dy])=>(ax.x(selected[0])+dx)+','+(ax.y(selected[1])+dy)).join(' ')}" fill="white" stroke="${colors[index]}" stroke-width="2"><title>${esc(s.name)} selected checkpoint: update ${number(selected[0])}, validation LPIPS ${selected[1].toFixed(5)}</title></polygon>`:'';
    return `<path d="${s.points.map((p,i)=>(i?'L':'M')+ax.x(p[0]).toFixed(2)+','+ax.y(p[1]).toFixed(2)).join(' ')}" fill="none" stroke="${colors[index]}" stroke-width="2.5" stroke-dasharray="${dash[index]}"/>${s.points.map(p=>`<circle class="measured-point" cx="${ax.x(p[0])}" cy="${ax.y(p[1])}" r="3" fill="${colors[index]}"><title>${esc(s.name)}: update ${number(p[0])}, LPIPS ${p[1].toFixed(5)}</title></circle>`).join('')}${marker}`;
  }).join('')}</svg>`;
}

function curveView(data) {
  if(!selectedLines)selectedLines=new Set(data.trajectories.series.map(s=>s.id));
  return `<section class="visual-panel"><div class="visual-panel-heading"><div><div class="eyebrow">Example 02 / What happened during training</div><h2>Learning curves, together</h2><p>${esc(data.trajectories.dataset_name || data.trajectories.dataset)} · ${esc(data.trajectories.marker || '')} · validation · one recorded seed per run</p></div><span class="chart-badge">Raw measurements · lower LPIPS is better</span></div>
    <div class="curve-choices">${data.trajectories.series.map((s,i)=>`<label style="--series-color:${colors[i%3]}"><input type="checkbox" data-line="${esc(s.id)}" ${selectedLines.has(s.id)?'checked':''}><i class="series-line series-${i%3}"></i>${esc(s.name)}<small>${number(s.recorded_points)} recorded values</small></label>`).join('')}</div>
    <div id="learning-plot">${learningPlot(data,selectedLines)}</div><p class="checkpoint-legend">◇ Selected checkpoint under the recorded run policy</p>
    <p class="visual-caption">Optimizer updates use the actual recorded axis; only compatible trajectories are overlaid. Lines end at the last measurement. No smoothing, fabricated points, or extension to a common budget. Different training recipes can use different effective batch sizes.</p>
    <div class="trajectory-cards">${data.trajectories.series.map((s,i)=>`<a href="${esc(s.href||'#/run/'+encodeURIComponent(s.id))}" style="--series-color:${colors[i%3]}"><span>${esc(s.name)}</span><strong>${s.points.at(-1)?.[1].toFixed(4) || '—'}<small> final recorded validation LPIPS</small></strong><p>${number(s.points.at(-1)?.[0])} updates · open the complete run ↗</p></a>`).join('')}</div>
  </section><div class="view-verdict horizontal"><span class="verdict-number">02</span><div><h3>Best for understanding the work over time</h3><p>Shows convergence, instability and how long each recipe was trained. Validation curves support interpretation; the final held-out test comparison remains separate.</p></div><div class="verdict-limit"><strong>Doesn’t show</strong><span>Statistical certainty or an equal-compute architecture comparison.</span></div></div>`;
}

function profileRuns(data) {
  return data.cohort.runs.filter(r=>selectedProfiles.has(r.id));
}

function profileLegend(data) {
  return `<div class="profile-legend">${profileRuns(data).map((r,i)=>`<span style="--series-color:${colors[i%3]}"><i class="series-line series-${i%3}"></i>${esc(r.name)}</span>`).join('')}</div>`;
}

function radarPlot(data) {
  const runs=profileRuns(data), metrics=data.cohort.metrics, n=metrics.length;
  const w=650,h=440,cx=w/2,cy=216,r=142;
  const point=(i,v)=>[cx+Math.cos(-Math.PI/2+2*Math.PI*i/n)*r*v,cy+Math.sin(-Math.PI/2+2*Math.PI*i/n)*r*v];
  const poly=v=>metrics.map((_,i)=>point(i,v).join(',')).join(' ');
  return `<svg class="research-chart radar-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Relative metric profiles on a fixed matched cohort. Outward means better on each axis; no overall score is calculated."><title>Relative metric trade-offs</title><desc>Each metric is scaled against all scored runs in the fixed cohort. Polygon area is not a model ranking.</desc>${[.25,.5,.75,1].map(v=>`<polygon points="${poly(v)}" fill="none" stroke="#dce3ef"/><text x="${cx+5}" y="${cy-r*v-5}">${Math.round(v*100)}</text>`).join('')}${metrics.map((m,i)=>{
    const end=point(i,1),label=point(i,1.23), anchor=Math.abs(label[0]-cx)<12?'middle':label[0]>cx?'start':'end';
    return `<line x1="${cx}" x2="${end[0]}" y1="${cy}" y2="${end[1]}"/><text class="radar-label" x="${label[0]}" y="${label[1]+5}" text-anchor="${anchor}">${esc(m.label)} ${m.direction==='higher'?'↑':'↓'}</text>`;
  }).join('')}${runs.map((run,i)=>{
    const color=colors[i%3];
    return `<polygon points="${metrics.map((m,k)=>point(k,run.normalized[m.key]).join(',')).join(' ')}" fill="${color}" fill-opacity=".045" stroke="${color}" stroke-width="2.3" stroke-dasharray="${dash[i%3]}"/>${metrics.map((m,k)=>{const p=point(k,run.normalized[m.key]);return `<circle cx="${p[0]}" cy="${p[1]}" r="4" fill="${color}"><title>${esc(run.name)} · ${esc(m.label)}: ${run.values[m.key].toFixed(5)} ${esc(m.unit||'')} · relative ${Math.round(run.normalized[m.key]*100)}</title></circle>`}).join('')}`;
  }).join('')}</svg>`;
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
    return `<text x="${left-15}" y="${y+5}" text-anchor="end">${esc(label)}<title>${esc(run.name)}</title></text><circle cx="${x(v)}" cy="${y}" r="6" fill="${colors[i%3]}"><title>${esc(run.name)}: ${v.toFixed(6)}</title></circle><text x="${w-right+12}" y="${y+5}">${v.toFixed(4)}</text>`;
  }).join('')}</svg>`;
}

function rawTable(data) {
  return `<div class="table-wrap profile-values"><table><caption class="sr-only">Unnormalized metric values behind the radar</caption><thead><tr><th>Selected model</th>${data.cohort.metrics.map(m=>`<th>${esc(m.label)} ${m.direction==='higher'?'↑':'↓'}</th>`).join('')}</tr></thead><tbody>${profileRuns(data).map((r,i)=>`<tr><th scope="row"><span class="profile-key" style="--series-color:${colors[i%3]}"></span><a href="#/run/${encodeURIComponent(r.id)}">${esc(r.name)}</a></th>${data.cohort.metrics.map(m=>`<td class="number">${r.values[m.key].toFixed(4)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function radarView(data) {
  if(!selectedProfiles)selectedProfiles=new Set(data.cohort.default_runs || data.cohort.runs.slice(0,3).map(r=>r.id));
  const c=data.cohort;
  return `<section class="visual-panel"><div class="visual-panel-heading"><div><div class="eyebrow">Example 03 / Different kinds of quality</div><h2>A profile, rather than a single winner</h2><p>${esc(c.dataset_name||c.dataset)} · ${esc(c.split)} · ${number(c.n)} tiles per run · ${esc(c.protocol)}</p></div><span class="chart-badge">Same cohort · same protocol</span></div>
    <div class="profile-picker"><span>Compare up to three models</span><div>${c.runs.map(r=>`<label><input type="checkbox" data-profile="${esc(r.id)}" ${selectedProfiles.has(r.id)?'checked':''}>${esc(r.name)}</label>`).join('')}</div><p id="profile-feedback" role="status" aria-live="polite"></p></div>
    <div class="profile-charts"><div><h3>Relative metric profile</h3><p class="small muted">Outward is better on each axis · scale 0–100</p><div id="profile-legend">${profileLegend(data)}</div><div id="radar-plot">${radarPlot(data)}</div></div><div class="profile-exact"><h3>Check the actual difference</h3><label class="field">Metric in original units<select id="profile-metric">${c.metrics.map(m=>`<option value="${esc(m.key)}" ${focusedMetric===m.key?'selected':''}>${esc(m.label)} · ${m.direction==='higher'?'higher':'lower'} is better${m.unit?' · '+esc(m.unit):''}</option>`).join('')}</select></label><label class="dot-zoom"><input type="checkbox" id="dot-zoom" ${zoomDots?'checked':''}>Zoom dot-plot axis to selected values</label><div id="metric-dots">${metricDots(data)}</div><div class="normalization-note"><strong>Keep the scale in view.</strong><p>Each radar axis uses the observed minimum and maximum across all ${number(c.runs.length)} eligible cohort runs. Changing the selection does not move the radar scale. The dot plot can zoom to reveal small raw differences; its labelled axis then changes with the selection.</p></div></div></div>
    <div id="profile-values">${rawTable(data)}</div>
    <p class="visual-caption">Radar scaling: higher-is-better = (value − minimum) / range; lower-is-better = (maximum − value) / range. Constant axes sit at 50. A larger polygon is not an overall score, and these single-seed numerical differences do not establish statistical superiority.</p>
  </section><div class="view-verdict horizontal"><span class="verdict-number">03</span><div><h3>Best for a small number of trade-offs</h3><p>Radar plots work for two or three profiles. The dot plot and exact values make the size of each difference easier to judge.</p></div><div class="verdict-limit"><strong>Use sparingly</strong><span>Many overlapping polygons become unreadable. Do not use radar area to rank the models.</span></div></div>`;
}

function activities(data) {
  const activity=data.activity;
  return `<section class="campaign-section" id="activity"><div class="visual-heading"><div><div class="eyebrow">Work continuing beyond the published scores</div><h2>The next chapter is already running.</h2></div><p>Activity observed ${date(activity.observed_utc || data.snapshot.activity)}<br>Saved evidence; this page does not poll the cluster.</p></div>
    <div class="campaign-grid"><article class="visual-panel"><div class="visual-panel-heading"><div><h3>HER2Match / 11 models</h3><p>21,172 paired images · author case splits</p></div><span class="chart-badge">New dataset</span></div><div class="campaign-runs">${activity.her2match.map(r=>{
      const progress=r.max_steps?Math.min(100,100*(r.step||0)/r.max_steps):0;
      return `<div class="campaign-run"><div><strong>${esc(r.name)}</strong><span>${esc(r.state)}</span></div><div class="campaign-progress"><i style="width:${progress}%"></i></div><small>${r.step?number(r.step)+' optimizer updates':r.production_started?'Production started':'Awaiting production'}${r.max_steps?' / '+number(r.max_steps):''}</small></div>`;
    }).join('')}</div><p class="visual-caption">Progress bars show the recorded update budget where available. Queued and unfinished runs carry no final scores.</p></article>
    <div><article class="visual-panel"><h3>Recoveries, with the evidence retained</h3><div class="recovery-list">${activity.recoveries.map(r=>`<div><span class="recovery-state ${r.complete?'done':''}" aria-hidden="true">${r.complete?'✓':'↗'}</span><div><strong>${esc(r.name)}</strong><span>${esc(r.state)}${r.step?' · '+number(r.step)+' updates':''}</span><p>${esc(r.note)}</p></div></div>`).join('')}</div></article>
    <article class="visual-panel metric-readiness"><h3>Pathology-aware evaluators</h3>${activity.evaluators.map(e=>`<div><strong>${esc(e.name)}</strong><span>${esc(e.state)}</span><p>${esc(e.note)}</p></div>`).join('')}</article></div></div>
    </section><details class="overview-provenance"><summary>How these visual examples are grounded</summary><p>The coverage counts come from the published benchmark snapshot. The separate campaign cards use dated execution receipts. These snapshots have different dates; new activity has not been promoted into final leaderboard scores.</p><ul>${(data.notes||[]).map(n=>`<li>${esc(n)}</li>`).join('')}</ul><a href="data/visual-summary.json" download>Download the chart data and evidence fingerprints</a></details>`;
}

function bind(main, data, layout) {
  const activityLink=$('a[href="#activity"]',main);
  if(activityLink)activityLink.onclick=e=>{e.preventDefault();$('#activity',main).scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});};
  $$('a.coverage-cell[href="#activity"],.dataset-bar-row[href="#activity"]',main).forEach(a=>a.onclick=e=>{e.preventDefault();$('#activity',main).scrollIntoView({behavior:'auto'});});
  if(layout==='curves')$$('[data-line]',main).forEach(input=>input.onchange=()=>{
    input.checked?selectedLines.add(input.dataset.line):selectedLines.delete(input.dataset.line);
    $('#learning-plot',main).innerHTML=learningPlot(data,selectedLines);
  });
  if(layout==='radar'){
    $$('[data-profile]',main).forEach(input=>input.onchange=()=>{
      if(input.checked&&selectedProfiles.size>=3){input.checked=false;$('#profile-feedback',main).textContent='Choose up to three models. Deselect one to add another.';return;}
      input.checked?selectedProfiles.add(input.dataset.profile):selectedProfiles.delete(input.dataset.profile);
      $('#profile-feedback',main).textContent=selectedProfiles.size?'':'Select a model to draw its profile.';
      $('#radar-plot',main).innerHTML=radarPlot(data);$('#metric-dots',main).innerHTML=metricDots(data);$('#profile-values',main).innerHTML=rawTable(data);$('#profile-legend',main).innerHTML=profileLegend(data);
    });
    $('#profile-metric',main).onchange=e=>{focusedMetric=e.target.value;$('#metric-dots',main).innerHTML=metricDots(data);};
    $('#dot-zoom',main).onchange=e=>{zoomDots=e.target.checked;$('#metric-dots',main).innerHTML=metricDots(data);};
  }
}

export async function renderOverview(main, layout='atlas') {
  if(!['atlas','curves','radar'].includes(layout))layout='atlas';
  if(!snapshotPromise)snapshotPromise=fetch('data/visual-summary.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('The visual summary could not load.');return r.json();});
  const data=await snapshotPromise;
  activeSummary=data;
  if(!location.hash.startsWith('#/overview')&&location.hash&&location.hash!=='#activity')return;
  main.innerHTML=`<div class="research-overview">${header(data,layout)}<section class="visual-example" id="visual-example" aria-label="${layout} example">${layout==='curves'?curveView(data):layout==='radar'?radarView(data):coverage(data)}</section>${activities(data)}</div>`;
  bind(main,data,layout);
}

matchMedia('(max-width:700px)').addEventListener('change',()=>{
  if(!activeSummary)return;
  if($('#learning-plot'))$('#learning-plot').innerHTML=learningPlot(activeSummary,selectedLines);
  if($('#metric-dots'))$('#metric-dots').innerHTML=metricDots(activeSummary);
});
