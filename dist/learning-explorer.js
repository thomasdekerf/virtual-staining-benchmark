// Interactive access to recorded scalar histories. Historical axis metadata is
// deliberately not promoted to optimizer updates without a verified convention.
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite = value => typeof value === 'number' && Number.isFinite(value);
const format = value => Number(value).toLocaleString('en-GB', {maximumFractionDigits: 5});
const palette = ['#4159dc','#b83475','#147b71','#ac6b16','#8252b6','#c64c37','#247b9e','#6b7830','#ad528f','#42699b','#968044','#4c8876'];
const detailCache = new Map();
let rememberedState;

function extent(values, project = value => value) {
  let low = Infinity, high = -Infinity;
  for (const row of values) { const value = project(row); if (value < low) low = value; if (value > high) high = value; }
  return [low, high];
}

export const learningScopeKey = run => [...(run.datasets || [])].sort().join('|');

export function learningAxisKind(run) {
  // The audited breast campaign exports global optimizer-step trajectories.
  // Some unstained epoch trainers were historically exported as "step" despite
  // recording epoch indices; their raw values must remain unconverted.
  if (['ihc','he_to_mif'].includes(run.task) && run.curve_axis === 'step') return 'updates';
  if (run.curve_axis === 'epoch') return 'epoch';
  return 'recorded';
}

export function normalizedProgress(points) {
  if (!points.length) return [];
  const first = points[0][0], span = points.at(-1)[0] - first;
  return points.map(([x,y]) => [span ? 100 * (x-first) / span : 0,y]);
}

function axisLabel(kind) {
  return kind === 'updates' ? 'Optimizer updates' : kind === 'epoch' ? 'Recorded epoch index' : 'Recorded position · units unverified';
}

function scalarName(index, tag) {
  const [part, ...rest] = tag.split('/');
  const key = rest.join('/');
  const label = index.metrics[key]?.label || key.replaceAll('_', ' ');
  return `${part === 'val' ? 'Validation' : part === 'train' ? 'Training' : part} / ${label}`;
}

function scopeLabel(key, index) {
  const names = key.split('|').map(id => index.datasets.find(d => d.id === id)?.name || id);
  return names.join(' + ') + (names.length > 1 ? ' · pooled' : ' · separate');
}

function selectedPosition(run, kind) {
  if (kind === 'updates') return run.position?.global_step;
  if (kind === 'epoch') return run.task === 'he_to_mif' ? run.selected_completed_epoch ?? run.position?.epoch : run.position?.epoch;
  return null;
}

function plotSVG(series, index, state, {small = false, fixedY = null} = {}) {
  if (!series.length) return '<div class="learning-empty">Select one or more runs to show their recorded history.</div>';
  const narrow = typeof matchMedia !== 'undefined' && matchMedia('(max-width:700px)').matches;
  const w = small || narrow ? 480 : 1000, h = small ? 320 : narrow ? 370 : 430;
  const pad = {left: 83, right: 25, top: 43, bottom: 62};
  const rows = series.map(row => ({...row, display: state.xmode === 'relative' ? normalizedProgress(row.points) : row.points}));
  const points = rows.flatMap(row => row.display);
  const allPositive = points.every(p => p[1] > 0);
  const logarithmic = state.yscale === 'log' && allPositive;
  const project = value => logarithmic ? Math.log10(value) : value;
  const [rawXMin, rawXMax] = extent(points, p=>p[0]);
  let xmin = state.xmode === 'relative' ? 0 : Math.min(0, rawXMin);
  let xmax = state.xmode === 'relative' ? 100 : rawXMax;
  if (xmax === xmin) xmax = xmin + 1;
  const measuredY = extent(points, p=>project(p[1]));
  let low = fixedY?.[0] ?? measuredY[0];
  let high = fixedY?.[1] ?? measuredY[1];
  const margin = (high-low || Math.abs(high)*.1 || .1)*.08;
  low -= margin; high += margin;
  const x = value => pad.left + (value-xmin)/(xmax-xmin)*(w-pad.left-pad.right);
  const y = value => h-pad.bottom-(project(value)-low)/(high-low)*(h-pad.top-pad.bottom);
  const kinds = new Set(rows.map(row => learningAxisKind(row.run)));
  const xlabel = state.xmode === 'relative' ? 'Observed training range (%)' : kinds.size === 1 ? axisLabel([...kinds][0]) : 'Recorded position · mixed units';
  const metricKey = state.tag.split('/').slice(1).join('/');
  const meta = index.metrics[metricKey];
  const ylabel = (meta?.label || metricKey.replaceAll('_',' ')) + (meta?.direction && meta.direction !== '—' ? ` ${meta.direction}` : '') + (logarithmic ? ' · log axis' : '');
  const grid = Array.from({length:5}, (_,i) => {
    const v = low+(high-low)*i/4, raw = logarithmic ? 10**v : v;
    const text = Math.abs(raw) >= 1000 ? Number(raw).toPrecision(3) : Math.abs(raw) > 0 && Math.abs(raw) < .001 ? raw.toExponential(1) : raw.toFixed(Math.abs(raw) >= 10 ? 1 : 3);
    const py = h-pad.bottom-i/4*(h-pad.top-pad.bottom);
    return `<line x1="${pad.left}" x2="${w-pad.right}" y1="${py}" y2="${py}"/><text x="${pad.left-12}" y="${py+4}" text-anchor="end">${text}</text>`;
  }).join('');
  const ticks = Array.from({length:5}, (_,i) => {
    const value = xmin+(xmax-xmin)*i/4;
    return `<text x="${x(value)}" y="${h-pad.bottom+26}" text-anchor="middle">${state.xmode === 'relative' ? Math.round(value) : format(Math.round(value))}</text>`;
  }).join('');
  const paths = rows.map(row => {
    const path = row.display.map(([px,py],i) => (i?'L':'M')+x(px).toFixed(2)+','+y(py).toFixed(2)).join(' ');
    const kind = learningAxisKind(row.run), chosen = selectedPosition(row.run,kind);
    const chosenIndex = finite(chosen) ? row.points.findIndex(p=>p[0] === chosen) : -1;
    let marker = '';
    if (chosenIndex >= 0) {
      const p = row.display[chosenIndex], px=x(p[0]), py=y(p[1]);
      marker = `<polygon points="${[[0,-5],[5,0],[0,5],[-5,0]].map(([dx,dy])=>(px+dx)+','+(py+dy)).join(' ')}" fill="white" stroke="${row.color}" stroke-width="2"><title>${escapeHTML(row.label)} selected checkpoint · recorded position ${format(chosen)}</title></polygon>`;
    }
    return `<path class="learning-series" data-history-id="${escapeHTML(row.run.id)}" d="${path}" fill="none" stroke="${row.color}" stroke-width="${small?2:2.2}" stroke-dasharray="${row.dash}" tabindex="0" aria-label="${escapeHTML(row.label)}: ${row.points.length} displayed ${escapeHTML(state.tag)} values"/>${marker}`;
  }).join('');
  return `<svg class="learning-svg ${small?'learning-small-svg':''}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${escapeHTML(scalarName(index,state.tag))} trajectories, ${escapeHTML(xlabel)}. Unsmoothed recorded values."><title>${escapeHTML(scalarName(index,state.tag))} trajectories</title><desc>Paths preserve recorded measurements. A diamond marks the selected checkpoint when its position and axis are verified.</desc>${grid}${ticks}<text class="learning-axis-label" x="${pad.left}" y="19">${escapeHTML(ylabel)}</text><text class="learning-axis-label" x="${(pad.left+w-pad.right)/2}" y="${h-9}" text-anchor="middle">${escapeHTML(xlabel)}</text>${paths}<line class="learning-crosshair" x1="0" x2="0" y1="${pad.top}" y2="${h-pad.bottom}" visibility="hidden"/><circle class="learning-hover-point" r="5" fill="white" stroke="#203259" stroke-width="2" visibility="hidden"/></svg>`;
}

function download(name, text, type) {
  const a = document.createElement('a'), url = URL.createObjectURL(new Blob([text], {type}));
  a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}

export async function renderLearningExplorer(container, index) {
  let alive = true, generation = 0, plotted = [];
  const datasets = index.datasets.filter(d=>index.runs.some(r=>r.datasets.includes(d.id) && Object.keys(r.curve_counts || {}).length));
  const initial = rememberedState || {dataset: datasets.find(d=>d.id === 'BCI')?.id || datasets[0]?.id, marker:'HER2', scope:'BCI', tag:'val/lpips', search:'', mode:'overlay', xmode:'recorded', yscale:'linear', sharedY:true, selected:new Set(['unet_BCI_HER2','pix2pix_BCI_HER2','restormer_BCI_HER2'])};
  const state = {...initial, selected: new Set(initial.selected)};
  rememberedState = state;
  const $ = selector => container.querySelector(selector);
  const $$ = selector => [...container.querySelectorAll(selector)];
  const baseRuns = () => index.runs.filter(r=>r.datasets.includes(state.dataset) && Object.keys(r.curve_counts || {}).length);
  const markers = () => [...new Set(baseRuns().map(r=>r.marker))].sort();
  const scopes = () => [...new Set(baseRuns().filter(r=>r.marker===state.marker).map(learningScopeKey))].sort((a,b)=>a.split('|').length-b.split('|').length || a.localeCompare(b));
  const scopeRuns = () => baseRuns().filter(r=>r.marker===state.marker && (state.scope==='all' || learningScopeKey(r)===state.scope));
  const tags = () => [...new Set(scopeRuns().flatMap(r=>Object.keys(r.curve_counts || {})))].sort((a,b)=>{const score=x=>['val/lpips','val/mif_lpips'].includes(x)?0:x.startsWith('val/')?1:2;return score(a)-score(b)||a.localeCompare(b);});
  const eligible = () => scopeRuns().filter(r=>(r.curve_counts?.[state.tag] || 0)>0).sort((a,b)=>a.name.localeCompare(b.name)||a.id.localeCompare(b.id));
  const visible = () => eligible().filter(r=>(r.name+' '+r.variant+' '+r.id).toLowerCase().includes(state.search.toLowerCase()));
  function repairFilters(reset=false) {
    if(!datasets.some(d=>d.id===state.dataset))state.dataset=datasets[0]?.id;
    if(!markers().includes(state.marker))state.marker=markers()[0];
    if(state.scope!=='all' && !scopes().includes(state.scope))state.scope=scopes()[0];
    if(!tags().includes(state.tag))state.tag=['val/mif_lpips','val/lpips'].find(t=>tags().includes(t))||tags()[0];
    const ids=new Set(eligible().map(r=>r.id));
    state.selected=new Set([...state.selected].filter(id=>ids.has(id)));
    if(reset && !state.selected.size)state.selected=new Set(eligible().slice(0,3).map(r=>r.id));
  }
  repairFilters(true);
  const option=(value,label,selected)=>`<option value="${escapeHTML(value)}" ${value===selected?'selected':''}>${escapeHTML(label)}</option>`;
  function filterMarkup() {
    return `<label>Dataset<select data-learning-control="dataset">${datasets.map(d=>option(d.id,d.name,state.dataset)).join('')}</select></label><label>Target stain<select data-learning-control="marker">${markers().map(m=>option(m,m,state.marker)).join('')}</select></label><label class="learning-scope-control">Training / validation scope<select data-learning-control="scope">${scopes().map(s=>option(s,scopeLabel(s,index),state.scope)).join('')}${scopes().length>1?option('all','All scopes containing this dataset',state.scope):''}</select></label><label>Recorded scalar<select data-learning-control="tag">${tags().map(t=>option(t,scalarName(index,t),state.tag)).join('')}</select></label>`;
  }
  function renderList() {
    const shown=visible(), all=eligible();
    $('#learning-selection-count').textContent=`${state.selected.size} selected / ${all.length} available runs`;
    $('#learning-visible-count').textContent=`${shown.length} matching the search`;
    $('#learning-select-all').textContent=state.search?`Select all ${shown.length} matching`:`Select all ${all.length}`;
    $('#learning-run-list').innerHTML=shown.map(run=>{
      const i=all.findIndex(r=>r.id===run.id), color=palette[i%palette.length], style=['solid','dashed','dotted'][Math.floor(i/palette.length)%3];
      return `<label class="learning-run-choice" style="--history-color:${color}"><input type="checkbox" data-learning-run="${escapeHTML(run.id)}" ${state.selected.has(run.id)?'checked':''}><i style="border-top-style:${style}"></i><span><strong>${escapeHTML(run.name)}</strong><small>${escapeHTML(run.variant)}${state.scope==='all'?' · '+escapeHTML(scopeLabel(learningScopeKey(run),index)):''}</small></span><em>${format(run.curve_counts[state.tag])} values</em></label>`;
    }).join('') || '<p class="learning-list-empty">No histories match this search.</p>';
    $$('[data-learning-run]').forEach(input=>input.onchange=()=>{input.checked?state.selected.add(input.dataset.learningRun):state.selected.delete(input.dataset.learningRun);renderList();renderPlot();});
  }
  function renderShell() {
    container.innerHTML=`<section class="visual-panel learning-explorer"><div class="visual-panel-heading"><div><div class="eyebrow">Learning explorer / All recorded histories</div><h2>Follow any model through training.</h2><p>Choose the dataset, training scope and scalar. Select any number of available runs; switch to separate panels when lines overlap.</p></div><span class="chart-badge">No three-model limit</span></div>
      <div class="learning-filters">${filterMarkup()}</div>
      <div class="learning-selection-toolbar"><div><strong id="learning-selection-count"></strong><span id="learning-visible-count"></span></div><div><button type="button" id="learning-select-all">Select all</button><button type="button" id="learning-clear">Clear</button><button type="button" id="learning-defaults">Example selection</button></div></div>
      <label class="learning-search">Find a model, recipe or run<input type="search" id="learning-search" placeholder="Search all available histories…" value="${escapeHTML(state.search)}"></label>
      <div id="learning-run-list" class="learning-run-list" aria-label="Available learning curves"></div>
      <div class="learning-plot-controls"><label>Display<select data-learning-control="mode">${option('overlay','Overlay selected runs',state.mode)}${option('multiples','One panel per run',state.mode)}</select></label><label>Horizontal axis<select data-learning-control="xmode">${option('recorded','Recorded training position',state.xmode)}${option('relative','Relative observed range (0–100%)',state.xmode)}</select></label><label>Vertical scale<select data-learning-control="yscale">${option('linear','Linear',state.yscale)}${option('log','Logarithmic (positive values)',state.yscale)}</select></label><label class="learning-shared"><input type="checkbox" id="learning-shared-y" ${state.sharedY?'checked':''}>Shared vertical scale in panels</label><button type="button" id="learning-export">Download plot-data CSV</button></div>
      <p id="learning-scope-note" class="learning-context-note"></p><p id="learning-load-state" class="learning-load-state" role="status" aria-live="polite"></p><div id="learning-chart-output"></div><p id="learning-inspect" class="learning-inspect" aria-live="polite">Hover over a curve, or focus one with the keyboard, to inspect a measured value.</p>
      <p class="visual-caption">Only recorded measurements are plotted, with no smoothing or extrapolation. Dense histories use the exported display points; each run’s full scalar CSV remains available on its detail page. ◇ marks a selected checkpoint where its axis is verified. Training objectives and effective batch sizes differ across recipes.</p>
      <details class="learning-axis-details"><summary>How to interpret the position axis and pooled histories</summary><p>Breast-campaign step positions are verified optimizer updates. Other historical exports can contain epoch indices despite a generic “step” label; these are shown as recorded positions with unverified units, without conversion. Unknown or mixed raw units use separate panels automatically. Relative range maps each run’s first and last recorded positions to 0 and 100%; it is not a fraction of a shared compute budget.</p><p>“Dataset” locates runs that trained on that dataset. Pooled histories aggregate their complete listed validation pool and are not dataset-specific validation curves. Choose a single training scope to compare the same recorded pool. Test measurements never enter these learning curves.</p></details>
      </section>`;
    renderList();
    $$('[data-learning-control]').forEach(input=>input.onchange=()=>{
      const key=input.dataset.learningControl;state[key]=input.value;
      if(['dataset','marker','scope','tag'].includes(key)){state.search='';repairFilters(true);renderShell();}else renderPlot();
    });
    $('#learning-search').oninput=e=>{state.search=e.target.value;renderList();};
    $('#learning-select-all').onclick=()=>{visible().forEach(r=>state.selected.add(r.id));renderList();renderPlot();};
    $('#learning-clear').onclick=()=>{state.selected.clear();renderList();renderPlot();};
    $('#learning-defaults').onclick=()=>{const ids=eligible().map(r=>r.id), wanted=['unet_BCI_HER2','pix2pix_BCI_HER2','restormer_BCI_HER2'].filter(id=>ids.includes(id));state.selected=new Set(wanted.length?wanted:ids.slice(0,3));renderList();renderPlot();};
    $('#learning-shared-y').onchange=e=>{state.sharedY=e.target.checked;renderPlot();};
    $('#learning-export').onclick=()=>{
      const cell=v=>'"'+String(v??'').replaceAll('"','""')+'"';
      const rows=[['run','model','training_scope','scalar','recorded_axis_kind','recorded_position','value','display_points','recorded_values']];
      plotted.forEach(row=>row.points.forEach(([x,y])=>rows.push([row.run.id,row.run.name,scopeLabel(learningScopeKey(row.run),index),state.tag,learningAxisKind(row.run),x,y,row.points.length,row.run.curve_counts[state.tag]])));
      download('learning-curve-plot-data.csv',rows.map(r=>r.map(cell).join(',')).join('\n'),'text/csv');
    };
    renderPlot();
  }
  function bindInspection(output, series, small) {
    output.querySelectorAll('.learning-svg').forEach((svg, panelIndex)=>{
      const candidates=small?[series[panelIndex]]:series;
      const show=(row,point)=>{$('#learning-inspect').textContent=`${row.label} · ${axisLabel(learningAxisKind(row.run))}: ${format(point[0])} · ${state.tag}: ${format(point[1])}`;};
      svg.querySelectorAll('[data-history-id]').forEach(path=>path.onfocus=()=>{
        const row=candidates.find(r=>r.run.id===path.dataset.historyId);if(row)show(row,row.points.at(-1));
      });
      svg.onpointermove=event=>{
        const rect=svg.getBoundingClientRect(),box=svg.viewBox.baseVal,w=box.width,h=box.height;
        const px=(event.clientX-rect.left)/rect.width*w, py=(event.clientY-rect.top)/rect.height*h;
        // The SVG path itself has the exact transformed geometry. Browser path
        // length sampling finds the closest curve without creating point nodes.
        let best=null;
        svg.querySelectorAll('[data-history-id]').forEach(path=>{
          const length=path.getTotalLength();
          for(let i=0;i<=70;i++){
            const p=path.getPointAtLength(length*i/70),distance=(p.x-px)**2+(p.y-py)**2;
            if(!best || distance<best.distance)best={path,p,distance};
          }
        });
        if(!best)return;
        const row=candidates.find(r=>r.run.id===best.path.dataset.historyId),display=state.xmode==='relative'?normalizedProgress(row.points):row.points;
        const all=candidates.flatMap(r=>state.xmode==='relative'?normalizedProgress(r.points):r.points),bounds=extent(all,p=>p[0]),xmin=state.xmode==='relative'?0:Math.min(0,bounds[0]),xmax=state.xmode==='relative'?100:bounds[1];
        const target=xmin+(px-83)/(w-83-25)*(xmax-xmin);
        let at=0;display.forEach((p,i)=>{if(Math.abs(p[0]-target)<Math.abs(display[at][0]-target))at=i;});
        show(row,row.points[at]);
        const cross=svg.querySelector('.learning-crosshair'),point=svg.querySelector('.learning-hover-point');
        const measuredX=83+(display[at][0]-xmin)/(xmax-xmin||1)*(w-83-25);
        let start=0,end=best.path.getTotalLength();
        for(let i=0;i<15;i++){const mid=(start+end)/2;if(best.path.getPointAtLength(mid).x<measuredX)start=mid;else end=mid;}
        const measured=best.path.getPointAtLength((start+end)/2);
        cross.setAttribute('x1',measured.x);cross.setAttribute('x2',measured.x);cross.setAttribute('visibility','visible');point.setAttribute('cx',measured.x);point.setAttribute('cy',measured.y);point.setAttribute('visibility','visible');
      };
      svg.onpointerleave=()=>svg.querySelectorAll('.learning-crosshair,.learning-hover-point').forEach(n=>n.setAttribute('visibility','hidden'));
    });
  }
  async function renderPlot() {
    const token=++generation, selected=eligible().filter(r=>state.selected.has(r.id));
    const message=$('#learning-load-state'), output=$('#learning-chart-output');
    $('#learning-export').disabled=!selected.length;
    $('#learning-scope-note').textContent=state.scope==='all'?'Multiple training scopes are selected. Pooled histories aggregate their entire listed validation pool; these are not isolated measurements on the selected dataset.':state.scope.includes('|')?`Pooled validation history: ${scopeLabel(state.scope,index)}. These measurements aggregate the full pool, rather than the selected dataset alone.`:`Single training scope: ${scopeLabel(state.scope,index)}. Scalar definitions, batches and training recipes can still differ.`;
    if(!selected.length){plotted=[];message.textContent='';output.innerHTML='<div class="learning-empty">No models selected. Choose individual runs or use “Select all”.</div>';return;}
    message.textContent=`Loading ${selected.length} recorded histories…`;
    try {
      const details=await Promise.all(selected.map(run=>{
        if(!detailCache.has(run.id))detailCache.set(run.id,fetch(`data/runs/${encodeURIComponent(run.id)}.json`).then(r=>{if(!r.ok)throw Error(`Could not load ${run.name}`);return r.json();}).catch(e=>{detailCache.delete(run.id);throw e;}));
        return detailCache.get(run.id);
      }));
      if(!alive || !container.isConnected || token!==generation)return;
      const all=eligible();
      plotted=details.map(run=>{
        const colorIndex=all.findIndex(r=>r.id===run.id),points=(run.curves[state.tag]||[]).filter(p=>finite(p[0])&&finite(p[1]));
        return {run,points,label:run.name+(state.scope==='all'?' · '+scopeLabel(learningScopeKey(run),index):''),color:palette[colorIndex%palette.length],dash:['','8 4','2 4'][Math.floor(colorIndex/palette.length)%3]};
      }).filter(row=>row.points.length);
      const points=plotted.flatMap(row=>row.points),positive=points.every(p=>p[1]>0);
      if(state.yscale==='log'&&!positive){state.yscale='linear';$('[data-learning-control="yscale"]').value='linear';}
      const kinds=new Set(plotted.map(row=>learningAxisKind(row.run)));
      const unverified=state.xmode==='recorded'&&(kinds.size>1||kinds.has('recorded'));
      const multiples=state.mode==='multiples'||unverified;
      message.textContent=`${plotted.length} runs · ${format(points.length)} displayed measurements`+(unverified?' · Separate panels keep unverified or mixed position units apart.':'')+(!positive?' · Nonpositive values require a linear vertical scale.':'');
      if(!plotted.length){output.innerHTML='<div class="learning-empty">No finite measurements were recorded for this selection.</div>';return;}
      const logarithmic=state.yscale==='log';
      const values=points.map(p=>logarithmic?Math.log10(p[1]):p[1]);
      const fixedY=state.sharedY?extent(values):null;
      output.innerHTML=multiples?`<div class="learning-multiples">${plotted.map(row=>`<article class="learning-mini-panel"><div><a href="#/run/${encodeURIComponent(row.run.id)}" style="color:${row.color}">${escapeHTML(row.run.name)}</a><span>${escapeHTML(axisLabel(learningAxisKind(row.run)))}</span></div>${plotSVG([row],index,state,{small:true,fixedY})}<p>${format(row.points.length)} displayed / ${format(row.run.curve_counts[state.tag])} recorded values · <a href="downloads/${encodeURIComponent(row.run.id)}_curves.csv" download>full scalar CSV</a></p></article>`).join('')}</div>`:`<div class="learning-overlay-legend">${plotted.map(row=>`<a href="#/run/${encodeURIComponent(row.run.id)}"><i style="border-color:${row.color};border-top-style:${row.dash?'dashed':'solid'}"></i>${escapeHTML(row.run.name)}</a>`).join('')}</div>${plotSVG(plotted,index,state)}`;
      bindInspection(output,plotted,multiples);
    } catch(error) {
      if(alive&&token===generation){message.textContent='';output.innerHTML=`<div class="learning-empty">${escapeHTML(error.message)}. Select the runs again to retry loading.</div>`;}
    }
  }
  renderShell();
  let resizeTimer;
  const resized=()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(alive&&container.isConnected)renderPlot();},120);};
  window.addEventListener('resize',resized);
  return ()=>{alive=false;generation++;clearTimeout(resizeTimer);window.removeEventListener('resize',resized);};
}
