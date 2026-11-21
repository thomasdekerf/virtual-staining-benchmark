const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const count=value=>Number(value||0).toLocaleString('en-GB');
const $=(selector,root=document)=>root.querySelector(selector);
const $$=(selector,root=document)=>[...root.querySelectorAll(selector)];
let snapshotPromise,generation=0,cleanup;
const state={dataset:null,field:null,query:'',missing:true,sticky:true,size:250,channel:null};

export function matchedPredictions(dataset,field){
  const records=new Map();
  for(const prediction of field?.predictions||[]){
    if(records.has(prediction.model))throw Error('Duplicate model prediction for one tissue field.');
    records.set(prediction.model,prediction);
  }
  return (dataset.models||[]).map(model=>{
    const prediction=records.get(model.id);
    let reason=prediction?.reason||model.reason||'A matched selected-checkpoint prediction has not been exported.';
    let available=!!prediction?.url&&prediction.available!==false;
    if(available&&(!/^[a-f0-9]{64}$/.test(prediction.sha256||'')||!/^[a-f0-9]{64}$/.test(model.checkpoint_sha256||'')||!prediction.checkpoint_sha256||!prediction.run_id||!model.run_id||prediction.field_id==null||!prediction.target_sha256||!field.target?.sha256)){available=false;reason='The field, reference image and selected checkpoint identities have not all been verified.';}
    if(prediction?.run_id&&model.run_id&&prediction.run_id!==model.run_id){available=false;reason='This image belongs to a different training run.';}
    if(prediction?.checkpoint_sha256&&model.checkpoint_sha256&&prediction.checkpoint_sha256!==model.checkpoint_sha256){available=false;reason='The selected checkpoint identity does not match this image.';}
    if(prediction?.field_id&&prediction.field_id!==field.id){available=false;reason='This image belongs to a different tissue field.';}
    if(prediction?.target_sha256&&field.target?.sha256&&prediction.target_sha256!==field.target.sha256){available=false;reason='The matched chemical reference identity differs.';}
    if(available&&field.input?.url&&prediction.input_sha256!==field.input.sha256){available=false;reason='The original input identity has not been verified against this field.';}
    return {model,prediction,available,reason};
  });
}

export function chooseGalleryField(dataset,identity){
  return dataset.fields?.find(f=>String(f.id)===String(identity))||dataset.fields?.[0]||null;
}

function inputImage(dataset,field){
  const channels=field?.input?.channels||[];
  const selected=channels.find(c=>c.id===state.channel);
  return selected?{...field.input,...selected}:field?.input;
}

function imageTag(record,alt,options=''){
  return record?.url?`<img src="${esc(record.url)}" alt="${esc(alt)}" width="256" height="256" decoding="async" ${options}>`:`<div class="gallery-image-unavailable"><span>Image unavailable</span>${record?.reason?`<small>${esc(record.reason)}</small>`:''}</div>`;
}

function inputLabel(dataset){
  return dataset.input_label||({orion:'Original multiplex input',BCI:'Original H&E',MIST:'Original H&E',IHC4BC:'Original H&E',her2match:'Original H&E',finland:'Original unstained tissue',skin:'Original unstained tissue'}[dataset.id])||'Original input';
}

export function disposePredictionGallery(){
  generation++;
  if(typeof cleanup==='function')cleanup();
  cleanup=null;
}

export async function renderPredictionGallery(main,index,query=new URLSearchParams()){
  const sequence=++generation;
  if(!snapshotPromise)snapshotPromise=fetch('data/prediction-gallery.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('The matched tissue gallery could not load.');return r.json();}).catch(error=>{snapshotPromise=null;throw error;});
  const data=await snapshotPromise;
  if(sequence!==generation||!location.hash.startsWith('#/gallery'))return;
  if(!Array.isArray(data.datasets)||!data.datasets.length)throw Error('No gallery dataset evidence is available.');
  const requested=query.get('dataset')||state.dataset;
  const populated=dataset=>dataset.fields?.some(field=>field.input?.url&&field.target?.url&&field.predictions?.some(p=>p.available!==false&&p.url));
  const initial=data.datasets.find(d=>d.id==='her2match'&&populated(d))||data.datasets.find(populated)||data.datasets[0];
  state.dataset=data.datasets.some(d=>d.id===requested)?requested:initial.id;
  if(query.has('field'))state.field=query.get('field');
  const knownRuns=new Set(index.runs.map(r=>r.id));
  const abort=new AbortController();cleanup=()=>{abort.abort();$('#prediction-inspector',main)?.close();};
  function draw(){
    if(sequence!==generation)return;
    const dataset=data.datasets.find(d=>d.id===state.dataset),field=chooseGalleryField(dataset,state.field);
    state.field=field?.id||null;
    const rows=matchedPredictions(dataset,field).sort((a,b)=>Number(b.available)-Number(a.available)||(a.model.name||a.model.id).localeCompare(b.model.name||b.model.id)),available=rows.filter(r=>r.available);
    const selectedInput=inputImage(dataset,field),channels=field?.input?.channels||[];
    const datasetInfo=index.datasets.find(d=>d.id===dataset.id);
    const targetLabel=dataset.target_label||'Chemical '+(dataset.marker||'H&E')+' reference';
    const selection=typeof dataset.selection?.rule==='string'?dataset.selection.rule:'Fixed test fields selected independently of model outputs.';
    const modelTitle=model=>model.name||model.id;
    main.innerHTML=`<div class="prediction-gallery" style="--gallery-card-size:${state.size}px"><div class="intro gallery-intro"><div><div class="eyebrow">Matched tissue comparisons</div><h1>One tissue field. Every model.</h1><p>Choose a dataset and a field. Compare model predictions against the same original input and chemically stained reference.</p></div><a class="gallery-back" href="#/overview">Back to the research atlas</a></div>
      <section class="gallery-selection panel" aria-label="Choose a tissue field"><div class="gallery-filters"><label class="field">Dataset<select id="gallery-dataset">${data.datasets.map(d=>`<option value="${esc(d.id)}" ${d.id===dataset.id?'selected':''}>${esc(d.name||d.id)} · ${count(d.fields?.length)} fields</option>`).join('')}</select></label><label class="field">Tissue field<select id="gallery-field" ${!field?'disabled':''}>${(dataset.fields||[]).map((f,i)=>`<option value="${esc(f.id)}" ${String(f.id)===String(field?.id)?'selected':''}>${i+1}. ${esc(f.label||f.source_id||f.id)}</option>`).join('')}</select></label><div class="gallery-cohort"><strong>${esc(dataset.marker||datasetInfo?.markers?.[0]||'H&E')} · ${esc(dataset.split||'test')}</strong><span>${count(dataset.crop_size||256)}px matched crops · ${count(dataset.fields?.length)} fixed test fields</span></div></div>
      <div class="gallery-field-strip" aria-label="Available tissue fields">${(dataset.fields||[]).map((f,i)=>`<button type="button" data-gallery-field="${esc(f.id)}" ${f.id===field?.id?'aria-pressed="true"':'aria-pressed="false"'} title="${esc(f.label||f.source_id||f.id)}">${imageTag(f.target||f.input,'Field '+(i+1),'loading="lazy"')}<span>Field ${i+1}</span></button>`).join('')}</div><p class="gallery-selection-note">${esc(selection)} ${dataset.selection?.note?esc(dataset.selection.note):''}</p></section>
      ${field?`<section class="gallery-reference ${state.sticky?'is-sticky':''}" aria-label="Original input and chemical reference"><figure><figcaption>${esc(inputLabel(dataset))}</figcaption>${imageTag(selectedInput,inputLabel(dataset))}${channels.length?`<label class="field gallery-channel-label">Input display<select id="gallery-input-channel"><option value="">${esc(field.input.label||'Default display')}</option>${channels.map(c=>`<option value="${esc(c.id)}" ${state.channel===c.id?'selected':''}>${esc(c.label||c.id)}</option>`).join('')}</select></label>`:''}${field.input?.kind==='channel_visualization'||dataset.id==='orion'?`<p class="gallery-channel-note">${esc(selectedInput?.description||field.input?.description||'A display of the recorded multiplex channels. The original model input has more than three channels.')}</p>`:''}</figure><figure><figcaption>${esc(targetLabel)}</figcaption>${imageTag(field.target,targetLabel)}<p class="gallery-reference-label">Same crop for every prediction below</p>${field.target?.compression?`<p class="gallery-source-note">${esc(field.target.compression)}. Some cached ORION predictions also retain JPEG compression.</p>`:''}</figure><div class="gallery-field-context"><div class="eyebrow">Field ${Math.max(1,(dataset.fields||[]).indexOf(field)+1)} / ${count(dataset.fields.length)}</div><h2>${esc(field.label||field.id)}</h2><p>${esc(field.group||field.source_id||'Fixed held-out tissue field')}</p><div class="gallery-availability"><strong>${count(available.length)}</strong><span>matched predictions<br>${count(rows.length-available.length)} unavailable</span><button type="button" id="gallery-jump">View predictions</button></div><p class="gallery-context-note">${esc(dataset.notes||'Predictions use the recorded selected checkpoint. Fields were fixed before inspecting model outputs.')}</p><label class="gallery-sticky-switch"><input type="checkbox" id="gallery-sticky" ${state.sticky?'checked':''}>Keep references visible while scrolling</label><details><summary>Field identity and provenance</summary><dl><dt>Manifest index</dt><dd>${esc(field.index??'Not recorded')}</dd><dt>Source field</dt><dd>${esc(field.source_id||field.id)}</dd><dt>Manifest SHA-256</dt><dd>${esc(dataset.manifest_sha256||'Not recorded')}</dd><dt>Reference SHA-256</dt><dd>${esc(field.target?.sha256||'Not recorded')}</dd></dl></details></div></section>`:`<section class="panel empty"><h2>Matched images are being prepared</h2><p>${esc(dataset.reason||'No verified gallery fields have been exported for this dataset yet.')}</p></section>`}
      <section class="gallery-model-section" aria-label="Model predictions"><div class="gallery-model-toolbar"><div><h2>Compare the predictions</h2><p id="gallery-visible-count">${count(available.length)} of ${count(rows.length)} model predictions available for this field</p></div><label class="field">Find a model<input type="search" id="gallery-model-search" value="${esc(state.query)}" placeholder="Model or training recipe…"></label><label class="field gallery-size-control">Image size<input id="gallery-size" type="range" min="180" max="360" step="10" value="${state.size}"></label><label class="gallery-missing-switch"><input type="checkbox" id="gallery-show-missing" ${state.missing?'checked':''}>Show unavailable models</label></div><div class="gallery-model-grid">${rows.map((row,i)=>{
        const model=row.model,prediction=row.prediction,name=modelTitle(model);
        return `<article class="gallery-model-card ${row.available?'':'is-unavailable'}" data-gallery-model="${i}" data-gallery-search="${esc((name+' '+(model.variant||'')+' '+(model.run_id||'')).toLowerCase())}"><header><h3>${knownRuns.has(model.run_id)?`<a href="#/run/${encodeURIComponent(model.run_id)}">${esc(name)}</a>`:esc(name)}</h3><span>${row.available?'Selected checkpoint':'Image unavailable'}</span></header>${row.available?`<button type="button" class="gallery-image-button" data-gallery-inspect="${i}" aria-label="Inspect ${esc(name)} on the selected tissue field">${imageTag(prediction,name+' prediction of '+(field?.label||field?.id),'loading="lazy"')}<span>Inspect against original and reference</span></button>`:`<div class="gallery-missing-image"><span aria-hidden="true">—</span><p>${esc(row.reason)}</p></div>`}<footer>${model.selected_step!=null?`<span>Selected update ${count(model.selected_step)}</span>`:''}${model.variant?`<span>${esc(model.variant)}</span>`:''}${row.available?`<details><summary>Image provenance</summary><dl><dt>Run</dt><dd>${esc(model.run_id||prediction.run_id)}</dd><dt>Checkpoint SHA-256</dt><dd>${esc(prediction.checkpoint_sha256||model.checkpoint_sha256||'Not recorded')}</dd><dt>Image SHA-256</dt><dd>${esc(prediction.sha256||'Not recorded')}</dd></dl></details>`:''}</footer></article>`;
      }).join('')}</div><p class="gallery-empty-search" hidden>No models match this filter.</p></section><p class="gallery-bottom-note">All cards use the selected tissue field. Unavailable predictions retain their recorded reason. <a href="data/prediction-gallery.json" download>Download gallery identities and provenance</a></p><dialog id="prediction-inspector" class="prediction-inspector"><div class="gallery-dialog-heading"><div><div class="eyebrow">Inspect the same tissue field</div><h2 id="gallery-inspector-title"></h2></div><button type="button" id="gallery-inspector-close">Close</button></div><label class="field gallery-zoom-control">Magnification<select id="gallery-inspector-zoom"><option value="1">1× native pixels</option><option value="2" selected>2×</option><option value="4">4×</option></select></label><div class="gallery-dialog-images" style="--inspection-scale:2"></div><p class="gallery-inspector-caption"></p></dialog></div>`;
    function syncHash(){
      const parameters=new URLSearchParams({dataset:dataset.id});if(field)parameters.set('field',String(field.id));
      history.replaceState(null,'','#/gallery?'+parameters);
    }
    function filter(){
      const cards=$$('[data-gallery-model]',main);let shown=0;
      cards.forEach(card=>{const row=rows[Number(card.dataset.galleryModel)];card.hidden=(!state.missing&&!row.available)||!card.dataset.gallerySearch.includes(state.query.toLowerCase());if(!card.hidden)shown++;});
      $('.gallery-empty-search',main).hidden=shown>0;
      $('#gallery-visible-count',main).textContent=`${count(shown)} model cards shown · ${count(available.length)} verified predictions available`;
    }
    $('#gallery-dataset',main).onchange=event=>{state.dataset=event.target.value;state.field=null;state.channel=null;draw();};
    $('#gallery-field',main).onchange=event=>{state.field=event.target.value;draw();};
    $$('[data-gallery-field]',main).forEach(button=>button.onclick=()=>{state.field=button.dataset.galleryField;draw();});
    $('#gallery-model-search',main).oninput=event=>{state.query=event.target.value;filter();};
    $('#gallery-show-missing',main).onchange=event=>{state.missing=event.target.checked;filter();};
    $('#gallery-size',main).oninput=event=>{state.size=Number(event.target.value);$('.prediction-gallery',main).style.setProperty('--gallery-card-size',state.size+'px');};
    if($('#gallery-jump',main))$('#gallery-jump',main).onclick=()=>{const reference=$('.gallery-reference',main);const offset=getComputedStyle(reference).position==='sticky'?reference.getBoundingClientRect().height+25:15;window.scrollTo({top:window.scrollY+$('.gallery-model-section',main).getBoundingClientRect().top-offset,behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth'});};
    if($('#gallery-sticky',main))$('#gallery-sticky',main).onchange=event=>{state.sticky=event.target.checked;$('.gallery-reference',main).classList.toggle('is-sticky',state.sticky);};
    if($('#gallery-input-channel',main))$('#gallery-input-channel',main).onchange=event=>{state.channel=event.target.value;draw();};
    const dialog=$('#prediction-inspector',main);
    $('#gallery-inspector-close',main).onclick=()=>dialog.close();
    dialog.onclick=event=>{if(event.target===dialog){const box=dialog.getBoundingClientRect();if(event.clientX<box.left||event.clientX>box.right||event.clientY<box.top||event.clientY>box.bottom)dialog.close();}};
    $('#gallery-inspector-zoom',main).onchange=event=>{
      $('.gallery-dialog-images',main).style.setProperty('--inspection-scale',event.target.value);
      $$('.gallery-dialog-images figure>div',main).forEach(pane=>pane.scrollTo(0,0));
    };
    $$('[data-gallery-inspect]',main).forEach(button=>button.onclick=()=>{
      const row=rows[Number(button.dataset.galleryInspect)];if(!row.available)return;
      $('#gallery-inspector-title',main).textContent=modelTitle(row.model)+' · '+(field.label||field.id);
      $('.gallery-dialog-images',main).innerHTML=[[inputLabel(dataset),selectedInput],[targetLabel,field.target],[modelTitle(row.model),row.prediction]].map(([label,image])=>`<figure><figcaption>${esc(label)}</figcaption><div>${imageTag(image,label)}</div></figure>`).join('');
      const panes=$$('.gallery-dialog-images figure>div',main);let syncing=false;
      panes.forEach(pane=>pane.onscroll=()=>{
        if(syncing)return;syncing=true;
        panes.filter(other=>other!==pane).forEach(other=>other.scrollTo(pane.scrollLeft,pane.scrollTop));
        requestAnimationFrame(()=>{syncing=false;});
      });
      $('.gallery-inspector-caption',main).textContent=`${dataset.name||dataset.id} · ${dataset.marker||'H&E'} · ${dataset.split||'test'} · manifest field ${field.index??field.id}. Magnification and panning stay aligned across all three images.`;
      dialog.showModal();
    });
    filter();syncHash();
  }
  draw();
  document.addEventListener('keydown',event=>{if(event.key==='Escape')$('#prediction-inspector',main)?.close();},{signal:abort.signal});
}
