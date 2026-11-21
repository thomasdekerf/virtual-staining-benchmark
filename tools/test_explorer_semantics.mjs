// Scientific semantics of the public explorers. These fixtures exercise cases
// that can change an apparent result: protocol mixing, missing values, metric
// direction, tied Pareto candidates and ambiguous learning-curve axes.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {
  cohortKey, cohortRows, normalizeValue, paretoFront, pearsonPairs,
} from '../dist/data-laboratory.js';
import {
  learningScopeKey, learningAxisKind, normalizedProgress,
} from '../dist/learning-explorer.js';
import {
  buildProfileCohorts, materializeProfileCohort,
} from '../dist/profile-data.js';

const finiteRows = [
  {id:'cheap',values:{x:1,y:8}},
  {id:'quality',values:{x:2,y:9}},
  {id:'third',values:{x:3,y:7}},
  {id:'tied',values:{x:1,y:8}},
  {id:'missing',values:{x:null,y:20}},
  {id:'invalid',values:{x:1,y:NaN}},
];
const ids = rows => rows.map(r=>r.id).sort();

test('Pareto fronts honor both metric directions and retain exact ties', () => {
  assert.deepEqual(ids(paretoFront(finiteRows,'x','y','lower','higher')), ['cheap','quality','tied']);
  assert.deepEqual(ids(paretoFront(finiteRows,'x','y','lower','lower')), ['cheap','third','tied']);
  assert.deepEqual(ids(paretoFront(finiteRows,'x','y','higher','lower')), ['third']);
  assert.deepEqual(ids(paretoFront(finiteRows,'x','y','higher','higher')), ['quality','third']);
  assert.deepEqual(paretoFront(finiteRows,'x','y','unknown','higher'), []);
});

test('direction aligned normalization retains missing values and treats constants neutrally', () => {
  assert.equal(normalizeValue(0,0,20,'higher'), 0);
  assert.equal(normalizeValue(0,0,20,'lower'), 1);
  assert.equal(normalizeValue(20,0,20,'lower'), 0);
  assert.equal(normalizeValue(10,10,10,'higher'), .5);
  assert.equal(normalizeValue(null,0,20,'higher'), null);
  assert.equal(normalizeValue(undefined,0,20,'higher'), null);
  assert.equal(normalizeValue(NaN,0,20,'higher'), null);
  assert.equal(normalizeValue(Infinity,0,20,'higher'), null);
});

test('correlation uses paired finite observations, never substitutes zero for missing metrics', () => {
  const rows=[
    {values:{x:1,y:2}}, {values:{x:2,y:4}}, {values:{x:3,y:6}},
    {values:{x:4,y:null}}, {values:{x:undefined,y:8}},
    {values:{x:NaN,y:3}}, {values:{x:7,y:Infinity}},
  ];
  const result=pearsonPairs(rows,'x','y');
  assert.equal(result.n,3);
  assert.ok(Math.abs(result.r-1)<1e-12);
  const anti=pearsonPairs([{values:{x:1,y:3}},{values:{x:2,y:2}},{values:{x:3,y:1}}],'x','y');
  assert.equal(anti.n,3);
  assert.ok(Math.abs(anti.r+1)<1e-12);
  assert.equal(pearsonPairs([{values:{x:1,y:2}},{values:{x:2,y:4}}],'x','y').r,null);
  assert.equal(pearsonPairs([{values:{x:1,y:2}},{values:{x:1,y:3}},{values:{x:1,y:4}}],'x','y').r,null);
});

function evaluation(overrides={}) {
  return {id:'evaluation',dataset:'finland',marker:'H&E',split:'test',protocol:'Complete',
    metric_protocol:'same-native-crops',manifest_sha256:'cohort-a',crop_size:256,n:10,
    rankable:true,metrics:{psnr:20,lpips:.3,ssim:.7},coverage:{psnr:10,lpips:10,ssim:10},...overrides};
}
function run(id, overrides={}) {
  return {id,model:id,name:id,variant:'one recipe',datasets:['finland'],task:'unstained',
    marker:'H&E',crop_size:256,evaluations:[evaluation({id:id+'-evaluation'})],...overrides};
}

test('scatter cohorts separate manifests, sample counts, splits and protocols', () => {
  const index={runs:[
    run('a'),run('b'),
    run('manifest',{evaluations:[evaluation({manifest_sha256:'cohort-b'})]}),
    run('subset',{evaluations:[evaluation({n:5})]}),
    run('validation',{evaluations:[evaluation({split:'val'})]}),
    run('protocol',{evaluations:[evaluation({metric_protocol:'different-crops'})]}),
    run('crop',{crop_size:512,evaluations:[evaluation({crop_size:512})]}),
    run('pending',{evaluations:[evaluation({rankable:false})]}),
  ]};
  const rows=cohortRows(index,{dataset:'finland',marker:'H&E',split:'test',protocol:'Complete'});
  assert.deepEqual(ids(rows),['a','b']);
  assert.notEqual(cohortKey(evaluation()),cohortKey(evaluation({n:5})));
  assert.notEqual(cohortKey(evaluation()),cohortKey(evaluation({manifest_sha256:'cohort-b'})));
  assert.notEqual(cohortKey(evaluation()),cohortKey(evaluation({metric_protocol:'different-crops'})));
  assert.notEqual(cohortKey(evaluation()),cohortKey(evaluation({crop_size:512})));
  assert.notEqual(cohortKey(evaluation({crop_size:undefined}),{crop_size:256}),
    cohortKey(evaluation({crop_size:undefined}),{crop_size:512}));
});

test('learning scopes distinguish pooled validation from dataset-specific validation', () => {
  assert.equal(learningScopeKey({datasets:['MIST','BCI']}),learningScopeKey({datasets:['BCI','MIST']}));
  assert.notEqual(learningScopeKey({datasets:['BCI']}),learningScopeKey({datasets:['BCI','MIST']}));
  assert.equal(learningAxisKind({task:'ihc',curve_axis:'step'}),'updates');
  assert.equal(learningAxisKind({task:'unstained',curve_axis:'step',model:'hemit'}),'recorded');
  assert.equal(learningAxisKind({task:'unstained',curve_axis:'epoch'}),'epoch');
});

test('relative learning progress changes only the x scale and preserves measured values', () => {
  const points=[[100,.7],[150,.4],[300,.5]];
  const copy=structuredClone(points);
  const relative=normalizedProgress(points);
  assert.deepEqual(relative,[[0,.7],[25,.4],[100,.5]]);
  assert.deepEqual(points,copy);
});

test('radar grouping cannot combine different scientific comparison scopes', () => {
  const index={datasets:[{id:'finland',name:'Finland'}],metrics:{psnr:{label:'PSNR',direction:'↑',unit:'dB'},lpips:{label:'LPIPS',direction:'↓',unit:'distance'},ssim:{label:'SSIM',direction:'↑',unit:'score'}},runs:[
    run('a'),run('b',{evaluations:[evaluation({id:'b-evaluation',metrics:{psnr:24,lpips:.2,ssim:.8}})]}),
    run('other-split',{evaluations:[evaluation({split:'val'})]}),
    run('other-samples',{evaluations:[evaluation({n:5})]}),
    run('other-crops',{crop_size:512,evaluations:[evaluation({crop_size:512})]}),
    run('other-protocol',{evaluations:[evaluation({metric_protocol:'other'})]}),
  ]};
  const groups=buildProfileCohorts(index);
  const group=groups.find(g=>g.entries.some(entry=>entry.run.id==='a'));
  assert.ok(group);
  assert.deepEqual(group.entries.map(e=>e.run.id).sort(),['a','b']);
  const cohort=materializeProfileCohort(index,group);
  assert.equal(cohort.runs.length,2);
  const first=cohort.runs.find(r=>r.id==='a');
  const second=cohort.runs.find(r=>r.id==='b');
  assert.equal(first.normalized.psnr,0);
  assert.equal(second.normalized.psnr,1);
  assert.equal(first.normalized.lpips,0);
  assert.equal(second.normalized.lpips,1);
});

test('radar axes omit metrics measured on only a subset of the common cohort', () => {
  const index={datasets:[{id:'finland',name:'Finland'}],metrics:{psnr:{label:'PSNR',direction:'↑',unit:'dB'},lpips:{label:'LPIPS',direction:'↓',unit:'distance'},ssim:{label:'SSIM',direction:'↑',unit:'score'}},runs:[
    run('a'),run('b',{evaluations:[evaluation({coverage:{psnr:10,lpips:10,ssim:5},metrics:{psnr:24,lpips:.2,ssim:.8}})]}),
  ]};
  const group=buildProfileCohorts(index).find(g=>g.entries.some(e=>e.run.id==='a'));
  const cohort=materializeProfileCohort(index,group);
  assert.ok(cohort.metrics.some(m=>m.key==='psnr'));
  assert.ok(cohort.metrics.some(m=>m.key==='lpips'));
  assert.ok(!cohort.metrics.some(m=>m.key==='ssim'));
});

test('published exploratory rows remain confined to one scored cohort', async () => {
  const index=JSON.parse(await readFile(new URL('../dist/data/index.json',import.meta.url),'utf8'));
  const rows=cohortRows(index,{dataset:'finland',marker:'H&E',split:'test',protocol:'Full production evaluation'});
  const expected=index.runs.flatMap(run=>run.evaluations.filter(e=>e.rankable&&e.dataset==='finland'&&e.split==='test'&&e.protocol==='Full production evaluation'));
  assert.equal(rows.length,expected.length);
  assert.ok(rows.length>=16);
  assert.equal(new Set(rows.map(r=>cohortKey(r.evaluation))).size,1);
  assert.ok(rows.every(r=>r.evaluation.rankable && r.evaluation.n===5179));
  const groups=buildProfileCohorts(index);
  assert.ok(groups.some(g=>g.dataset==='BCI' && g.marker==='HER2' && g.split==='test'));
});

test('incomplete and continuation compute cannot become a complete cost axis', () => {
  const index={runs:[run('complete',{training_hours:10,gpu_hours:40}),
    run('partial',{training_hours:2,gpu_hours:8,compute_complete:false}),
    run('child',{training_hours:1,gpu_hours:4,compute_parent_run:'parent'})]};
  const rows=cohortRows(index,{dataset:'finland',split:'test'});
  assert.equal(rows.find(r=>r.id==='complete').values.gpu_hours,40);
  for(const id of ['partial','child']){
    assert.equal(rows.find(r=>r.id===id).values.training_hours,undefined);
    assert.equal(rows.find(r=>r.id===id).values.gpu_hours,undefined);
  }
  assert.deepEqual(paretoFront(rows,'gpu_hours','lpips','lower','lower').map(r=>r.id),['complete']);
});

test('reviewed wording aliases preserve every other scientific cohort boundary', () => {
  const first=evaluation({metric_protocol:'reviewed original wording',comparison_protocol:'reviewed paired protocol',crop_size:256});
  const second={...first,metric_protocol:'reviewed newer wording'};
  assert.equal(cohortKey(first),cohortKey(second));
  for(const [key,value] of Object.entries({dataset:'other',marker:'HER2',split:'val',protocol:'another measurement',n:11,manifest_sha256:'other-manifest',crop_size:512})){
    assert.notEqual(cohortKey(first),cohortKey({...second,[key]:value}),key+' boundary was erased');
  }
  const {comparison_protocol:_,...rawFirst}=first;
  const {comparison_protocol:__,...rawSecond}=second;
  assert.notEqual(cohortKey(rawFirst),cohortKey(rawSecond),'unreviewed wording was silently normalized');
});

test('published breast comparisons include the new families while preserving raw protocols', async () => {
  const index=JSON.parse(await readFile(new URL('../dist/data/index.json',import.meta.url),'utf8'));
  const groups=buildProfileCohorts(index);
  for(const dataset of ['BCI','MIST','IHC4BC','her2match']){
    const group=groups.find(g=>g.dataset===dataset && g.marker==='HER2' && g.split==='test' && g.entries.some(e=>e.run.id.startsWith('balance__')));
    assert.ok(group, dataset+' reviewed cohort missing');
    const expected=new Set(index.runs.filter(run=>run.evaluations.some(e=>e.rankable&&e.dataset===dataset&&e.marker==='HER2'&&e.split==='test'&&e.comparison_protocol===group.metric_protocol)).map(run=>run.model));
    assert.ok(expected.size>=13,dataset+' lost the previously reviewed families');
    assert.equal(new Set(group.entries.map(e=>e.run.model)).size,expected.size,dataset+' lost eligible families');
    assert.equal(new Set(group.entries.map(e=>e.evaluation.metric_protocol)).size,2,'raw source protocol descriptions changed');
    assert.ok(group.entries.every(e=>e.evaluation.comparison_protocol_review_sha256 && e.evaluation.crop_size===256));
  }
});
