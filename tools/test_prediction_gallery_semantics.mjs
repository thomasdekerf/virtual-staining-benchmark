// A gallery comparison must keep tissue field, selected run/checkpoint and
// chemical reference identities together, including when an image is missing.
import test from 'node:test';
import assert from 'node:assert/strict';
import {matchedPredictions, chooseGalleryField} from '../dist/prediction-gallery.js';

const digest = letter => letter.repeat(64);
const model = (id='first') => ({id, name:id, run_id:'run-'+id,
  checkpoint_sha256:digest(id==='first'?'a':'b')});
const prediction = (id='first', overrides={}) => ({model:id,
  run_id:'run-'+id, field_id:'test-0', target_sha256:digest('c'), input_sha256:digest('e'),
  checkpoint_sha256:digest(id==='first'?'a':'b'), sha256:digest('d'),
  url:'assets/gallery/'+id+'.png', available:true, ...overrides});
const field = (predictions=[prediction()], overrides={}) => ({id:'test-0',index:0,
  input:{url:'assets/gallery/input.png',sha256:digest('e'),available:true},
  target:{url:'assets/gallery/target.png',sha256:digest('c')},predictions,...overrides});
const dataset = (models=[model(),model('second')], fields=[]) => ({id:'BCI',models,fields});

test('gallery only offers an exactly bound selected-checkpoint prediction', () => {
  const [first,second]=matchedPredictions(dataset(),field());
  assert.equal(first.available,true);
  assert.equal(first.model.id,'first');
  assert.equal(first.prediction.field_id,'test-0');
  assert.equal(second.available,false);
  assert.equal(second.prediction,undefined);
});

for(const [key,value,reason] of [
  ['field_id','test-1','tissue field'],
  ['run_id','different-run','training run'],
  ['checkpoint_sha256',digest('f'),'checkpoint'],
  ['target_sha256',digest('e'),'reference'],
  ['input_sha256',digest('f'),'input'],
]) test('gallery rejects a prediction bound to another '+key, () => {
  const [row]=matchedPredictions(dataset(),field([prediction('first',{[key]:value})]));
  assert.equal(row.available,false);
  assert.match(row.reason,new RegExp(reason));
});

for(const key of ['field_id','run_id','checkpoint_sha256','target_sha256','input_sha256'])
  test('gallery rejects a prediction with missing '+key+' identity', () => {
    const image=prediction();delete image[key];
    assert.equal(matchedPredictions(dataset(),field([image]))[0].available,false);
  });

test('gallery rejects duplicate model predictions within one field', () => {
  assert.throws(()=>matchedPredictions(dataset(),field([prediction(),prediction()])),/Duplicate/);
});

test('a missing model never borrows another model or another field image', () => {
  const current=field([prediction('second')]);
  const another=field([prediction()],{id:'test-1',index:1});
  const scope=dataset(undefined,[current,another]);
  const [missing,available]=matchedPredictions(scope,current);
  assert.equal(missing.model.id,'first');
  assert.equal(missing.available,false);
  assert.equal(missing.prediction,undefined);
  assert.equal(available.model.id,'second');
  assert.equal(available.available,true);
});

test('explicit unavailable status and its recorded reason take precedence over a URL', () => {
  const [row]=matchedPredictions(dataset(),field([prediction('first',{
    available:false,reason:'Original selected checkpoint retired to archive.'})]));
  assert.equal(row.available,false);
  assert.equal(row.reason,'Original selected checkpoint retired to archive.');
});

test('field selection uses an exact identity and supports an empty cohort', () => {
  const first=field(),second=field([],{id:'test-1',index:1});
  const scope=dataset(undefined,[first,second]);
  assert.equal(chooseGalleryField(scope,'test-1'),second);
  assert.equal(chooseGalleryField(scope,'test-0'),first);
  assert.equal(chooseGalleryField(scope,null),first);
  assert.equal(chooseGalleryField(dataset(),null),null);
});
