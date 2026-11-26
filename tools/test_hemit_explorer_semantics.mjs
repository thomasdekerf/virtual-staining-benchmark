import test from 'node:test';
import assert from 'node:assert/strict';
import {buildProfileCohorts, materializeProfileCohort} from '../dist/profile-data.js';
import {learningAxisKind} from '../dist/learning-explorer.js';

test('fluorescence radar has usable native axes and excludes incompletely defined channel correlation',()=>{
  const index={datasets:[{id:'hemit_dataset',name:'HEMIT',task:'he_to_mif'}],metrics:{
    mif_psnr:{label:'Fluorescence PSNR',direction:'↑'},mif_ssim:{label:'Fluorescence SSIM',direction:'↑'},
    mif_lpips:{label:'Fluorescence feature distance',direction:'↓'},mif_mae:{label:'Fluorescence MAE',direction:'↓'},
    mif_DAPI_pcc:{label:'DAPI Pearson correlation',direction:'↑'},
  },runs:[0,1].map(i=>({id:`model_${i}`,model:`model_${i}`,name:`Model ${i}`,variant:'native',
    task:'he_to_mif',datasets:['hemit_dataset'],marker:'DAPI / CD3 / panCK',crop_size:256,
    evaluations:[{dataset:'hemit_dataset',marker:'DAPI / CD3 / panCK',split:'test',rankable:true,n:945,
      protocol:'Native numerical fluorescence',manifest_sha256:'fixed-test-manifest',checkpoint_sha256:`checkpoint-${i}`,
      metrics:{mif_psnr:30-i,mif_ssim:.9-i*.1,mif_lpips:.2+i*.1,mif_mae:.02+i*.01,mif_DAPI_pcc:.8},
      coverage:{mif_DAPI_pcc:944}}]}))};
  const groups=buildProfileCohorts(index);
  assert.equal(groups.length,1);
  const cohort=materializeProfileCohort(index,groups[0]);
  assert.deepEqual(cohort.metrics.map(m=>m.key),['mif_psnr','mif_ssim','mif_lpips','mif_mae']);
  assert.equal(cohort.available_metrics.some(m=>m.key==='mif_DAPI_pcc'),false);
  assert.equal(cohort.runs[0].normalized.mif_lpips,1);
  assert.equal(cohort.runs[1].normalized.mif_lpips,0);
});

test('verified HEMIT update histories remain distinct from native epoch histories and legacy ambiguous positions',()=>{
  assert.equal(learningAxisKind({task:'he_to_mif',curve_axis:'step'}),'updates');
  assert.equal(learningAxisKind({task:'he_to_mif',curve_axis:'epoch'}),'epoch');
  assert.equal(learningAxisKind({task:'unstained',curve_axis:'step'}),'recorded');
});
