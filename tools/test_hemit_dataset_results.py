"""Scientific boundaries for numerical fluorescence and continuation lineage."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from hemit_dataset_results import (CHANNELS,merge_allocations,merge_history,
    numerical_metrics,replay_patience,verify_gate,metric_glossary)

class HemitDatasetTests(unittest.TestCase):
    def test_conditional_pcc_preserves_undefined_coverage(self):
        channels={c:dict(mae=.02,psnr=30.,ssim=.8,pcc_mean_defined_tiles=.75,
            pcc_defined_tiles=8,pcc_total_tiles=10) for c in CHANNELS}
        record=dict(n=10,per_dataset={'HEMIT':dict(metrics=dict(psnr=29.,ssim=.7,mae=.03,lpips=.2,
            pcc_sum_per_tile_DAPI=.6),fluorescence_channels=channels)})
        values,coverage,public=numerical_metrics(record)
        self.assertEqual(values['mif_DAPI_pcc'],.75)
        self.assertEqual(coverage['mif_DAPI_pcc'],8)
        self.assertEqual(public['DAPI']['pcc_undefined_tiles'],2)
        self.assertNotIn('pcc_sum_per_tile_DAPI',values)
        record['per_dataset']['HEMIT']['fluorescence_channels']['DAPI']['pcc_total_tiles']=9
        with self.assertRaises(ValueError):numerical_metrics(record)

    def test_empty_learned_proof_and_missing_gan_branch_fail_closed(self):
        gate=dict(passed=True,source_sha256='a'*64,world_size=4,validation_tiles_recomputed=9,
            full_width=True,resume=True,positive_learning_rate=True,actual_model_tensor_update=True,
            resumed_step=4,first_leg_step=3,actual_learned_optimizer_resume=True,test_used=False,
            native_learned_evidence={})
        with self.assertRaises(ValueError):verify_gate(gate,'a'*64,True,'pix2pix')
        gate['native_learned_evidence']['generator']=dict(learned_parameters_sha256='b'*64,
            parameter_states=2,min_optimizer_step=4,learning_rates=[.0001])
        with self.assertRaises(ValueError):verify_gate(gate,'a'*64,True,'pix2pix')
        verify_gate(gate,'a'*64,True,'unet')
        gate['native_learned_evidence']['generator']['learning_rates']=[0.]
        with self.assertRaises(ValueError):verify_gate(gate,'a'*64,True,'unet')

    def test_fresh_patience_excludes_pre_horizon_plateau(self):
        minimum=dict(kind='updates',horizon=20000)
        history=[dict(step=x,validation=dict(lpips=.5 if x==1000 else .6)) for x in range(1000,29000,1000)]
        self.assertEqual(replay_patience(history[:20],minimum,.0001),0)
        self.assertEqual(replay_patience(history,minimum,.0001),8)
        history[-2]['validation']['lpips']=.4
        self.assertEqual(replay_patience(history,minimum,.0001),1)

    def test_parent_history_and_compute_are_counted_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            inherited=dict(epoch=1,global_step=116,validation=dict(lpips=.5),train=dict(loss=2.))
            later=dict(epoch=2,global_step=232,validation=dict(lpips=.4),train=dict(loss=1.))
            parent=directory/'parent';child=directory/'child'
            parent.write_text(json.dumps(inherited)+'\n')
            child.write_text('\n'.join(map(json.dumps,[inherited,later]))+'\n')
            self.assertEqual(merge_history([parent,child],'epoch')['val/mif_lpips'],[[1,.5],[2,.4]])
            changed=copy.deepcopy(inherited);changed['validation']['lpips']=.3
            child.write_text(json.dumps(changed)+'\n')
            with self.assertRaises(ValueError):merge_history([parent,child],'epoch')
            allocation=dict(job_id='123',restart=0,start_unix=0,end_unix=3600,seconds=3600,world_size=4)
            parent.write_text(json.dumps(allocation)+'\n');child.write_text(json.dumps(allocation)+'\n')
            self.assertEqual(merge_allocations([parent,child])['gpu_hours'],4.)

    def test_raw_pcc_curve_has_explicit_total_denominator(self):
        glossary=metric_glossary()
        for channel in CHANNELS:
            self.assertIn('TOTAL tiles',glossary[f'mif_pcc_sum_per_tile_{channel}']['description'])
            self.assertIn('coverage',glossary[f'mif_pcc_valid_fraction_{channel}']['description'])

if __name__=='__main__':unittest.main()
