"""Prevent submitted jobs, smoke counters or partial splits becoming green scores."""
import copy
import unittest

from her2_campaigns import (campaign_summary,digest,scheduler_state,verify_continuation_gate,
                            verify_final_report,verify_parent_reuse,verify_resume_gate,preparation_summary,TARGET_MODELS)
from followup_results import verify_gpu_snapshot


def config():
    return dict(seed=42,data=dict(datasets=['HER2Match'],marker='HER2'),model=dict(name='dgr',base_channels=64),loss=dict(l1=1),train=dict(batch_size=1,lr=1e-4,max_seconds=2700),validation=dict(lpips=True),breast_protocol=dict(manifest_sha256=dict(train='a'*64,val='b'*64,test='c'*64),status='untrained'))


def receipt(cfg):
    return dict(passed=True,source_sha256='d'*64,world_size=4,resume=True,full_width=True,validation_tiles_recomputed=9,resumed_step=4,first_leg_step=2,positive_learning_rate=True,actual_model_tensor_update=True,first_leg_model_tensor_sha256='e'*64,resumed_model_tensor_sha256='f'*64,recipe=copy.deepcopy(cfg))


def result(cfg):
    full={split:dict(n=n,manifest_sha256=cfg['breast_protocol']['manifest_sha256'][split],per_dataset={'HER2Match':dict(n=n,metrics=dict(psnr=25.,ssim=.8,lpips=.2,mae=.05,delta_e76=8.),dab=dict(samples=n,mean_absolute_error=.1,correlation=.8,mean_pred=.2,mean_target=.2))}) for split,n in [('val',3582),('test',5980)]}
    return dict(source_sha256='d'*64,checkpoint_sha256='e'*64,full_splits=full,full_split_counts=dict(val=3582,test=5980),protocol=copy.deepcopy(cfg['breast_protocol']),clinical_score_evaluation=False)


class Her2CampaignTests(unittest.TestCase):
    def setUp(self):self.cfg=config();self.task=dict(model='dgr',source_sha256='d'*64)

    def test_actual_native_gate_requires_changed_tensors_and_positive_lr(self):
        self.assertTrue(verify_resume_gate(receipt(self.cfg),self.cfg,self.task))
        for update in [dict(actual_model_tensor_update=False),dict(positive_learning_rate=False),dict(resumed_model_tensor_sha256='e'*64),dict(world_size=1),dict(source_sha256='0'*64)]:
            with self.subTest(update=update),self.assertRaises(ValueError):verify_resume_gate(receipt(self.cfg)|update,self.cfg,self.task)

    def test_gate_rejects_smaller_architecture_or_changed_scientific_recipe(self):
        r=receipt(self.cfg);r['recipe']['model']['base_channels']=1
        with self.assertRaises(ValueError):verify_resume_gate(r,self.cfg,self.task)
        r=receipt(self.cfg);r['recipe']['train']['batch_size']=2
        with self.assertRaises(ValueError):verify_resume_gate(r,self.cfg,self.task)

    def test_operational_budget_and_protocol_status_are_not_scientific_changes(self):
        changed=copy.deepcopy(self.cfg);changed['train']['max_seconds']=10800;changed['breast_protocol']['status']='production'
        self.assertEqual(digest(changed),digest(self.cfg))
        changed['train']['lr']=0
        self.assertNotEqual(digest(changed),digest(self.cfg))

    def test_final_requires_full_exact_splits_and_finite_metrics(self):
        self.assertTrue(verify_final_report(result(self.cfg),self.cfg,self.task))
        for change in ['partial','manifest','source','nan','clinical','counts_missing']:
            r=result(self.cfg)
            if change=='partial':r['full_splits']['test']['n']=10
            if change=='manifest':r['full_splits']['test']['manifest_sha256']='f'*64
            if change=='source':r['source_sha256']='a'*64
            if change=='nan':r['full_splits']['test']['per_dataset']['HER2Match']['metrics']['lpips']=float('nan')
            if change=='clinical':r['clinical_score_evaluation']=True
            if change=='counts_missing':del r['full_split_counts']
            with self.subTest(change=change),self.assertRaises(ValueError):verify_final_report(r,self.cfg,self.task)

    def test_undefined_dab_correlation_does_not_invent_a_zero_or_hide_valid_core(self):
        r=result(self.cfg);r['full_splits']['val']['per_dataset']['HER2Match']['dab']['correlation']=None
        self.assertTrue(verify_final_report(r,self.cfg,self.task))
        r['full_splits']['val']['per_dataset']['HER2Match']['dab']['correlation']=float('nan')
        with self.assertRaises(ValueError):verify_final_report(r,self.cfg,self.task)

    def test_unchanged_parent_is_referenced_without_duplicate_scored_rows(self):
        r=result(self.cfg)
        parent=dict(evaluations=[dict(split=s,n=v['n'],manifest_sha256=v['manifest_sha256'],checkpoint_sha256=r['checkpoint_sha256'],rankable=True,metrics=v['per_dataset']['HER2Match']['metrics'].copy()) for s,v in r['full_splits'].items()])
        self.assertTrue(verify_parent_reuse(r,parent))
        r['full_splits']['test']['per_dataset']['HER2Match']['metrics']['lpips']=.1
        with self.assertRaises(ValueError):verify_parent_reuse(r,parent)
        r['checkpoint_sha256']='f'*64
        self.assertFalse(verify_parent_reuse(r,parent))

    def test_cut_stage_requires_both_positive_lr_optimizers_and_parameter_updates(self):
        task=dict(source_sha256='d'*64,parent_step=20000,parent_checkpoint_sha256='e'*64,continuation_kind='explicit_positive_lr_extension')
        provenance=dict(source_sha256='d'*64,parent_checkpoint_sha256='e'*64,manifest_sha256=self.cfg['breast_protocol']['manifest_sha256'])
        gate=dict(passed=True,source_sha256='d'*64,world_size=4,parent_step=20000,observed_step=20001,test_used=False,positive_learning_rates={'optimizer':[1e-5],'discriminator_optimizer':[1e-5]},parameter_updates={'generators':True,'discriminators':True},original_stopping_reference_preserved=True)
        self.assertTrue(verify_continuation_gate(gate,provenance,self.cfg,task))
        gate['parameter_updates']['discriminators']=False
        with self.assertRaises(ValueError):verify_continuation_gate(gate,provenance,self.cfg,task)

    def test_parent_progress_is_not_new_work_and_queue_is_not_a_score(self):
        r=dict(scheduler=dict(state='PENDING'),files={})
        self.assertEqual(scheduler_state(r,{},False),('Queued / native update check required','PENDING'))
        r['scheduler']['state']='RUNNING'
        self.assertEqual(scheduler_state(r,{},False),('Native update check running','RUNNING'))
        self.assertEqual(scheduler_state(r,dict(complete=True),True)[0],'Training complete / evaluation pending')

    def test_one_field_cellpose_probe_is_never_a_full_production_map_set(self):
        raw=dict(used_splits=['train'],source_rows=11610,train_manifest_sha256='6c8cb9f704e5df8062b15c78b1d6226569e46dd7e24fcb919dbbcf6869fe7cb2',cellpose_version='2.2.3',nuclei_models=['nuclei','cyto2'],prepared_rows=1,complete=False,net_avg=True)
        item=dict(kind='probe',job='1',gpus=1,required_rows=1,receipt=dict(data=raw),cached_map_files=1,accounting=dict(state='COMPLETED'))
        self.assertFalse(preparation_summary(dict(preparation=[item]))[0]['complete'])
        self.assertTrue(preparation_summary(dict(preparation=[item]))[0]['probe_passed'])
        item.update(kind='full_maps',required_rows=11610)
        with self.assertRaises(ValueError):preparation_summary(dict(preparation=[item]))
        raw.update(prepared_rows=11610,complete=True)
        with self.assertRaises(ValueError):preparation_summary(dict(preparation=[item]))
        item['cached_map_files']=11610
        self.assertTrue(preparation_summary(dict(preparation=[item]))[0]['complete'])
        raw['used_splits']=['train','val']
        with self.assertRaises(ValueError):preparation_summary(dict(preparation=[item]))

    def test_preparation_gpus_are_not_counted_as_model_training(self):
        record=dict(jobs=[dict(status='RUNNING',gpus=4)],own_training_gpus=4,own_running_training_jobs=1,preparation_jobs=[dict(status='RUNNING',gpus=1)],own_preparation_gpus=1,partition=dict(allocated_gpus=5,schedulable_gpus=8,physical_gpus=8,free_schedulable_gpus=3,largest_free_gpus_per_node=3))
        verify_gpu_snapshot(record)
        record['own_preparation_gpus']=0
        with self.assertRaises(ValueError):verify_gpu_snapshot(record)

    def test_a_complete_evaluation_does_not_erase_a_collapse_warning(self):
        index=dict(her2match_campaign={},models=[dict(id=mid,name=mid) for mid in TARGET_MODELS],runs=[])
        index['her2match_campaign']=dict(snapshot='2026-11-23T00:00:00+00:00')
        index['runs']=[dict(id='dgr-original',model='dgr',datasets=['her2match'],evaluations=[dict(dataset='her2match',split='test',rankable=True)],quality_failure=True,quality_warning='Colour collapse')]
        campaign=campaign_summary(index)
        self.assertEqual(campaign['counts']['evaluated'],1)
        self.assertEqual(campaign['quality_warning_count'],1)
        self.assertEqual(next(row for row in campaign['rows'] if row['model']=='dgr')['quality_warning'],'Colour collapse')

    def test_dgr_rescue_must_update_g_reggt_late_regx_and_discriminator(self):
        task=self.task|dict(requires_decoupled_registration_gate=True,smoke_steps=104)
        r=receipt(self.cfg);r.update(first_leg_step=1,resumed_step=104)
        def item(step,letter):return dict(max_optimizer_step=step,parameter_states=int(step>0),learning_rates=[1e-5],learned_parameters_sha256=letter*64)
        before=dict(native_dgr_evidence={name:item(step,'a') for name,step in [('generator',1),('reg_gt',1),('reg_x',0),('discriminator',0)]})
        r['native_dgr_evidence']={name:item(step,'b') for name,step in [('generator',104),('reg_gt',104),('reg_x',4),('discriminator',26)]}
        self.assertTrue(verify_resume_gate(r,self.cfg,task,before))
        for name in ['reg_x','discriminator']:
            bad=copy.deepcopy(r);bad['native_dgr_evidence'][name]['learned_parameters_sha256']='a'*64
            with self.assertRaises(ValueError):verify_resume_gate(bad,self.cfg,task,before)
        r['resumed_step']=4
        with self.assertRaises(ValueError):verify_resume_gate(r,self.cfg,task,before)

    def test_specialist_gate_requires_all_three_resumed_native_systems(self):
        task=self.task|dict(model='pgvms');r=receipt(self.cfg)
        def evidence(step,letter):return {name:dict(max_optimizer_step=step,min_optimizer_step=step,parameter_states=1,learned_parameters_sha256=letter*64,learning_rates=[1e-4]) for name in ['generator','discriminator','projections']}
        r['native_specialist_evidence']=evidence(4,'a');before=dict(native_specialist_evidence=evidence(2,'b'))
        self.assertTrue(verify_resume_gate(r,self.cfg,task,before))
        del r['native_specialist_evidence']['projections']
        with self.assertRaises(ValueError):verify_resume_gate(r,self.cfg,task,before)


if __name__=='__main__':unittest.main()
