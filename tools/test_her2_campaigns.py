"""Prevent submitted jobs, smoke counters or partial splits becoming green scores."""
import copy
import unittest

from her2_campaigns import (campaign_summary,digest,scheduler_state,verify_continuation_gate,
                            verify_final_report,verify_parent_reuse,verify_resume_gate,preparation_summary,TARGET_MODELS,
                            apply_checkpoint_bound_mcs_findings,build_capacity)
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

    def test_draining_existing_allocations_remain_in_actual_capacity(self):
        nodes=[dict(name='draining',gpus=8,allocated_gpus=8,free_gpus=0,usable=False),
               dict(name='usable',gpus=8,allocated_gpus=5,free_gpus=3,usable=True),
               dict(name='down',gpus=8,allocated_gpus=0,free_gpus=0,usable=False)]
        record=build_capacity(dict(records={},parsed_nodes=nodes,captured_utc='2026-11-23T21:35:22Z'),{}, {})
        self.assertEqual(record['partition']['schedulable_gpus'],16)
        self.assertEqual(record['partition']['allocated_on_draining_nodes'],8)
        verify_gpu_snapshot(record)

    def test_cross_dataset_quality_findings_require_the_exact_mcs_checkpoint(self):
        rid='skin__mcs_stain_seed42'
        runs={rid:dict(id=rid,model='mcs_stain',datasets=['skin'],checkpoint_sha256='a'*64)}
        audit=dict(source_files_unchanged=True,scope='Read-only selected-checkpoint audit',
                   full_evaluation_receipts=dict(skin=dict(checkpoint_sha256='a'*64)))
        self.assertEqual(apply_checkpoint_bound_mcs_findings(runs,audit),[rid])
        self.assertTrue(runs[rid]['quality_failure'])
        self.assertNotIn('evaluations',runs[rid])
        audit['full_evaluation_receipts']['skin']['checkpoint_sha256']='b'*64
        with self.assertRaises(ValueError):apply_checkpoint_bound_mcs_findings(runs,audit)

    def test_dclgan_requires_each_native_direction_to_resume_learned_updates(self):
        task=self.task|dict(model='dclgan')
        def item(name,step,letter):return dict(max_optimizer_step=step,min_optimizer_step=step,
            parameter_states=8,parameters=11_378_179 if name.startswith('G_') else 2_764_737 if name.startswith('D_') else 493_568,
            learning_rates=[.001 if name.startswith('F') else .0002],
            betas=[[.9,.999] if name.startswith('F') else [.5,.999]],weight_decays=[0.],learned_parameters_sha256=letter*64)
        prior=dict(native_architecture_evidence={name:item(name,2,'a') for name in ['G_A','G_B','D_A','D_B','F1','F2']})
        passed=receipt(self.cfg)|dict(native_architecture_evidence={name:item(name,4,'b') for name in prior['native_architecture_evidence']})
        self.assertTrue(verify_resume_gate(passed,self.cfg,task,prior))
        for name in prior['native_architecture_evidence']:
            broken=copy.deepcopy(passed)
            broken['native_architecture_evidence'][name]['learned_parameters_sha256']='a'*64
            with self.subTest(direction=name),self.assertRaises(ValueError):verify_resume_gate(broken,self.cfg,task,prior)
        broken=copy.deepcopy(passed);del broken['native_architecture_evidence']['F2']
        with self.assertRaises(ValueError):verify_resume_gate(broken,self.cfg,task,prior)

    @staticmethod
    def native_system(step,letter,count):
        return dict(max_optimizer_step=step,min_optimizer_step=step,parameter_states=8,
                    parameters=count,learning_rates=[.0001],betas=[[.9,.999]],weight_decays=[0.],learned_parameters_sha256=letter*64)

    def test_uformer_gate_requires_actual_full_b_capacity_and_adam_updates(self):
        cfg=copy.deepcopy(self.cfg);cfg['train'].update(betas=[.9,.999],weight_decay=0.)
        task=self.task|dict(model='uformer');r=receipt(cfg)
        before=dict(native_architecture_evidence=dict(generator=self.native_system(2,'a',50_880_946)))
        r['native_architecture_evidence']=dict(generator=self.native_system(4,'b',50_880_946))
        for state in [before,r]:
            state['native_architecture_evidence']['generator']['rng_resume']=dict(
                scheme='stateless_epoch_rank_rng_v1',world_size=4,rank_states=4,loader_states=4)
        self.assertTrue(verify_resume_gate(r,cfg,task,before))
        broken=copy.deepcopy(r);del broken['native_architecture_evidence']['generator']['rng_resume']
        with self.assertRaises(ValueError):verify_resume_gate(broken,cfg,task,before)
        r['native_architecture_evidence']['generator']['parameters']=1_000_000
        with self.assertRaises(ValueError):verify_resume_gate(r,cfg,task,before)

    def test_i2sb_gate_requires_independent_online_ema_and_native_initialization(self):
        cfg=copy.deepcopy(self.cfg);cfg['model']['adm_sha256']='9'*64
        task=self.task|dict(model='i2sb',pretrained_assets=[dict(sha256='9'*64)])
        before=dict(native_architecture_evidence=dict(online=self.native_system(2,'a',552_814_083),
            ema=dict(learned_parameters_sha256='a'*64,ema_updates=2,decay=.99),pretrained_asset_sha256='9'*64))
        r=receipt(cfg)|dict(native_architecture_evidence=dict(online=self.native_system(4,'b',552_814_083),
            ema=dict(learned_parameters_sha256='b'*64,ema_updates=4,decay=.99),pretrained_asset_sha256='9'*64))
        self.assertTrue(verify_resume_gate(r,cfg,task,before))
        for key,value in [('learned_parameters_sha256','a'*64),('ema_updates',2),('decay',.999)]:
            broken=copy.deepcopy(r);broken['native_architecture_evidence']['ema'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):verify_resume_gate(broken,cfg,task,before)
        with self.assertRaises(ValueError):verify_resume_gate(r,cfg,dict(task,pretrained_assets=[]),before)

    def test_pix2pixhd_gate_requires_both_native_networks_and_unchanged_vgg(self):
        cfg=copy.deepcopy(self.cfg);cfg['pix2pixhd']=dict(vgg19=dict(sha256='9'*64))
        task=self.task|dict(model='pix2pixhd',pretrained_assets=[dict(sha256='9'*64)])
        teacher=dict(parameters_frozen=True,training=False,checkpoint_sha256='9'*64,
                     initial_parameters_sha256='e'*64,parameters_sha256='e'*64)
        before=dict(native_architecture_evidence=dict(generator=self.native_system(2,'a',182_443_267),
            discriminator=self.native_system(2,'a',5_535_618),fixed_feature_teacher=copy.deepcopy(teacher)))
        r=receipt(cfg)|dict(native_architecture_evidence=dict(generator=self.native_system(4,'b',182_443_267),
            discriminator=self.native_system(4,'b',5_535_618),fixed_feature_teacher=copy.deepcopy(teacher)))
        self.assertTrue(verify_resume_gate(r,cfg,task,before))
        r['native_architecture_evidence']['fixed_feature_teacher']['parameters_sha256']='f'*64
        with self.assertRaises(ValueError):verify_resume_gate(r,cfg,task,before)

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
