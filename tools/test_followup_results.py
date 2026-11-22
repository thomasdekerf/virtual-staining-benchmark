import copy
import json
import unittest

from followup_results import apply_live_execution, new_research_validation, verify_completed_integration, verify_gpu_snapshot, verify_native_gate


class FollowupResultsTests(unittest.TestCase):
    def test_native_gate_rejects_a_different_source_or_reduced_verification(self):
        gate=dict(passed=True,world_size=4,validation_tiles_recomputed=9,resumed_step=4,first_leg_step=2,source_sha256='source')
        verify_native_gate(gate,'source')
        for key,value in [('source_sha256','other'),('world_size',1),('validation_tiles_recomputed',8),('resumed_step',2),('passed',False)]:
            with self.subTest(key=key),self.assertRaises(ValueError):verify_native_gate(dict(gate,**{key:value}),'source')

    def test_complete_report_preserves_dataset_counts_and_manifests(self):
        task=dict(source_sha256='source',dataset='HER2Match')
        config=dict(breast_protocol=dict(manifest_sha256=dict(val='v',test='t')))
        counts=dict(val=3582,test=5980)
        core=dict(psnr=20.,ssim=.5,lpips=.4,mae=.1,delta_e76=10.)
        report=dict(source_sha256='source',checkpoint_sha256='a'*64,full_split_counts=counts,
            full_splits={split:dict(manifest_sha256=config['breast_protocol']['manifest_sha256'][split],n=n,
                per_dataset={'HER2Match':dict(n=n,metrics=core)}) for split,n in counts.items()})
        verify_completed_integration(report,config,task,counts)
        for key,value in [('manifest_sha256','other'),('n',5979),('per_dataset',{'other':dict(n=5980)})]:
            changed=copy.deepcopy(report);changed['full_splits']['test'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):verify_completed_integration(changed,config,task,counts)
        changed=copy.deepcopy(report);del changed['full_splits']['test']['per_dataset']['HER2Match']['metrics']['lpips']
        with self.assertRaises(ValueError):verify_completed_integration(changed,config,task,counts)

    def test_native_gate_must_use_the_actual_production_model_and_data(self):
        config=dict(data=dict(crop_size=256),model=dict(embed_dim=180),loss=dict(l1=1),breast_protocol=dict(marker='HER2'))
        gate=dict(passed=True,world_size=4,validation_tiles_recomputed=9,resumed_step=4,first_leg_step=2,source_sha256='source',full_width=True,resume=True,recipe=copy.deepcopy(config))
        verify_native_gate(gate,'source',config)
        config['breast_protocol']['status']='untrained'
        gate['recipe']['breast_protocol']['status']='production'
        verify_native_gate(gate,'source',config)
        gate['recipe']['model']['embed_dim']=8
        with self.assertRaises(ValueError):verify_native_gate(gate,'source',config)

    def test_inherited_checkpoint_is_not_reported_as_new_work(self):
        run={};record=dict(step=80000,parent_step=80000,new_optimizer_updates=0,gate_passed=False,job_id='1_0',scheduler=dict(state='PENDING',reason='Resources'))
        apply_live_execution(run,record,'now')
        self.assertTrue(run['registered_only']);self.assertEqual(run['new_optimizer_updates'],0)
        self.assertEqual(run['scheduler_status'],'PENDING')

    def test_verified_running_continuation_keeps_its_actual_new_updates(self):
        run={};record=dict(step=80089,parent_step=80000,new_optimizer_updates=89,gate_passed=True,job_id='1_0',scheduler=dict(state='RUNNING'))
        apply_live_execution(run,record,'now')
        self.assertEqual(run['last_step'],80089);self.assertEqual(run['new_optimizer_updates'],89)
        self.assertEqual(run['state'],'Training');self.assertFalse(run['registered_only'])

    def test_completed_evaluation_cannot_be_reset_to_queued_by_old_submission(self):
        run=dict(evaluation_complete=True)
        apply_live_execution(run,dict(step=24000,new_optimizer_updates=24000,job_id='2_0',gate_passed=True,scheduler=None),'now')
        self.assertEqual(run['state'],'Evaluated');self.assertEqual(run['scheduler_status'],'Complete')

    def test_real_snapshot_uses_string_scheduler_status_and_bound_file_body(self):
        run={}
        record=dict(step=24000,parent_step=0,new_optimizer_updates=24000,job_id='2_0',gate_passed=True,
            status='COMPLETED',scheduler=None,files={'status.json':{'data':{'complete':True,'step':24000}}})
        apply_live_execution(run,record,'now')
        self.assertTrue(run['training_complete'])
        self.assertEqual(run['state'],'Training complete / evaluation pending')

    def test_continuation_curves_exclude_parent_and_different_full_validation(self):
        rows=[dict(step=30000,n=256,stage='main',lpips=.47,test_used=False),
              dict(step=31000,n=5979,stage='main',lpips=.44,test_used=False),
              dict(step=31000,n=256,stage='main',lpips=.45,test_used=False)]
        parsed=new_research_validation('\n'.join(json.dumps(r) for r in rows),30000)
        self.assertEqual(parsed,[dict(step=31000,n=256,metrics=dict(lpips=.45))])
        rows.append(dict(step=31000,n=256,stage='main',lpips=.43,test_used=False))
        with self.assertRaises(ValueError):new_research_validation('\n'.join(json.dumps(r) for r in rows),30000)

    def test_gpu_totals_exclude_queued_and_completed_allocations(self):
        record=dict(jobs=[dict(status='RUNNING',gpus=4),dict(status='RUNNING',gpus=4),dict(status='RUNNING',gpus=1),dict(status='PENDING',gpus=4),dict(status='COMPLETED',gpus=4)],own_training_gpus=9,own_running_training_jobs=3,
            partition=dict(physical_gpus=176,schedulable_gpus=168,allocated_gpus=165,free_schedulable_gpus=3,largest_free_gpus_per_node=2))
        verify_gpu_snapshot(record)
        with self.assertRaises(ValueError):verify_gpu_snapshot(dict(record,own_training_gpus=13))
        bad=copy.deepcopy(record);bad['partition']['free_schedulable_gpus']=4
        with self.assertRaises(ValueError):verify_gpu_snapshot(bad)
