"""A scored parent must not turn an unfinished BBDM extension into a score."""
import copy
import unittest
from runnable_results import verify_bbdm_extension

class RunnableResultTests(unittest.TestCase):
    def test_extension_requires_real_online_and_ema_updates_and_selected_reference(self):
        task=dict(source_sha256='a'*64,parent_step=100000)
        provenance=dict(source_sha256='a'*64,parent_step=100000,parent_checkpoint_sha256='b'*64,
            test_used_for_continuation_decision=False,learning_rates=[.0001],
            parent_online_learned_sha256='c'*64,parent_ema_learned_sha256='d'*64,
            metadata_only_rebinding_verified_after_reload=True,inherited_files={'best.pt':'e'*64})
        gate=dict(passed=True,world_size=4,source_sha256='a'*64,parent_step=100000,
            observed_step=100001,parent_checkpoint_sha256='b'*64,actual_online_and_ema_updates=True,
            selected_reference_preserved=True,test_used=False,learning_rates=[.0001],
            optimizer_min_max_steps=[100001.,100001.],scheduler_last_epoch=100001,
            online_learned_sha256='f'*64,ema_learned_sha256='0'*64)
        parent=dict(checkpoint_sha256='e'*64)
        verify_bbdm_extension(gate,provenance,task,parent)
        for mutation in [dict(ema_learned_sha256='d'*64),dict(selected_reference_preserved=False),
                         dict(learning_rates=[0.]),dict(test_used=True),dict(observed_step=100000)]:
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                verify_bbdm_extension(gate|mutation,provenance,task,parent)
        changed=copy.deepcopy(provenance);changed['inherited_files']['best.pt']='1'*64
        with self.assertRaises(ValueError):verify_bbdm_extension(gate,changed,task,parent)

if __name__=='__main__':unittest.main()
