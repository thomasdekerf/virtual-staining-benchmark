"""Archive connectivity cannot manufacture an available input or prediction."""
import copy
import unittest
from archive_readiness import apply_archive_observation


class ArchiveReadinessTests(unittest.TestCase):
    def test_reachable_archive_changes_only_unavailable_explanations(self):
        observation=dict(schema='archive-readiness-observation-v1',snapshot='2026-11-24T19:45:24Z',
            read_only=True,archive_server_reachable=True,archive_volume_detected=True,
            archive_volume_mounted=False,original_assets_verified=False,source=dict(sha256='a'*64))
        available=dict(available=True,url='images/bound.webp',sha256='b'*64,checkpoint_sha256='c'*64)
        gallery=dict(datasets=[dict(models=[dict(status='unavailable',reason='Archived selected checkpoint unavailable: archive server offline')],fields=[dict(input=dict(available=False,reason='Original 17-channel ORION input bundle; archive server is offline'),
            predictions=[available,dict(available=False,reason='Archived selected checkpoint unavailable: archive server offline'),
                         dict(available=False,reason='Genuine HER2 labels unavailable')])])])
        before=copy.deepcopy(gallery);updated=apply_archive_observation(gallery,observation)
        field=updated['datasets'][0]['fields'][0]
        self.assertFalse(field['input']['available']);self.assertIn('volume is unmounted',field['input']['reason'])
        self.assertEqual(field['predictions'][0],available)
        self.assertFalse(field['predictions'][1]['available']);self.assertIn('server is reachable',field['predictions'][1]['reason'])
        self.assertIn('server is reachable',updated['datasets'][0]['models'][0]['reason'])
        self.assertEqual(field['predictions'][2],before['datasets'][0]['fields'][0]['predictions'][2])
        self.assertEqual(gallery,before)

    def test_unverified_observation_cannot_relabel_archive_state(self):
        with self.assertRaises(ValueError):apply_archive_observation({},dict(archive_server_reachable=True))


if __name__=='__main__':unittest.main()
