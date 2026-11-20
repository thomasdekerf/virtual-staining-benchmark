"""Exercise the real readiness guard at its GitHub CLI boundary."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class PagesArtifactTests(unittest.TestCase):
    def guard(self, responses, artifact_id='123'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, source in {
                'sleep': '#!/bin/bash\nexit 0\n',
                'gh': '#!/bin/bash\nn=$(cat "$MOCK_COUNT")\nn=$((n+1))\nprintf "%s" "$n" > "$MOCK_COUNT"\nsed -n "${n}p" "$MOCK_RESPONSES"\n',
            }.items():
                path=root/name;path.write_text(source);path.chmod(0o755)
            (root/'count').write_text('0')
            (root/'responses').write_text('\n'.join(responses)+'\n')
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ['PATH'],
                PAGES_ARTIFACT_ID=artifact_id,GITHUB_REPOSITORY='owner/site',GITHUB_RUN_ID='99',
                MOCK_COUNT=str(root/'count'),MOCK_RESPONSES=str(root/'responses'))
            result=subprocess.run(['bash',str(Path(__file__).with_name('wait_for_pages_artifact.sh'))],
                env=env,capture_output=True,text=True)
            return result,int((root/'count').read_text())

    def test_delayed_exact_upload_becomes_ready(self):
        result,calls=self.guard(['','','123'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(calls,3)

    def test_previous_attempt_artifact_does_not_satisfy_readiness(self):
        result,calls=self.guard(['122']*20)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(calls,20)
        self.assertIn('not discoverable',result.stderr)

    def test_missing_or_invalid_upload_identity_fails_before_query(self):
        for identity in ['', 'not-an-id']:
            result,calls=self.guard(['123'],identity)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(calls,0)
