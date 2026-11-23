"""Architecture citations and registration preserve the scored-data boundary."""
import copy
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from architecture_catalog import ARCHITECTURE_MODELS,augment_catalog,create_run
from her2_campaigns import TARGET_MODELS,campaign_summary,review_new_campaign_protocol
from protocol_compatibility import HER2_NEW


class ArchitectureCatalogTests(unittest.TestCase):
    def test_new_catalog_entries_do_not_create_runs_or_overwrite_existing_scores(self):
        existing=dict(id='uformer',paper=dict(url='original-reviewed-paper'),runs=0,
                      datasets=[],source=dict(sheet='Original workbook'))
        data=dict(models=[existing,dict(id='unet',name='Original U-Net')],
                  runs=[dict(id='original',evaluations=[dict(metrics=dict(psnr=23.123456789))])])
        before=copy.deepcopy(data['runs'])
        augment_catalog(data)
        self.assertEqual(data['runs'],before)
        self.assertEqual({m['id'] for m in data['models']},set(ARCHITECTURE_MODELS)|{'unet'})
        self.assertEqual(existing['paper']['url'],'original-reviewed-paper')
        self.assertEqual(existing['source'],dict(sheet='Original workbook'))
        self.assertEqual(existing['name'],'Uformer-B')

    def test_all_new_methods_are_in_generic_landscape_without_invented_scores(self):
        index=dict(models=[dict(id=m,name=m) for m in TARGET_MODELS],runs=[],
                   her2match_campaign=dict(snapshot='captured'))
        result=campaign_summary(index)
        self.assertEqual(result['registered_model_families'],0)
        self.assertEqual(len(result['rows']),len(TARGET_MODELS))
        self.assertTrue(set(ARCHITECTURE_MODELS)<=set(TARGET_MODELS))
        self.assertTrue(all(not r['test_score_available'] for r in result['rows']))

    def test_bci_architecture_rows_are_ihc_not_unstained_evaluations(self):
        for dataset in ['BCI','HER2Match','her2match','IHC4BC']:
            run=create_run('new','uformer',dataset)
            self.assertEqual((run['task'],run['marker']),('ihc','HER2'))
            self.assertEqual(run['evaluations'],[])
        run=create_run('new','uformer','finland')
        self.assertEqual((run['task'],run['marker']),('unstained','H&E'))

    def test_new_comparison_review_binds_common_code_and_keeps_old_payload_exact(self):
        site=Path(__file__).resolve().parents[1]
        reference=json.loads((site/'dist/data/protocol-compatibility.json').read_text())
        evaluation=dict(id='new-test',rankable=True,split='test',dataset='her2match',metric_protocol=HER2_NEW)
        run=dict(id='new',evaluations=[evaluation]);task=dict(source_sha256='a'*64)
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);(output/'dist/data').mkdir(parents=True)
            plan=output/'frozen/campaign.json';code=plan.parent/'code/src/virtual_staining';code.mkdir(parents=True)
            fixtures={'metrics.py':'# Independent common metric fixture\n',
                      'breast_data.py':'# Independent common data fixture\n',
                      'losses.py':'def ssim_loss(a,b):\n    return a-b\n'}
            for filename,text in fixtures.items():(code/filename).write_text(text)
            reference['source_hashes']['shared_validation_metrics']=hashlib.sha256(fixtures['metrics.py'].encode()).hexdigest()
            reference['source_hashes']['her2match_dataset']=hashlib.sha256(fixtures['breast_data.py'].encode()).hexdigest()
            reference['source_hashes']['unchanged_ssim_method_ast']=hashlib.sha256(ast.dump(ast.parse(fixtures['losses.py']).body[0],include_attributes=False).encode()).hexdigest()
            review=review_new_campaign_protocol(run,task,plan,output,reference,{})
            self.assertTrue(review.is_file())
            self.assertEqual(evaluation['comparison_protocol_review_sha256'],hashlib.sha256(review.read_bytes()).hexdigest())
            old=dict(id='old-test',rankable=True,split='test',dataset='her2match',metric_protocol=HER2_NEW)
            unchanged=copy.deepcopy(old);run=dict(id='old',evaluations=[old])
            self.assertIsNone(review_new_campaign_protocol(run,task,plan,output,reference,{('old','old-test'):unchanged}))
            self.assertEqual(old,unchanged)
            (code/'metrics.py').write_text('# Altered metric implementation\n')
            with self.assertRaises(ValueError):review_new_campaign_protocol(run,task,plan,output,reference,{})
