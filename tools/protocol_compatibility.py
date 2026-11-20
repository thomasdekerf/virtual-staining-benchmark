"""Attach only explicitly source-reviewed metric protocol identities.

Source wording stays intact. Geometry, manifest, marker, split and tile count
remain independent comparison keys in both chart engines.
"""
import hashlib
import json

NATIVE_OLD='FP32 native 256 center crops; per-tile PSNR, project 11x11 box SSIM, AlexNet LPIPS, MAE, DeltaE76; dataset macro for pooled summary'
NATIVE_NEW='FP32 native 256 center crops; per-tile PSNR, project 11x11 box SSIM, AlexNet LPIPS, MAE, DeltaE76; tile mean'
HER2_OLD='FP32, TF32 disabled; original HER2Match case splits; single center256 crop per tile'
HER2_NEW='FP32, TF32 disabled; physical0.46um/px256 center crop; per-tile PSNR, box SSIM, LPIPS, MAE, DeltaE76'


def attach_reviewed_protocols(runs,review_path):
    review=json.loads(review_path.read_text());digest=hashlib.sha256(review_path.read_bytes()).hexdigest()
    for run in runs.values():
        for e in run['evaluations']:
            if not e.get('rankable') or e['split'] not in {'val','test'} or not e.get('manifest_sha256'):continue
            raw=e.get('metric_protocol');path=e.get('evidence',{}).get('file','')
            native=e['dataset'] in {'BCI','MIST','IHC4BC'} and ((raw==NATIVE_OLD and path=='artifacts/breast_ihc/campaign/progress.json') or (raw==NATIVE_NEW and run['id'].startswith('balance__') and path.endswith('/final_evaluation.json')))
            her2=e['dataset']=='her2match' and ((raw==HER2_OLD and run['id'].startswith('her2match__')) or (raw==HER2_NEW and run['id'].startswith('balance__'))) and path.endswith('/final_evaluation.json')
            if native or her2:
                key='native_breast' if native else 'her2match_physical'
                e['comparison_protocol']=review['protocols'][key]['comparison_protocol']
                e['comparison_protocol_review_sha256']=digest
                e['crop_size']=256
