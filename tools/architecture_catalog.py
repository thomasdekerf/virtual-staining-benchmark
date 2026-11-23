"""Reviewed published architecture additions; registration never supplies scores."""
from __future__ import annotations

import copy


ARCHITECTURE_MODELS = {
    'pix2pixhd': dict(name='Pix2PixHD', priority=2,
        task='Paired RGB image-to-image translation',
        backbone='Global ResNet generator + multiscale PatchGAN',
        family='Conditional GAN + discriminator/VGG feature matching',
        pairing='Paired',
        implementation='Author architecture integrated; native four-GPU gate required',
        url='https://github.com/NVIDIA/pix2pixHD',
        notes='Original global generator, multiscale discriminators and VGG/discriminator feature objectives. RGB staining uses the author image-input mode. Full-capacity resume checks and complete selected-checkpoint evaluation are required before scores.'),
    'dclgan': dict(name='DCLGAN', priority=2,
        task='Unpaired RGB image-to-image translation',
        backbone='Two ResNet generators, two discriminators and two projection branches',
        family='Dual contrastive unpaired GAN', pairing='Unpaired training; paired held-out evaluation',
        implementation='Author architecture integrated; native four-GPU gate required',
        url='https://github.com/JunlinHan/DCLGAN',
        notes='Published dual contrastive learning method with both directions, projection heads and native GAN/PatchNCE/identity objectives. Training does not use the target paired with each source field. Validation uses the common held-out source/target fields.'),
    'i2sb': dict(name='I²SB', priority=2,
        task='Paired RGB image-to-image translation',
        backbone='Conditional ADM U-Net + analytic diffusion bridge',
        family='Image-to-image Schrödinger bridge diffusion', pairing='Paired',
        implementation='Author architecture integrated; native four-GPU gate required',
        url='https://github.com/NVlabs/I2SB',
        notes='Original bridge objective, endpoint conditioning, online/EMA weights and bridge sampler. The precise sampling count and initialization are disclosed in each frozen staining recipe; these are local staining results rather than ImageNet paper scores.'),
    'uformer': dict(name='Uformer-B', priority=2,
        task='Paired RGB restoration-backbone staining adaptation',
        backbone='Native Uformer-B LeWin/LeFF Transformer',
        family='Supervised regression; no GAN', pairing='Paired',
        implementation='Native author Uformer-B integrated; native four-GPU gate required',
        url='https://github.com/ZhendongWang6/Uformer',
        notes='Full 50,880,946-parameter author Uformer-B, including the original RGB residual output and decoder modulators. Paired staining L1 is an explicit benchmark task adaptation trained from scratch, with no added sigmoid or output-head replacement.'),
}


def create_run(rid,model,dataset,names=None,**values):
    """Create an unscored record with an explicit stained/unstained task identity."""
    names=names or {}
    ihc=dataset in {'her2match','HER2Match','BCI','MIST','IHC4BC'}
    return dict(id=rid,model=model,name=names.get(model,model),variant='Production · seed 42',
        datasets=[dataset],task='ihc' if ihc else 'unstained',marker='HER2' if ihc else 'H&E',
        state='Registered',seed=42,evaluations=[],sources=[],curves={},gallery=[],
        distributions={},curve_counts={},curve_axis='step')|values


def augment_catalog(data):
    """Add citations and implementation descriptions without manufacturing runs."""
    models = {model['id']: model for model in data['models']}
    for mid, reviewed in ARCHITECTURE_MODELS.items():
        entry = models.get(mid)
        if entry is None:
            entry = dict(id=mid, source=dict(type='Reviewed architecture expansion',
                reviewed_date='2026-11-23'), runs=0, evaluated_runs=0, datasets=[])
            data['models'].append(entry)
        entry.update(copy.deepcopy(reviewed))
    return data
