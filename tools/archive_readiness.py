"""Update unavailable-asset explanations without changing any image binding."""
from __future__ import annotations
import copy
import re


def apply_archive_observation(gallery, observation):
    if not (observation.get('schema')=='archive-readiness-observation-v1'
            and observation.get('read_only') is True
            and observation.get('archive_server_reachable') is True
            and observation.get('archive_volume_detected') is True
            and observation.get('archive_volume_mounted') is False
            and observation.get('original_assets_verified') is False
            and observation.get('snapshot')
            and re.fullmatch('[0-9a-f]{64}',observation.get('source',{}).get('sha256',''))):
        raise ValueError('Source-bound reachable-server/unmounted-volume observation required')
    result=copy.deepcopy(gallery)
    def visit(value):
        if isinstance(value,dict):
            reason=value.get('reason') or ''
            if (value.get('available') is False or value.get('status')=='unavailable') and ('archive server' in reason or 'archive access' in reason):
                prefix=('Original 17-channel ORION input bundle remains archived.'
                        if reason.startswith('Original 17-channel ORION') else
                        'Archived selected checkpoint remains unavailable.')
                value['reason']=prefix+' The archive server is reachable, but its 26 TB archive volume is unmounted (verified '+observation['snapshot'][:10]+').'
            for nested in value.values():visit(nested)
        elif isinstance(value,list):
            for nested in value:visit(nested)
    visit(result)
    result['archive_availability']=copy.deepcopy(observation)
    return result
