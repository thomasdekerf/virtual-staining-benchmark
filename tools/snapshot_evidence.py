"""Choose the latest explicitly captured, dated production audit."""
import json


def current_cluster_snapshot(workspace):
    candidates = []
    for path in (workspace / 'artifacts').glob('coverage_balance_*/live_audit/cluster_snapshot.json'):
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if value.get('captured_utc'):
            candidates.append((value['captured_utc'], path, value))
    if not candidates:
        return None, {}
    _, path, value = max(candidates, key=lambda item: item[0])
    return path, value


def current_campaign_snapshot(workspace):
    candidates=[]
    for path in (workspace/'artifacts').glob('coverage_balance_*/campaign_snapshot.json'):
        try:value=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if value.get('snapshot_utc'):candidates.append((value['snapshot_utc'],path,value))
    if not candidates:return None,{}
    _,path,value=max(candidates,key=lambda item:item[0])
    return path,value


def current_followup_snapshot(workspace):
    candidates=[]
    for path in (workspace/'artifacts').glob('archives_resume_*/live_audit/followups_snapshot.json'):
        try:value=json.loads(path.read_text())
        except (OSError,ValueError):continue
        if value.get('captured_utc'):candidates.append((value['captured_utc'],path,value))
    if not candidates:return None,{}
    _,path,value=max(candidates,key=lambda item:item[0])
    return path,value
