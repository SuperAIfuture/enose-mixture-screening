"""Check measurement integrity, group splits, and reference statistical results."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from metrics_utils import aggregate_probabilities_by_group, classification_metrics, stratified_group_bootstrap_macro_f1

ROOT = Path(__file__).resolve().parents[1]
PC = [f'probability_class_{i}' for i in range(7)]

def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))

def equal(observed, expected):
    if isinstance(expected, dict):
        assert observed.keys() == expected.keys()
        for k in expected: equal(observed[k],expected[k])
    elif isinstance(expected, list):
        assert len(observed)==len(expected)
        for a,b in zip(observed,expected): equal(a,b)
    elif isinstance(expected,(int,float)):
        assert np.isclose(observed,expected,atol=1e-12,rtol=0), (observed,expected)
    else:
        assert observed==expected

def grouped(frame):
    assert np.isfinite(frame[PC]).all().all()
    assert np.allclose(frame[PC].sum(axis=1),1,atol=1e-9,rtol=0)
    assert np.array_equal(frame[PC].to_numpy().argmax(axis=1),frame.predicted_class_id)
    return aggregate_probabilities_by_group(frame[PC].to_numpy(),frame.class_id.to_numpy(),frame.group_id.to_numpy())

def main():
    checked=0
    for line in (ROOT/'SHA256SUMS').read_text(encoding='utf8').splitlines():
        expected,name=line.split('  ',1)
        path=ROOT/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, name
        checked+=1
    raw=pd.read_csv(ROOT/'data/raw_manifest.csv')
    m=pd.read_csv(ROOT/'data/source_manifest.csv')
    a=np.load(ROOT/'data/source_responses.npz',allow_pickle=False)['responses']
    assert len(raw)==550 and raw.raw_id.nunique()==550
    assert raw.groupby('dg_parts').size().to_dict()=={i:50 for i in range(11)}
    assert int(raw.included_in_benchmark.sum())==350
    assert len(m)==350 and m.source_id.nunique()==350 and m.group_id.nunique()==348
    assert m.groupby('class_id').size().to_dict()=={i:50 for i in range(7)}
    assert a.shape==(350,150,8) and a.dtype==np.float32
    assert np.isfinite(a).all()
    for r in raw.itertuples():
        p=ROOT/r.raw_path
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r.sha256
        x=np.loadtxt(p,delimiter=',')
        assert x.shape==(r.rows,r.columns) and r.columns in (8,10)
    raw_lookup=raw.set_index('raw_id')
    for i,r in enumerate(m.itertuples()):
        p=ROOT/r.response_path
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r.response_sha256
        assert np.array_equal(np.loadtxt(p,delimiter=',',dtype=np.float32),a[i])
        rr=raw_lookup.loc[r.raw_id]
        assert rr.raw_path==r.raw_path and rr.sha256==r.raw_sha256
    assert m.groupby('response_sha256').group_id.nunique().max()==1
    assert m.groupby('group_id').response_sha256.nunique().max()==1
    assert sorted(m.groupby('group_id').size().value_counts().items())==[(1,346),(2,2)]
    for r in m.itertuples():
        members=m[m.group_id==r.group_id].source_id.tolist()
        assert r.group_size==len(members)
        assert set(str(r.group_members).split(';'))==set(members)
    config=read_json(ROOT/'config/benchmark.json')
    folds=pd.read_csv(ROOT/'data/outer_folds.csv')
    assert folds.source_id.tolist()==m.source_id.tolist()
    assert folds.groupby('group_id').outer_fold.nunique().max()==1
    features=a[:,30:130,:].reshape(350,800)
    y=m.class_id.to_numpy();groups=m.group_id.to_numpy()
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=config['outer_seed'])
    for fold,(train,test) in enumerate(outer.split(features,y,groups),1):
        assert set(groups[train]).isdisjoint(groups[test])
        assert np.array_equal(test,np.flatnonzero(folds.outer_fold.to_numpy()==fold))
        inner=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=config['inner_seed_base']+fold)
        for tr,va in inner.split(features[train],y[train],groups[train]):
            assert set(groups[train[tr]]).isdisjoint(groups[train[va]])
    pred=pd.read_csv(ROOT/'results/benchmark/nested_oof_predictions.csv')
    assert pred[['source_id','group_id','class_id','outer_fold']].equals(folds)
    g=grouped(pred)
    stored=pd.read_csv(ROOT/'results/benchmark/nested_oof_group_predictions.csv')
    assert g.group_ids.tolist()==stored.group_id.tolist()
    assert np.array_equal(g.source_counts,stored.source_count)
    assert np.allclose(g.probabilities,stored[PC],atol=1e-12,rtol=0)
    metrics=classification_metrics(g.labels,g.probabilities)
    equal(metrics,read_json(ROOT/'results/benchmark/pooled_metrics.json'))
    bootstrap=stratified_group_bootstrap_macro_f1(g.labels,g.probabilities,resamples=config['bootstrap_resamples'],seed=config['bootstrap_seed'])
    equal(bootstrap,read_json(ROOT/'results/benchmark/pooled_bootstrap.json'))
    sensitivity=pd.read_csv(ROOT/'results/sensitivity/summary.csv')
    assert len(sensitivity)==13
    for r in sensitivity.itertuples():
        p=ROOT/'results/sensitivity'/r.condition_id
        frame=pd.read_csv(p/'nested_oof_predictions.csv')
        assert frame.groupby('group_id').outer_fold.nunique().max()==1
        rebuilt=grouped(frame)
        mm=classification_metrics(rebuilt.labels,rebuilt.probabilities)
        equal(mm,read_json(p/'pooled_metrics.json'))
        assert abs(mm['macro_f1']-r.macro_f1)<1e-12
    for stem in ['permutation','fixed_parameter_refit_permutation']:
        frame=pd.read_csv(ROOT/f'results/permutation/{stem}_metrics.csv')
        summary=read_json(ROOT/f'results/permutation/{stem}_summary.json')
        assert len(frame)==1000
        assert np.array_equal(frame.permutation_index_zero_based,np.arange(1000))
        assert np.array_equal(frame.seed,2026080711+np.arange(1000))
        values=frame.macro_f1.to_numpy()
        assert abs(values.mean()-summary['null_mean_macro_f1'])<1e-12
        assert abs(values.max()-summary['null_maximum_macro_f1'])<1e-12
        assert abs(np.quantile(values,.95)-summary['null_95th_percentile_macro_f1'])<1e-12
        assert (1+np.sum(values>=summary['observed_macro_f1']))/1001==summary['empirical_p_value']
    print(f"RELEASE_PASS files={checked} raw=550 normalized=350 groups=348 sensitivity=13 permutations=1000+1000")
    print(f"METRICS_PASS macro_f1={metrics['macro_f1']:.12f} ci95={bootstrap['ci95_lower']:.12f},{bootstrap['ci95_upper']:.12f} ece={metrics['expected_calibration_error']:.12f} brier={metrics['multiclass_brier_score']:.12f}")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
