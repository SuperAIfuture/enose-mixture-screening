"""Group-label permutation under fixed OOF decisions or fixed-parameter refits."""
from __future__ import annotations
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from metrics_utils import aggregate_probabilities_by_group,classification_metrics

ROOT=Path(__file__).resolve().parents[1]
PC=[f'probability_class_{i}' for i in range(7)]
STATE={}

def initialize(features,groups,group_ids,labels,folds,frozen,seed):
    global STATE
    STATE=dict(features=features,groups=groups,group_ids=group_ids,labels=labels,folds=folds,frozen=frozen,seed=seed)

def iteration(item):
    index,mode=item
    s=STATE
    perm=np.random.default_rng(2026080711+index).permutation(len(s['labels']))
    group_labels=s['labels'][perm]
    if mode=='association':
        metrics=classification_metrics(group_labels,s['frozen'])
    else:
        mapping=dict(zip(s['group_ids'],group_labels))
        labels=np.asarray([mapping[g] for g in s['groups']],dtype=np.int64)
        probability=np.full((len(labels),7),np.nan)
        for fold in range(1,6):
            train=s['folds']!=fold;test=~train
            model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=100,max_iter=5000,random_state=s['seed']+fold,multi_class='auto'))])
            model.fit(s['features'][train],labels[train])
            p=model.predict_proba(s['features'][test])
            aligned=np.zeros((int(test.sum()),7))
            aligned[:,model.named_steps['model'].classes_.astype(int)]=p
            probability[test]=np.clip(aligned,1e-12,1)
        g=aggregate_probabilities_by_group(probability,labels,s['groups'])
        metrics=classification_metrics(g.labels,g.probabilities)
    return dict(permutation_index_zero_based=index,seed=2026080711+index,macro_f1=metrics['macro_f1'],accuracy=metrics['accuracy'],balanced_accuracy=metrics['balanced_accuracy'])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['association','refit'],required=True)
    parser.add_argument('--permutations',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/permutation')
    args=parser.parse_args()
    if args.permutations<1 or args.workers<1:parser.error('counts must be positive')
    manifest=pd.read_csv(ROOT/'data/source_manifest.csv')
    observed=pd.read_csv(ROOT/'results/permutation/observed_fixed_c100_raw_predictions.csv')
    assert manifest.source_id.tolist()==observed.source_id.tolist()
    responses=np.load(ROOT/'data/source_responses.npz',allow_pickle=False)['responses']
    features=responses[:,30:130,:].reshape(350,800).astype(np.float32)
    groups=manifest.group_id.to_numpy()
    g=aggregate_probabilities_by_group(observed[PC].to_numpy(),manifest.class_id.to_numpy(),groups)
    seed=json.loads((ROOT/'config/benchmark.json').read_text(encoding='utf8'))['outer_seed']
    initargs=(features,groups,g.group_ids,g.labels,observed.outer_fold.to_numpy(),g.probabilities,seed)
    items=[(i,args.mode) for i in range(args.permutations)]
    if args.mode=='association' or args.workers==1:
        initialize(*initargs)
        rows=[iteration(item) for item in items]
    else:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize,initargs=initargs) as pool:
            rows=list(pool.map(iteration,items))
    frame=pd.DataFrame(rows)
    values=frame.macro_f1.to_numpy()
    observed_f1=classification_metrics(g.labels,g.probabilities)['macro_f1']
    summary=dict(mode=args.mode,permutations=args.permutations,model_refit_per_permutation=args.mode=='refit',inner_selection_repeated_per_permutation=False,observed_macro_f1=observed_f1,null_mean_macro_f1=float(values.mean()),null_95th_percentile_macro_f1=float(np.quantile(values,.95)),null_maximum_macro_f1=float(values.max()),empirical_p_value=float((1+np.sum(values>=observed_f1))/(1+args.permutations)))
    output=args.output/args.mode
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'metrics.csv',index=False,encoding='utf8',lineterminator='\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
