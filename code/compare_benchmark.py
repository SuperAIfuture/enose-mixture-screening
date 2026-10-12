"""Compare a newly fitted nested benchmark with the supplied reference."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from metrics_utils import aggregate_probabilities_by_group,classification_metrics

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    observed=pd.read_csv(args.results/'nested_oof_predictions.csv')
    reference=pd.read_csv(ROOT/'results/benchmark/nested_oof_predictions.csv')
    identity=['source_id','group_id','class_id','outer_fold']
    assert observed[identity].equals(reference[identity])
    probability=[f'probability_class_{i}' for i in range(7)]
    count=int((observed.predicted_class_id==reference.predicted_class_id).sum())
    delta=float(np.abs(observed[probability]-reference[probability]).to_numpy().max())
    g=aggregate_probabilities_by_group(observed[probability].to_numpy(),observed.class_id.to_numpy(),observed.group_id.to_numpy())
    print(json.dumps({'source_decision_agreement':count,'source_count':len(observed),'maximum_probability_difference':delta,'group_metrics':classification_metrics(g.labels,g.probabilities)},indent=2))
    assert count==350, 'fresh-fit decisions differ from the reference; inspect numerical environment'
    return 0

if __name__=='__main__':
    raise SystemExit(main())
