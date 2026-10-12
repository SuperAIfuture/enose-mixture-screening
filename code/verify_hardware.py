"""Recalculate the provided integer inference vectors and their memory encoding."""
from pathlib import Path
import numpy as np
import pandas as pd
from c2_reference import load_signed_int8_mem,run_integer_reference,encode_extension_frames

ROOT=Path(__file__).resolve().parents[1]

def memory(path,bits,signed=False):
    v=np.asarray([int(l,16) for l in path.read_text(encoding='utf8').splitlines() if l],dtype=object)
    if signed:v=np.where(v>=(1<<(bits-1)),v-(1<<bits),v)
    return v.astype(np.int64)

def main():
    total=0
    for fold in sorted((ROOT/'hardware/c1_fixed_point/folds').iterdir()):
        source=pd.read_csv(fold/'source_manifest.csv')
        x=memory(fold/'inputs.mem',12).reshape(-1,800)
        w=memory(fold/'weights.mem',24,True).reshape(7,800)
        b=memory(fold/'biases.mem',64,True)
        p=(x@w.T+b).argmax(axis=1)
        expected=memory(fold/'expected_fixed_classes.mem',3)
        assert np.array_equal(p,expected)
        assert np.array_equal(p,memory(fold/'expected_float_classes.mem',3))
        assert np.array_equal(p,source.fixed_point_class_id)
        total+=len(p)
    assert total==350
    vr=ROOT/'hardware/c2/vectors'
    a=np.load(vr/'bridge_full_vectors.npz',allow_pickle=False)
    frames=a['frames'];expected=a['counts'];decisions=a['predictions']
    hidden=load_signed_int8_mem(vr/'hidden_weights.mem',(64,800))
    output=load_signed_int8_mem(vr/'output_weights.mem',(7,64))
    counts,p,_,_=run_integer_reference(frames,hidden,output)
    assert np.array_equal(counts,expected) and np.array_equal(p,decisions)
    lines=(vr/'bridge_frames.mem').read_text(encoding='utf8').splitlines()
    memframes=np.stack([np.unpackbits(np.frombuffer(int(line,16).to_bytes(100,'little'),dtype=np.uint8),bitorder='little') for line in lines]).reshape(350,30,800)
    assert np.array_equal(memframes,frames)
    assert np.array_equal(memory(vr/'bridge_expected_counts.mem',16).reshape(350,7),expected)
    assert np.array_equal(memory(vr/'bridge_expected_classes.mem',3),decisions)
    ledger=pd.read_csv(vr/'source_ledger.csv')
    assert np.array_equal(ledger.source_id.to_numpy(),a['source_ids'])
    frozen=np.load(ROOT/'hardware/c2/frozen_vectors.npz',allow_pickle=False)
    frozen_lookup={str(s):i for i,s in enumerate(frozen['source_ids'])}
    responses=np.load(ROOT/'data/source_responses.npz',allow_pickle=False)['responses']
    features=responses[:,30:130,:].reshape(350,800)
    preserved=extended=0
    for i,r in enumerate(ledger.itertuples()):
        if r.c2_vector_origin=='frozen_split_01':
            j=frozen_lookup[r.source_id]
            assert np.array_equal(frames[i],frozen['spike_frames'][j])
            preserved+=1
        else:
            assert np.array_equal(frames[i],encode_extension_frames(features[i],r.source_id,.5))
            extended+=1
    assert (preserved,extended)==(70,280)
    print('HARDWARE_REFERENCE_PASS c1_fixed=350/350 c2_counts=350/350 c2_decisions=350/350 frozen=70 extension=280')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
