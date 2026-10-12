"""Run Icarus on C1 linear and C2 spiking reference memories in temporary folders."""
from pathlib import Path
import argparse,shutil,subprocess,tempfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def execute(cmd,cwd,marker=None):
    r=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True)
    if r.returncode or (marker and marker not in r.stdout):
        raise RuntimeError(f'{cmd[0]} failed (exit {r.returncode}):\n{r.stdout}\n{r.stderr}')
    if marker: print(r.stdout.strip())

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iverilog',default='iverilog')
    parser.add_argument('--vvp',default='vvp')
    parser.add_argument('--mode',choices=['all','c1','c2'],default='all')
    args=parser.parse_args()
    iv=shutil.which(args.iverilog);vvp=shutil.which(args.vvp)
    if not iv or not vvp:parser.error('provide paths to Icarus iverilog and vvp')
    with tempfile.TemporaryDirectory(prefix='enose-rtl-') as tmp:
        work=Path(tmp)
        if args.mode in ['all','c1']:
            core=ROOT/'hardware/c1_fixed_point/rtl'
            total=0
            for index,fold in enumerate(sorted((ROOT/'hardware/c1_fixed_point/folds').iterdir()),1):
                for f in fold.glob('*.mem'):shutil.copyfile(f,work/f.name)
                count=len(pd.read_csv(fold/'source_manifest.csv'))
                program=work/'c1.vvp'
                execute([iv,'-g2012',f'-DSAMPLE_COUNT={count}',f'-DFOLD_INDEX={index}','-s','tb_c1_fixed_point_fold','-o',str(program),str(core/'c1_fixed_point_linear_800x7.sv'),str(core/'tb_c1_fixed_point_fold.sv')],work)
                execute([vvp,str(program)],work,'C1_FIXED_POINT_RTL_PASS')
                total+=count
            print(f'C1_RTL_PASS folds=5 samples={total} fixed={total}/{total} float={total}/{total}')
        if args.mode in ['all','c2']:
            core=ROOT/'hardware/c2/rtl'
            for f in (ROOT/'hardware/c2/vectors').glob('*.mem'):shutil.copyfile(f,work/f.name)
            program=work/'c2.vvp'
            execute([iv,'-g2012','-s','tb_c1_c2_bridge_350','-o',str(program),str(core/'tloc_qsnn_fixed_core.sv'),str(core/'tb_c1_c2_bridge_350.sv')],work)
            execute([vvp,str(program)],work,'C1_C2_BRIDGE_RTL_PASS samples=350 decisions=350/350 counts=350/350')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
