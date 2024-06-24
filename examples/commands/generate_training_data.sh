#BSUB -nnodes 1
#BSUB -J snapshots
#BSUB -W 12:00
#BSUB -q pbatch

export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

# jsrun -r 1 python generate_residual_data.py --nx 480 --ny 24 --na 20 --nlam 20 --viscosity 1e-1 
# jsrun -r 1 python generate_snapshot_data.py --nx 480 --ny 24 --na 80 --nlam 80 --viscosity 1e-1 

# Commands
# --------
python generate_data.py --nx 480 --ny 24 --na 20 --nlam 20 --viscosity 1e-1 --save_dir "/Users/zanardi1/Workspace/Codes/DD-NM-ROM/run/data/residual/"
# python generate_data.py --nx 480 --ny 24 --na 80 --nlam 80 --viscosity 1e-1 --save_dir "/Users/zanardi1/Workspace/Codes/DD-NM-ROM/run/data/snapshot/"