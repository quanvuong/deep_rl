import os
from subprocess import call

grid_sizes = [(6, 6), (10, 10), (16, 16), (25, 25), (50, 50)]

HUNTER_FOLDER = 'hunters_results'
BASE_SBATCH_SCRIPT = """#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error_output}

#SBATCH --time=00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40000mb
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qhv200@nyu.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=debug
#SBATCH --share
cd
cd /gpfsnyu/home/qhv200/deep_rl
source activate pytorch_gpu
python policy_gradient_batch_baseline.py --game hunters --hunter 1 --rabbit 1 --gamma 0.8 --grid_size {grid_size} --save_policy {save_policy}"""
SCRIPT_SUFFIX = '.sbatch_script'

if not os.path.exists(HUNTER_FOLDER):
    os.mkdir(HUNTER_FOLDER)

for grid_size in grid_sizes:
    grid_x = grid_size[0]
    grid_y = grid_size[1]

    grid_result_folder = '{}/{}x{}/'.format(
        HUNTER_FOLDER,
        grid_x,
        grid_y
    )

    if not os.path.exists(grid_result_folder):
        os.mkdir(grid_result_folder)

    job_name = 'hunters_1h1r_{}x{}_08'.format(grid_x, grid_y)
    sbatch_script = BASE_SBATCH_SCRIPT.format(
        job_name=job_name,
        output='{}{}.o'.format(grid_result_folder, job_name),
        error_output='{}{}.e'.format(grid_result_folder, job_name),
        # Assume square grid
        grid_size=grid_x,
        save_policy=job_name + '.policy'
    )
    script_loc = grid_result_folder + job_name + SCRIPT_SUFFIX

    with open(script_loc, 'w') as f:
        f.write(sbatch_script)

    call(["sbatch", script_loc])


