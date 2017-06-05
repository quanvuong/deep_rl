import os
from subprocess import call

grid_size = (6, 6)
hidden_layers = [[128], [128, 128]]

HUNTER_FOLDER = 'hunters_results/diff_hidden_layers'
BASE_SBATCH_SCRIPT = """#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error_output}

#SBATCH --time=00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=16000MB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qhv200@nyu.edu
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --exclusive
cd
cd /gpfsnyu/home/qhv200/deep_rl
source activate pytorch_gpu
python policy_gradient.py --game hunters --rabbit 6 --hunter 6 --grid_size 6 --gamma 0.8 --hidden_layers {hidden_layers} --save_policy {save_policy}"""
SCRIPT_SUFFIX = '.sbatch_script'

if not os.path.exists(HUNTER_FOLDER):
    os.makedirs(HUNTER_FOLDER)

for hidden_layer in hidden_layers:
    grid_x = grid_size[0]
    grid_y = grid_size[1]

    grid_result_folder = '{}/{}x{}/'.format(
        HUNTER_FOLDER,
        grid_x,
        grid_y
    )

    if not os.path.exists(grid_result_folder):
        os.mkdir(grid_result_folder)

    hl_as_str = ' '.join(str(size) for size in hidden_layer)
    hl_as_str_in_name = hl_as_str.replace(' ', '_')

    job_name = 'hunters_1h1r_{}x{}_08_{}'.format(grid_x, grid_y, hl_as_str_in_name)
    sbatch_script = BASE_SBATCH_SCRIPT.format(
        job_name=job_name,
        output='{}{}.o'.format(grid_result_folder, job_name),
        error_output='{}{}.e'.format(grid_result_folder, job_name),
        # Assume square grid
        grid_size=grid_x,
        hidden_layers=hl_as_str,
        save_policy=job_name + '.policy'
    )
    script_loc = grid_result_folder + job_name + SCRIPT_SUFFIX

    with open(script_loc, 'w') as f:
        f.write(sbatch_script)

    call(["sbatch", script_loc])


