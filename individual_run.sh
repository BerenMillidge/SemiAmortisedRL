source ~/.bashrc
set -e
echo "Setting up log files"
SCRATCH_DISK=/disk/scratch
USER=s1686853
log_name=$1
log_path=${SCRATCH_DISK}/${USER}/${log_name}
env_name=$2
use_reward=$3
use_exploration=$4
action_noise=$5
#mkdir -p ${log_path}

echo "Initializing Conda Environment"
CONDA_NAME=env
conda activate ${CONDA_NAME}

echo "Running experiment command"
#pip install git+https://github.com/Bmillidgework/exploration-baselines
python main.py --env_name ${env_name} --logdir ${log_path} --action_noise ${action_noise} --plan_horizon "30" --action_repeat "1" --ensemble_size "15" --use_exploration ${use_exploration} --use_reward ${use_reward} --expl_scale "0.1" --n_episodes "100"

echo "Experiment Finished. Moving data back to DFS"
save_path=/home/${USER}/fe_mbrl/logs/
rsync --archive --update --compress --progress ${log_path}/ ${save_path}

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
