BASE_SCRIPT=$SCRATCH/UniVTAC/docker/cluster/slurm_eval_policy.sh
TASK_NAME=${1:-"pick_and_place_fruits"}
TASK_MODE=${2:-"demo"}
MODEL_CONFIG=${3-"policy/ACT/deploy.yml"}

sbatch --export=ALL,TASK_NAME=$TASK_NAME,TASK_MODE=$TASK_MODE,MODEL_CONFIG=$MODEL_CONFIG\
           $BASE_SCRIPT