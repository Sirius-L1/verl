#!/bin/bash
#SBATCH --job-name=ui-verl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1024G
#SBATCH --partition=AISS2025031801
#SBATCH --account=polyullm
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=/lustre/projects/polyullm/yuhang/r2/checkpoints/infigui-grounder-v1/enum_v2_rloo_bs128_n16_7b/log-%j.out
#SBATCH --error=/lustre/projects/polyullm/yuhang/r2/checkpoints/infigui-grounder-v1/enum_v2_rloo_bs128_n16_7b/log-%j.err

# set -x

# replace these information with your own
verl_workdir=/lustre/projects/polyullm/yuhang/r2/verl
container_image=/lustre/projects/polyullm/container/verl+cu126+0503.sqsh
container_name=verl+cu126+0503
container_mounts=/lustre/projects/polyullm:/lustre/projects/polyullm,/home/projects/polyullm:/home/projects/polyullm
# replace these information with your own

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start Ray head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=offline
export WANDB_DIR="/lustre/projects/polyullm/yuhang/r2/wandb"
export HOME=$verl_workdir

printenv

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-image=$container_image \
    --container-workdir=$verl_workdir \
    --container-writable \
    bash -c "ray start --head --node-ip-address=$head_node_ip --port=$port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block" &

echo "Waiting for Ray head node to be ready..."
max_retries=120
for ((i=1; i<=max_retries; i++)); do
    if nc -z $head_node_ip $port >/dev/null 2>&1; then
        sleep 3
        echo "Ray head node port is available after $i seconds!"
        break
    fi
    
    echo "Attempt $i/$max_retries: Head node port not ready yet..."
    sleep 1
    
    if [ $i -eq $max_retries ]; then
        echo "Error: Ray head node failed to start within $max_retries seconds"
        exit 1
    fi
done
# sleep 60

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        --container-name=$container_name \
        --container-mounts=$container_mounts \
        --container-image=$container_image \
        --container-workdir=$verl_workdir \
        --container-writable \
        bash -c "ray start --address $ip_head --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --block" &
    sleep 5
done

sleep 10

SCRIPTS="
set -x
ulimit -n 65535

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    data.train_files=/lustre/projects/polyullm/yuhang/r2/data/train/ui_44k_0725_r1_grounding_point_enum_1.parquet \
    data.val_files=/lustre/projects/polyullm/yuhang/r2/data/validation/ssp_5600_eval.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=7168 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    custom_reward_function.path=/lustre/projects/polyullm/yuhang/r2/verl/cp/reward_fn/enum_v2_gui_reward.py \
    custom_reward_function.name=enum_v2_gui_reward_function \
    actor_rollout_ref.model.path=/lustre/projects/polyullm/models/Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name='infigui-grounder-v1' \
    trainer.experiment_name='enum_v2_rloo_bs128_n16_7b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    trainer.save_freq=16 \
    trainer.test_freq=16 \
    trainer.total_epochs=2
"

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-image=$container_image \
    --container-workdir=$verl_workdir \
    --container-writable \
    bash -c "ray status ; $SCRIPTS"


# Clean up Ray processes
cleanup() {
    echo "Shutting down Ray cluster..."
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        --container-name=$container_name \
        --container-mounts=$container_mounts \
        --container-image=$container_image \
        --container-writable \
        bash -c "ray stop"
    
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        srun --overlap --nodes=1 --ntasks=1 -w "$node_i" \
            --container-name=$container_name \
            --container-mounts=$container_mounts \
            --container-image=$container_image \
            --container-writable \
            bash -c "ray stop"
    done
}

# Set up trap to call cleanup function on script exit
trap cleanup EXIT
