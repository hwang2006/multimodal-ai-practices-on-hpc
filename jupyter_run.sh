#!/bin/bash
#SBATCH --comment=pytorch
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100_4
#SBATCH --time=12:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=8     # number of cpus per task

set +e

#######################################
# Config you may tweak
#######################################
SERVER="$(hostname)"
OLLAMA_PORT=11434

# Set a model to preload/warm in the UI; set to 0 to disable preloading
#DEFAULT_MODEL="gemma:latest"
#DEFAULT_MODEL="llama3.2-vision:latest"
DEFAULT_MODEL=0

WORK_DIR="/scratch/$USER/multimodal-ai-use-cases-on-hpc/"
OLLAMA_MODELS="/scratch/$USER/.ollama"

# Force NVIDIA path by unsetting AMD/ROCm vars
unset ROCR_VISIBLE_DEVICES

# Detect SLURM job ID or set a fallback
JOB_ID="${SLURM_JOB_ID:-none}"

if [ "$JOB_ID" = "none" ]; then
    OLLAMA_LOG="${WORK_DIR}/ollama_server.log"
    PORT_FWD_FILE="${WORK_DIR}/ollama_port_forwarding.txt"
else
    OLLAMA_LOG="${WORK_DIR}/ollama_server_${JOB_ID}.log"
    PORT_FWD_FILE="${WORK_DIR}/ollama_port_forwarding_${JOB_ID}.txt"
fi

export TMPDIR="/scratch/${USER}/tmp"

mkdir -p "$WORK_DIR" "$OLLAMA_MODELS" "$TMPDIR"

#######################################
# Cleanup — kill only what we started
#######################################
cleanup() {
  echo "[$(date)] Cleaning up processes..."

  # Try to gracefully stop models (best effort; ignore failures)
  if curl -fsS --max-time 2 "http://127.0.0.1:${OLLAMA_PORT}/api/ps" >/dev/null 2>&1; then
    singularity exec --nv ./ollama_latest.sif ollama stop all >/dev/null 2>&1 || true
  fi

  # Kill the entire Ollama serve process group (that we created with setsid)
  if [ -n "${OLLAMA_PGID:-}" ]; then
    kill -TERM -- -"${OLLAMA_PGID}" 2>/dev/null || true
    sleep 3
    kill -KILL -- -"${OLLAMA_PGID}" 2>/dev/null || true
  elif [ -n "${OLLAMA_PID:-}" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    # Fallback: kill by parent PID (children first)
    pkill -TERM -P "$OLLAMA_PID" 2>/dev/null || true
    kill -TERM "$OLLAMA_PID" 2>/dev/null || true
    sleep 3
    pkill -KILL -P "$OLLAMA_PID" 2>/dev/null || true
    kill -KILL "$OLLAMA_PID" 2>/dev/null || true
  fi

  echo "[$(date)] Cleanup complete"
}
trap cleanup EXIT INT TERM

#######################################
# Info
#######################################
echo "========================================"
echo "Starting Ollama"
echo "Date: $(date)"
echo "Server: $SERVER"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Ollama Port: $OLLAMA_PORT"
echo "Default Model: $DEFAULT_MODEL"
echo "========================================"
echo "ssh -L localhost:${OLLAMA_PORT}:${SERVER}:${OLLAMA_PORT} ${USER}@neuron.ksc.re.kr" > "$PORT_FWD_FILE"

#######################################
# Clean stale logs / procs (narrow match)
#######################################
rm -f "$OLLAMA_LOG" 

#######################################
# Start Ollama (Singularity RUN) in its own process group
#######################################
echo "🚀 Starting Ollama server..."
cd "$WORK_DIR"  # ensure ollama_latest.sif is here

# Launch in a new session so we can kill just this group later
nohup setsid singularity run --nv \
  --env OLLAMA_LLM_LIBRARY=cuda_v12 \
  --env OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT} \
  --env OLLAMA_MODELS="$OLLAMA_MODELS" \
  --env OLLAMA_MAX_LOADED_MODELS=3 \
  --env OLLAMA_NUM_PARALLEL=6 \
  --env OLLAMA_FLASH_ATTENTION=1 \
  --env OLLAMA_KV_CACHE_TYPE=f16 \
  --env OLLAMA_GPU_OVERHEAD=209715200 \
  --env OLLAMA_KEEP_ALIVE=30m \
  --env OLLAMA_MAX_QUEUE=128 \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env OLLAMA_FORCE_GPU=1 \
  --env DEFAULT_MODEL="${DEFAULT_MODEL}" \
  ./ollama_latest.sif serve > "$OLLAMA_LOG" 2>&1 &

OLLAMA_PID=$!
# Get the process group id of the singularity process we just started
OLLAMA_PGID="$(ps -o pgid= "$OLLAMA_PID" | tr -d ' ')"
echo "Ollama PID: $OLLAMA_PID (PGID: $OLLAMA_PGID)"

#######################################
# Wait for Ollama API
#######################################
MAX_WAIT=180
COUNTER=0
while [ $COUNTER -lt $MAX_WAIT ]; do
  if curl -s "http://127.0.0.1:${OLLAMA_PORT}/api/tags" >/dev/null; then
    echo "✅ Ollama API is up!"
    break
  fi
  COUNTER=$((COUNTER + 2))
  echo "Waiting for Ollama API... (${COUNTER}s)"
  sleep 2
done
if [ $COUNTER -ge $MAX_WAIT ]; then
  echo "❌ Ollama API startup timeout"
  tail -60 "$OLLAMA_LOG" || true
  exit 1
fi

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the port and node name
#SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

#######################################
# Env / modules
#######################################
if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
module load gcc/10.2.0 cuda/12.4 cmake/3.26.2

# Activate conda environment
source ~/.bashrc
conda activate multimodal-ai

#######################################
# Start Jupyter 
#######################################
echo "🚀 Starting Jupyter server..."
cd "$WORK_DIR"

jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your account ID
echo "end of the job"
