# Hands-on Multimodal AI Practices on HPC

This repo contains hands-on examples of running **multimodal AI workloads** (text + image) on the **KISTI Neuron GPU cluster**, from basic multimodal LLMs to RAG and fine-tuning custom embedding models.

> **Note:** Based on [Shaw Talebi's multimodal AI tutorials](https://github.com/ShawhinT/YouTube-Blog/tree/main/multimodal-ai), adapted for HPC.

---

## Overview

Each subdirectory under `multimodal-ai-practices-on-hpc` demonstrates a different multimodal use cases designed to run on the Neuron cluster:

- **1-mm-llms** – Run multimodal LLMs (vision–language models) via Ollama to chat over **text + images** on Neuron.
- **2-mm-embeddings** – Build and query **joint image–text embeddings** (CLIP-style) for similarity search and retrieval.
- **3-multimodal-rag** – Implement a **multimodal RAG pipeline** with text/image embedding indices and a Gradio chat UI backed by `llama3.2-vision`.
- **4-ft-mm-embeddings** – Create a **YouTube title–thumbnail dataset** and fine-tune CLIP-based SentenceTransformers models (with Recall@1 / triplet evaluation) on Neuron.
- **5-ft-flux** *(optional)* – Explore fine-tuning image generation models (e.g., FLUX) in the same Neuron + Conda + Slurm setup.

---

## Environments
### KISTI Neuron GPU Cluster
Neuron is a [KISTI GPU cluster system](https://docs-ksc.gitbook.io/neuron-user-guide) consisting of 65 nodes with 260 GPUs (120 NVIDIA A100 GPUs and 140 NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is a distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

### 1. Check the Neuron system specification
```bash
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"
CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"
CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

### 2. Download Anaconda or Miniconda
Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices.

**Option 1: Anaconda**
```bash
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
```

**Option 2: Miniconda (Recommended)**
```bash
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
```

### 3. Install Miniconda
By default, conda will be installed in your home directory, which has limited disk space. You will install and create subsequent conda environments on your scratch directory.

```bash
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
no change     /scratch/qualis/miniconda3/etc/profile.d/conda.csh
modified      /home01/qualis/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

Thank you for installing Miniconda3!
```

### 4. Finalize Miniconda installation
Set environment variables including conda path:

```bash
[glogin01]$ source ~/.bashrc    # set conda path and environment variables
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 23.11.0
```

## Cloning the Repository
Clone this repository to your scratch directory:

```bash
[glogin01]$ cd /scratch/$USER
[glogin01]$ git clone https://github.com/hwang2006/hands-on-multimodal-ai-practices-on-hpc.git
[glogin01]$ cd hands-on-multimodal-ai-practices-on-hpc
[glogin01]$ ls
./   1-mm-llms/        3-multimodal-rag/    5-ft-flux/  .gitignore      LICENSE     README.md
../  2-mm-embeddings/  4-ft-mm-embeddings/  .git/       jupyter_run.sh
```

## Preparing Ollama Singularity Image
Download the Ollama container image:

```bash
[glogin01]$ singularity pull ollama_latest.sif docker://ollama/ollama:latest
INFO:    Converting OCI blobs to SIF format
INFO:    Starting build...
INFO:    Fetching OCI image...
79.6MiB / 79.6MiB [======================================================] 100 % 126.1 MiB/s 0s
11.9MiB / 11.9MiB [======================================================] 100 % 126.1 MiB/s 0s
1.8GiB / 1.8GiB [========================================================] 100 % 126.1 MiB/s 0s
28.3MiB / 28.3MiB [======================================================] 100 % 126.1 MiB/s 0s
INFO:    Extracting OCI image...
INFO:    Inserting Singularity configuration...
INFO:    Creating SIF file...
```

Verify the Ollama installation:
```bash
[glogin01]$ singularity exec ./ollama_latest.sif ollama --version
Warning: could not connect to a running Ollama instance
Warning: client version is 0.13.0
```

## Creating a Conda Virtual Environment
Create a virtual environment for multimodal AI practices.

### 1. Create a conda virtual environment with Python 3.12
```bash
[glogin01]$ conda create -n multimodal-ai python=3.12 -y
Channels:
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3/envs/multimodal-ai

  added / updated specs:
    - python=3.12
.
.
.
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate multimodal-ai
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

### 2. Install PyTorch
```bash
[glogin01]$ module load gcc/10.2.0 cmake/3.26.2 cuda/12.4
[glogin01]$ conda activate multimodal-ai
(multimodal-ai) [glogin01]$ pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
Looking in indexes: https://download.pytorch.org/whl/cu124, https://pypi.ngc.nvidia.com
Collecting torch==2.6.0
  Downloading https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl (780.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 780.4/780.4 MB 304.5 MB/s  0:00:03
.
.
.
Successfully installed MarkupSafe-2.1.5 filelock-3.19.1 fsspec-2025.9.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.5 numpy-2.1.2 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.9.86 nvidia-nvtx-cu12-12.1.105 pillow-11.3.0 sympy-1.13.1 torch-2.6.0+cu124 torchaudio-2.6.0+cu124 torchvision-0.20.0+cu124 triton-3.1.0 typing-extensions-4.15.0
```

## Running Jupyter Lab
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. JupyterLab is the latest web-based interactive development environment for notebooks, code, and data. You will run a notebook server on a worker node (*not* on a login node), which will be accessed from your browser through SSH tunneling.

<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/></p>

### Setting up Jupyter

### 1. Activate the multimodal-ai virtual environment (optional)
```bash
[glogin01]$ conda activate multimodal-ai
```

### 2. Install Jupyter
```bash
(multimodal-ai) [glogin01]$ pip install jupyter
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting jupyter
  Downloading jupyter-1.1.1-py2.py3-none-any.whl.metadata (2.0 kB)
.
.
.
Successfully installed anyio-4.11.0 argon2-cffi-25.1.0 argon2-cffi-bindings-25.1.0 arrow-1.4.0 asttokens-3.0.1 async-lru-2.0.5 attrs-25.4.0 babel-2.17.0 beautifulsoup4-4.14.2 bleach-6.3.0 certifi-2025.11.12 cffi-2.0.0 charset_normalizer-3.4.4 comm-0.2.3 debugpy-1.8.17 decorator-5.2.1 defusedxml-0.7.1 executing-2.2.1 fastjsonschema-2.21.2 fqdn-1.5.1 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 idna-3.11 ipykernel-7.1.0 ipython-9.7.0 ipython-pygments-lexers-1.1.1 ipywidgets-8.1.8 isoduration-20.11.0 jedi-0.19.2 json5-0.12.1 jsonpointer-3.0.0 jsonschema-4.25.1 jsonschema-specifications-2025.9.1 jupyter-1.1.1 jupyter-client-8.6.3 jupyter-console-6.6.3 jupyter-core-5.9.1 jupyter-events-0.12.0 jupyter-lsp-2.3.0 jupyter-server-2.17.0 jupyter-server-terminals-0.5.3 jupyterlab-4.5.0 jupyterlab-pygments-0.3.0 jupyterlab-server-2.28.0 jupyterlab_widgets-3.0.16 lark-1.3.1 matplotlib-inline-0.2.1 mistune-3.1.4 nbclient-0.10.2 nbconvert-7.16.6 nbformat-5.10.4 nest-asyncio-1.6.0 notebook-7.5.0 notebook-shim-0.2.4 packaging-25.0 pandocfilters-1.5.1 parso-0.8.5 pexpect-4.9.0 platformdirs-4.5.0 prometheus-client-0.23.1 prompt_toolkit-3.0.52 psutil-7.1.3 ptyprocess-0.7.0 pure-eval-0.2.3 pycparser-2.23 pygments-2.19.2 python-dateutil-2.9.0.post0 python-json-logger-4.0.0 pyyaml-6.0.3 pyzmq-27.1.0 referencing-0.37.0 requests-2.32.5 rfc3339-validator-0.1.4 rfc3986-validator-0.1.1 rfc3987-syntax-1.1.0 rpds-py-0.29.0 send2trash-1.8.3 six-1.17.0 sniffio-1.3.1 soupsieve-2.8 stack_data-0.6.3 terminado-0.18.1 tinycss2-1.4.0 tornado-6.5.2 traitlets-5.14.3 tzdata-2025.2 uri-template-1.3.0 urllib3-2.5.0 wcwidth-0.2.14 webcolors-25.10.0 webencodings-0.5.1 websocket-client-1.9.0 widgetsnbextension-4.0.15
```

### 3. Add the virtual environment as a Jupyter kernel
```bash
(multimodal-ai) [glogin01]$ python -m ipykernel install --user --name multimodal-ai
```

### 4. Check the list of installed kernels
```bash
(multimodal-ai) [glogin01]$ jupyter kernelspec list
Available kernels:
  python3           /home01/$USER/.local/share/jupyter/kernels/python3
  multimodal-ai     /home01/$USER/.local/share/jupyter/kernels/multimodal-ai
```

### Launching Jupyter Lab

### 5. Deactivate the virtual environment
```bash
(multimodal-ai) [glogin01]$ conda deactivate
```

### 6. Create a batch script for launching Jupyter Lab
- to create a batch script for launching a jupyter notebook server:
```
[glogin01]$ cat jupyter_run.sh
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
DEFAULT_MODEL=0

WORK_DIR="/scratch/$USER/multimodal-ai-practices-on-hpc/"
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
  --env OLLAMA_LLM_LIBRARY=cuda \
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
```

### 7. Submit the Jupyter job
```bash
[glogin01]$ sbatch jupyter_run.sh
Submitted batch job XXXXXX
```

### 8. Check if Jupyter is running
```bash
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu30
```

Check the job output:
```bash
[glogin01]$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://gpu##:#####/lab?token=...
.
.
```

### 9. Get SSH tunneling information
```bash
[glogin01]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
```

### 10. Set up SSH tunnel from your local machine
Open a new SSH client (e.g., PuTTY, MobaXterm, PowerShell, Command Prompt) on your PC or laptop and execute the port forwarding command from the previous step.

![SSH Tunnel Example](https://github.com/hwang2006/Generative-AI-with-LLMs/assets/84169368/1f5dd57f-9872-491b-8dd4-0aa99b867789)

### 11. Access Jupyter Lab in your browser
Open a web browser on your PC or laptop:

```
URL: http://localhost:8888
Password or token: $USER    # your account name on Neuron
```

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p>

---

## License
This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments
This repository is based on the [multimodal AI repository](https://github.com/ShawhinT/YouTube-Blog/tree/main/multimodal-ai) created by Shaw Talebi. The original tutorials have been adapted and extended for execution on KISTI's Neuron HPC cluster, including:
- Integration with Slurm workload manager
- Singularity container deployment for Ollama
- Conda environment setup for HPC systems
- SSH tunneling for Jupyter Lab access

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact
For questions or support regarding KISTI Neuron cluster access, please contact KISTI support.
