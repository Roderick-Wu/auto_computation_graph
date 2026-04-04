#!/bin/bash
#SBATCH --job-name=generate_traces
#SBATCH --time=0-0:15:00 # D-HH:MM
#SBATCH --account=def-rgrosse
#SBATCH --mem=128G
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=1

#salloc --account=def-zhijing --mem=128G --gpus=h100:2

# Load required modules
module load python/3.11.5
module load cuda/12.6
module load scipy-stack/2023b
module load arrow/21.0.0
#module load python cuda scipy-stack arrow

#source venv/bin/activate

#pip install -e ../TransformerLens

EXPERIMENT=${1:-velocity}
MODEL_NAME=${2:-Qwen2.5-32B}
N_PROMPTS=${3:-50}
MODEL_PATH=/home/wuroderi/links/projects/def-rgrosse/wuroderi/models/$MODEL_NAME
OUTPUT_DIR=/home/wuroderi/links/scratch/traces/$MODEL_NAME/$EXPERIMENT
TRACE_FILE="$OUTPUT_DIR/traces.json"
CONFIG_FILE="$OUTPUT_DIR/config.json"
BATCH=8

if [ -f "$TRACE_FILE" ] && [ -f "$CONFIG_FILE" ]; then
	if TRACE_FILE="$TRACE_FILE" CONFIG_FILE="$CONFIG_FILE" EXPERIMENT="$EXPERIMENT" N_PROMPTS="$N_PROMPTS" MODEL_PATH="$MODEL_PATH" python - <<'PY'
import json
import os
import sys
from pathlib import Path

trace_file = Path(os.environ["TRACE_FILE"])
config_file = Path(os.environ["CONFIG_FILE"])
expected_experiment = os.environ["EXPERIMENT"]
expected_n_prompts = int(os.environ["N_PROMPTS"])
expected_model_path = os.environ["MODEL_PATH"]

try:
	traces = json.loads(trace_file.read_text())
	config = json.loads(config_file.read_text())
except Exception:
	sys.exit(1)

if (
	isinstance(traces, list)
	and len(traces) == expected_n_prompts
	and config.get("experiment") == expected_experiment
	and config.get("model_path") == expected_model_path
	and int(config.get("n_prompts", -1)) == expected_n_prompts
):
	sys.exit(0)

sys.exit(1)
PY
	then
		echo "Skipping ${EXPERIMENT}: already generated ${N_PROMPTS} prompts in ${OUTPUT_DIR}"
		exit 0
	fi
fi

#python generate_traces.py --experiment current --n_prompts 200 --max_new_tokens 256 --model_path /home/wuroderi/projects/def-zhijing/wuroderi/models/QwQ-32B-Preview

python generate_traces.py --experiment "$EXPERIMENT" --n_prompts "$N_PROMPTS" --max_new_tokens 256 --model_path "$MODEL_PATH" --batch_size $BATCH
#python generate_traces.py --experiment $1 --n_prompts 50 --max_new_tokens 256 --model_path /home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Llama3.1-8B
#python generate_traces.py --experiment $1 --n_prompts 50 --max_new_tokens 256 --model_path /home/wuroderi/links/projects/def-rgrosse/wuroderi/models/Llama3.1-70
