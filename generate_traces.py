"""
Generate Reasoning Traces with CoT

This script generates chain-of-thought responses for physics problems and saves:
1. Token sequences (token IDs and decoded token strings)
2. Prompt metadata (variables, hidden variables, expected answers)

The traces are saved to ~/links/scratch/traces/<model_name>/ when
available, otherwise ~/scratch/traces/<model_name>/.

Usage:
    python generate_traces.py --experiment velocity --n_prompts 250
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import prompts

# ==========================================
# CONFIGURATION
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='velocity', 
                    help='Experiment type: velocity, current, etc.')

repo_root = Path(__file__).resolve().parent.parent
default_model_path = repo_root / 'models' / 'Qwen2.5-72B'

parser.add_argument('--model_path', type=str, 
                    default=str(default_model_path),
                    help='Path to local HF model directory')
parser.add_argument('--n_prompts', type=int, default=50,
                    help='Number of prompts to generate (will be split across formats)')
parser.add_argument('--max_new_tokens', type=int, default=256,
                    help='Maximum tokens to generate per prompt')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Number of prompts to generate per model forward pass')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
#np.random.seed(args.seed)

# Output directory
model_name = Path(args.model_path).name
scratch_root = Path.home() / 'links' / 'scratch'
if not scratch_root.exists():
    scratch_root = Path.home() / 'scratch'

OUTPUT_DIR = scratch_root / 'traces' / model_name / args.experiment
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*70)
print("GENERATE REASONING TRACES WITH COT")
print("="*70)
print(f"Experiment: {args.experiment}")
print(f"Model: {args.model_path}")
print(f"Output: {OUTPUT_DIR}")
print(f"Prompts to generate: {args.n_prompts}")
print(f"Batch size: {args.batch_size}")
print(f"Max new tokens: {args.max_new_tokens}")
print()

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading model...")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

print("Loading HuggingFace model with device_map='auto'...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

print(f"Model loaded: {model.config.num_hidden_layers} layers, {model.config.hidden_size} dimensions\n")

# ==========================================
# PROMPT GENERATION
# ==========================================

def generate_prompts_with_cot_wrapper(experiment_name, n_prompts):
    """
    Generate prompts using the prompts module and add CoT instruction wrapper.
    
    Args:
        experiment_name: Name of experiment (e.g., 'velocity_from_ke', 'current_from_power')
        n_prompts: Total number of prompts to generate
    
    Returns:
        List of prompt dictionaries with CoT instruction added
    """
    # Calculate samples per format (assuming 5 formats per experiment)
    samples_per_format = n_prompts // 5
    
    # Generate prompts using the prompts module
    prompts_data = prompts.generate_prompts_for_experiment(experiment_name, samples_per_format)
    
    # Add CoT instruction wrapper to each prompt
    for prompt_dict in prompts_data:
        original_prompt = prompts.normalize_prompt_numbers(prompt_dict['prompt'])
        #prompt_dict['prompt'] = f"Question: {original_prompt} Answer (step-by-step): "
        prompt_dict['prompt'] = f"{original_prompt}"
    
    # Trim to exact number requested (in case rounding created extras)
    return prompts_data[:n_prompts]

# ==========================================
# TRACE GENERATION
# ==========================================

def generate_trace(prompt_text, model, tokenizer, max_new_tokens=256):
    """
    Generate CoT response using HuggingFace generate().
    
    Returns:
        Dictionary with:
            - tokens: List of token IDs
            - token_strings: List of decoded token strings
            - prompt_length: Number of tokens in prompt
            - generated_text: Full generated text
    """
    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_length = inputs.input_ids.shape[1]
    
    print(f"    Generating (max {max_new_tokens} tokens)...", end='', flush=True)
    
    # Generate with greedy decoding for determinism
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding (top-1 sampling)
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract generated tokens
    generated_ids = outputs.sequences[0]
    token_ids = generated_ids.cpu().tolist()
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f" [Generated {len(token_ids) - prompt_length} tokens]")
    
    return {
        'tokens': token_ids,
        'token_strings': token_strings,
        'prompt_length': prompt_length,
        'generated_text': generated_text,
    }


def generate_trace_batch(prompt_texts, model, tokenizer, max_new_tokens=256):
    """Generate a batch of CoT responses in one forward pass."""
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
    prompt_lengths = inputs['attention_mask'].sum(dim=1).tolist()

    print(f"    Generating batch of {len(prompt_texts)} prompts (max {max_new_tokens} tokens)...", end='', flush=True)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    batch_traces = []
    for idx, generated_ids in enumerate(outputs.sequences):
        token_ids = generated_ids.cpu().tolist()
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        batch_traces.append(
            {
                'tokens': token_ids,
                'token_strings': token_strings,
                'prompt_length': int(prompt_lengths[idx]),
                'generated_text': generated_text,
            }
        )

    print(f" [Generated {len(batch_traces)} traces]")
    return batch_traces

# ==========================================
# MAIN GENERATION LOOP
# ==========================================

print(f"Generating {args.n_prompts} prompts...")

# Map common experiment names to their prompt generator names
experiment_mapping = {
    'velocity': 'velocity_from_ke',
    'velocity_uniform_t': 'velocity_from_ke_uniform_t',
    'current': 'current_from_power',
    'radius': 'radius_from_area',
    'side_length': 'side_length_from_volume',
    'wavelength': 'wavelength_from_speed',
    'cross_section': 'cross_section_from_flow',
    'displacement': 'displacement_from_spring',
    'market_cap': 'market_cap_from_shares'
}

# Get the prompt generator name
if args.experiment in experiment_mapping:
    prompt_experiment_name = experiment_mapping[args.experiment]
elif args.experiment in prompts.get_all_generators():
    prompt_experiment_name = args.experiment
else:
    available = list(experiment_mapping.keys()) + list(prompts.get_all_generators().keys())
    raise ValueError(f"Unknown experiment: {args.experiment}. Available: {available}")

prompts_data = generate_prompts_with_cot_wrapper(prompt_experiment_name, args.n_prompts)
print(f"Generated {len(prompts_data)} prompts\n")

# Generate traces
all_traces = []
batch_size = max(1, int(args.batch_size))

for batch_start in tqdm(range(0, len(prompts_data), batch_size), desc="Generating batches"):
    batch_prompts = prompts_data[batch_start:batch_start + batch_size]
    batch_prompt_texts = [prompt_data['prompt'] for prompt_data in batch_prompts]

    batch_traces = generate_trace_batch(
        batch_prompt_texts,
        model,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    for offset, (prompt_data, trace) in enumerate(zip(batch_prompts, batch_traces)):
        idx = batch_start + offset
        print(f"\n[{idx+1}/{len(prompts_data)}] Prompt: {prompt_data['prompt'][:80]}...")

        # Combine prompt metadata with trace data
        full_trace = {
            'id': idx,
            **prompt_data,  # Includes: prompt, format_id, variables, hidden variable, expected answer
            **trace  # Includes: tokens, token_strings, prompt_length, generated_text
        }

        all_traces.append(full_trace)

        # Save intermediate traces every 50 prompts
        if (idx + 1) % 50 == 0:
            traces_file = OUTPUT_DIR / 'traces.json'
            with open(traces_file, 'w') as f:
                json.dump(all_traces, f, indent=2)
            print(f"\n  Saved intermediate traces to {traces_file}")

# ==========================================
# SAVE FINAL RESULTS
# ==========================================

traces_file = OUTPUT_DIR / 'traces.json'
with open(traces_file, 'w') as f:
    json.dump(all_traces, f, indent=2)

# Save generation config
config = {
    'experiment': args.experiment,
    'model_path': args.model_path,
    'model_name': model_name,
    'n_prompts': len(all_traces),
    'max_new_tokens': args.max_new_tokens,
    'temperature': args.temperature,
    'top_p': args.top_p,
    'seed': args.seed,
}

config_file = OUTPUT_DIR / 'config.json'
with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*70)
print("GENERATION COMPLETE")
print("="*70)
print(f"Generated {len(all_traces)} traces")
print(f"Traces saved to: {traces_file}")
print(f"Config saved to: {config_file}")
print()
print("Traces contain:")
print("- Prompt text and metadata")
print("- Token sequences (IDs and decoded strings)")
print("- Prompt length and full generated text")
print()
