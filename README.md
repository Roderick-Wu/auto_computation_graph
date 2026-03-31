# auto_computation_graph


generate_traces.py
1. Generate the questions. Have q different questions. For each question, have f different formats. Randomly sample prerequisite values -- create about b different examples. For each each example, have a model answer the question with cot. 
    - Ultimately have q\*f\*b different cot traces. 
output: traces.json

intervene_fix_traces.py
2. Ensure the answer is correct first. 
    - API calls to fix up generated text -- if answer is wrong, get them to fix the values. 
output: fixed_traces.json

intervene_generate_pairs.py
3. Generate a counterfactual prerequisite data tuple for each cot trace. Recreate the cot trace with the counterfactual tuple (LLM can do this). 
    1. First sample counterfactual values from the same distributions. 
    2. Can pass in pervious numerical values to give pointers on where to substitute. 
    3. Also return the modified numerical values. 
output: paired_traces.json

post_process_pairs.py
4. Post-process pairs
    - Pairs must have same number of tokens
        - Easy for qwen tokenizer -- post process and grep values and fix source/base to have identical lengths of each value. 

intervene_graph.py
5. Causal tracing
    for each numerical value starting from the end (only cot values):
        try patching every token at every layer and record the change in logprobs
        create heatmap + log data
    

construct_graph.py