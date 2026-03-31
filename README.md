# auto_computation_graph


generate_traces.py
1. Generate the questions. Have q different questions. For each question, have f different formats. Randomly sample prerequisite values -- create about b different examples. For each each example, have a model answer the question with cot. 
    1. Ultimately have q\*f\*b different cot traces. 


intervene_fix_traces.py
2. Ensure the answer is correct first. 
    1. API calls to fix up generated text -- if answer is wrong, get them to fix the values. Also return all numerical values in the reasoning cot so we can find it later. 

intervene_generate_pairs.py
3. Generate a counterfactual prerequisite data tuple for each cot trace. Recreate the cot trace with the counterfactual tuple (LLM can do this). 
    1. First sample counterfactual values from the same distributions. 
    2. Can pass in pervious numerical values to give pointers on where to substitute. 
    3. Also return the modified numerical values. 

post_process_pairs.py
4. Post-process pairs
    1. Pairs must have same number of tokens
        1. Easy for qwen tokenizer -- post process and grep values and fix source/base to have identical lengths of each value. 

intervene_graph.py
5. 