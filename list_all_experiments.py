#!/usr/bin/env python3
"""Print all experiments registered in prompts.py as TSV lines: experiment<TAB>n_formats."""

import ast
from pathlib import Path

mod = ast.parse(Path(__file__).with_name('prompts.py').read_text())

generators = {}
format_counts = {}

for node in mod.body:
    if isinstance(node, ast.FunctionDef):
        if node.name == 'get_all_generators':
            for st in node.body:
                if isinstance(st, ast.Return) and isinstance(st.value, ast.Dict):
                    for key_node, value_node in zip(st.value.keys, st.value.values):
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str) and isinstance(value_node, ast.Name):
                            generators[key_node.value] = value_node.id

        for st in node.body:
            if isinstance(st, ast.Assign):
                for target in st.targets:
                    if isinstance(target, ast.Name) and target.id == 'prompt_formats' and isinstance(st.value, ast.List):
                        format_counts[node.name] = len(st.value.elts)

for experiment_name in sorted(generators):
    function_name = generators[experiment_name]
    n_formats = format_counts.get(function_name, 5)
    print(f'{experiment_name}\t{n_formats}')
