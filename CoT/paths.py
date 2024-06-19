from os import path

# Main directories
cot_dir = path.dirname(path.realpath(__file__))
root_dir = path.dirname(cot_dir)
results_dir = path.join(root_dir, 'results')

# Prompts
prompt_dir = path.join(cot_dir, 'prompts')