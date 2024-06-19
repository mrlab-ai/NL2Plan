from os import path

# Main directories
utils_dir = path.dirname(path.realpath(__file__))
ips_dir = path.dirname(utils_dir)
root_dir = path.dirname(ips_dir)
results_dir = path.join(root_dir, 'results')

# Prompts
prompt_dir = path.join(ips_dir, 'prompts')
type_extraction_prompts = path.join(prompt_dir, '1_type_extraction')
type_hierarchy_prompts = path.join(prompt_dir, '2_hierarchy_construction')
action_extraction_prompts = path.join(prompt_dir, '3_action_extraction')
action_construction_prompts = path.join(prompt_dir, '4_action_construction')
state_goal_extraction_prompts = path.join(prompt_dir, '5_task_extraction')

# External tools
scorpion_dir = path.join(root_dir, 'scorpion')