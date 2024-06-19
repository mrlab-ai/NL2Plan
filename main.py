from NL2Plan.main import main as NL2Plan_main
from CoT.main import main as CoT_main
import argparse, os
from typing import Literal

root_dir = os.path.dirname(os.path.realpath(__file__))
domain_dir = os.path.join(root_dir, 'domains')
domains = os.listdir(domain_dir)

def main():
    # Subparser for each planner
    parser = argparse.ArgumentParser(description='Text to Plan.')
    subparsers = parser.add_subparsers(dest='planner', required=True, help="The planner to use. One of 'NL2Plan', 'CoT'")
    NL2Plan_parser = subparsers.add_parser("NL2Plan")
    cot_parser = subparsers.add_parser("CoT")

    # Joint arguments for each planner
    for sub in [NL2Plan_parser, cot_parser]:
        sub.add_argument('--domain', default='logistics', type=str, help='The domain name.', choices=domains, nargs='?')
        sub.add_argument('--task', default=1, type=int, help='The task.', nargs='?')
        sub.add_argument('--llm', type=str, default='gpt-4-1106-preview', help='The LLM engine name.', nargs='?') 

    # NL2Plan arguments
    NL2Plan_parser.add_argument('--feedback', default='llm', type=str, help='Which feedback source to use.', choices=["llm","human","none"], nargs='?')
    NL2Plan_parser.add_argument('--act_constr_iters', type=int, default=2, help='The maximum number of iterations.')
    NL2Plan_parser.add_argument('--full_history', action='store_true', help='Whether to send full (not shortened) messages.')
    NL2Plan_parser.add_argument('--max_attempts', type=int, default=8, help='The maximum number of messages.')
    NL2Plan_parser.add_argument('--mute_solver', action='store_true', help='Whether to mute the solver.')
    NL2Plan_parser.add_argument('--start_from', type=int, default=1, help='The step to start from.', nargs='?')
    NL2Plan_parser.add_argument('--start_dir', type=str, default=None, help='The path to start from.', nargs='?')
    NL2Plan_parser.add_argument('--no_checkpoints', action='store_true', help='Whether to not save checkpoints.')

    # CoT arguments
    # Doesn't need any further arguments
    
    args = parser.parse_args()

    with open(os.path.join(root_dir, 'domains', args.domain, f'desc.txt'), 'r') as f:
        desc = f.read()
    with open(os.path.join(root_dir, 'domains', args.domain, f'task{args.task}.txt'), 'r') as f:
        task = f.read()
    args.desc_task = f'{desc}\n\n{task}'

    print(f"Running {args.planner} on domain {args.domain} with task {args.task} and LLM {args.llm}.\n{'-'*50}")

    if args.planner == "NL2Plan":
        plan = NL2Plan_planner(args)
    elif args.planner == "CoT":
        plan = CoT_planner(args)
    else:
        raise ValueError("Invalid planner.")

def NL2Plan_planner(args):
    feedback = args.feedback if args.feedback != "none" else None
    plan = NL2Plan_main(
        domain_name = args.domain, 
        domain_task = args.desc_task,
        engine = args.llm, 
        act_constr_iters = args.act_constr_iters, 
        shorten_message = not args.full_history, 
        max_attempts = args.max_attempts, 
        mute_solver = args.mute_solver, 
        feedback = feedback,
        checkpoints = not args.no_checkpoints,
        start_from = args.start_from,
        start_dir = args.start_dir,
    )
    return plan

def CoT_planner(args):
    return CoT_main(args.domain, args.desc_task, args.llm)

if __name__ == "__main__":
    main()