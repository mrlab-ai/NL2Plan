import sys, os, contextlib, re
from .utils.paths import root_dir, results_dir
from .utils.logger import Logger

sys.path.append(root_dir) # To import scorpion modules
from scorpion.driver import run_components
from scorpion.driver import arguments

@Logger.section("6 Planning")
def planning(problem = None, mute: bool = True) -> list[str] | None:
    if problem is None:
        if not Logger.started:
            raise FileNotFoundError("Logger not started and no problem specified. Start logger or specify problem to solver.")
        problem = Logger.name
    
    domain_file = os.path.join(results_dir, problem, "domain.pddl")
    problem_file = os.path.join(results_dir, problem, "problem.pddl")
    plan_file = os.path.join(results_dir, problem, "plan.txt")
    print(f"Problem file: {problem_file}")
    print(f"Plan file: {plan_file}")

    # Set arguments for the Scorpion planner via spoofing command line arguments
    current_args = sys.argv
    sys.argv = [
        current_args[0],
        "--transform-task",
        "preprocess-h2",
        "--plan-file",
        plan_file, # Plan file
        "--alias",
        "lama-first",
        domain_file, # Domain file
        problem_file, # Problem file
    ]
    args = arguments.parse_args()
    sys.argv = current_args
    
    # Run Scorpion planner
    Logger.print("Running Scorpion planner on ", problem)
    with stdout_redirected(mute=mute): # Suppress output from planner
        # Translate
        exitcode, _ = run_components.run_translate(args)
        if exitcode != 0:
            raise Exception(f"Error translating problem: {exitcode}")

        # Transform
        exitcode, _ = run_components.transform_task(args)
        if exitcode != 0:
            raise Exception(f"Error transforming problem: {exitcode}")

        # Search
        exitcode, _ = run_components.run_search(args)
        if exitcode != 0 and exitcode != 12: # 12 is "Search ended without finding a solution."
            raise Exception(f"Error searching problem: {exitcode}")
        os.remove(args.sas_file) # Remove intermediate file

    # Find plan, output as a file
    plan = get_plan(args)

    if plan is None:
        Logger.print("Problem not solvable.")
    else:
        Logger.print("Plan for ", problem, ":\n - ", "\n - ".join(plan))
    return plan

def get_plan(args) -> list[str] | None:
    """Get plan from file specified by args, or None if no plan found."""
    try:
        with open(args.plan_file, "r") as file:
            lines = file.readlines()
            plan = []
            for line in lines:
                if line[0] == ";":
                    continue
                plan.append(line.strip(" \n()"))
        return plan
    except FileNotFoundError:
        return None # No plan found, problem not solvable

@contextlib.contextmanager
def stdout_redirected(mute : bool = False, to = os.devnull):
    """Mute stdout temporarily, mutes both python and C++ (planner) print statements."""
    if not mute:
        yield
        return
    # From: https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

if __name__ == "__main__":
    print("No demo for planning.py is available.")