import os, pickle, argparse

from .utils.logger import Logger
from .utils.pddl_generator import PddlGenerator
from .utils.paths import root_dir

from .llm_model import get_llm
from .hierarchy_construction import hierarchy_construction
from .type_extraction import type_extraction
from .action_extraction import action_extraction
from .action_construction import action_construction
from .task_extraction import task_extraction
from .planning import planning

def main(
        domain_name: str,
        domain_task: str, 
        engine: str, 
        act_constr_iters: int = 2, 
        shorten_message: bool = False, 
        max_attempts: int = 8,
        mute_solver: bool = False,
        feedback: str | None = None,
        mirror: bool = False,
        checkpoints: bool = False,
        start_from: int = 1,
        start_dir: str = None,
    ) -> list[str] | None:
    Logger.start(
        domain_name,
        domain_desc_task = domain_task, 
        engine=engine,
        act_constr_iters=act_constr_iters,
        shorten_message=shorten_message,
        max_attempts=max_attempts,
        mute_solver=mute_solver,
        feedback=feedback,
        mirror=mirror,
        checkpoints=checkpoints,
    )
    if start_dir is None and start_from > 1:
        raise ValueError("Start path must be provided if starting from a checkpoint.")
    
    # init PDDL generator
    PddlGenerator.start()

    # init LLM
    llm_gpt = get_llm(engine=engine)

    # extract domain info
    Logger.add_domain_desc(domain_task)

    # extract the available types
    if start_from <= 1:
        types = type_extraction(llm_gpt, domain_task, feedback=feedback)
        PddlGenerator.generate()

        if checkpoints:
            with open(f"{Logger.directory}/checkpoint1.pkl", 'wb') as f:  # Open the file in binary mode
                pickle.dump([types, PddlGenerator], f)  # Write the data as bytes
    elif start_from == 2:
        with open(os.path.join(start_dir, "checkpoint1.pkl"), 'rb') as f:
            types, PddlGenerator2 = pickle.load(f)
        PddlGenerator.copy(PddlGenerator2)


    if start_from <= 2:
        # construct the type hierarchy
        type_hierarchy = hierarchy_construction(
            llm_gpt, types, domain_desc=domain_task,replace_comments=True, feedback=feedback
        )
        PddlGenerator.generate()

        if checkpoints:
            with open(f"{Logger.directory}/checkpoint2.pkl", 'wb') as f:
                pickle.dump([types, type_hierarchy, PddlGenerator], f)
    elif start_from == 3:
        with open(os.path.join(start_dir, "checkpoint2.pkl"), 'rb') as f:
            types, type_hierarchy, PddlGenerator2 = pickle.load(f)
        PddlGenerator.copy(PddlGenerator2)

    # extract action info
    if start_from <= 3:
        action_desc = action_extraction(llm_gpt, domain_task, type_hierarchy, feedback=feedback)
        PddlGenerator.generate()

        if checkpoints:
            with open(f"{Logger.directory}/checkpoint3.pkl", 'wb') as f:
                pickle.dump([types, type_hierarchy, action_desc, PddlGenerator], f)
    elif start_from == 4:
        with open(os.path.join(start_dir, "checkpoint3.pkl"), 'rb') as f:
            types, type_hierarchy, action_desc, PddlGenerator2 = pickle.load(f)
        PddlGenerator.copy(PddlGenerator2)

    # construct the actions
    if start_from <= 4:
        actions, predicates, pruned_types = action_construction(
            llm_gpt, action_desc, domain_task, type_hierarchy, feedback=feedback, 
            shorten_message=shorten_message, max_attempts=max_attempts, max_iters=act_constr_iters,
            unsupported_keywords=[], mirror_symmetry=mirror
        )
        # remove types that are not used in the actions or predicates
        type_hierarchy = type_hierarchy.prune_to(pruned_types)
        PddlGenerator.set_types(type_hierarchy.type_list())

        PddlGenerator.generate()
        if checkpoints:
            with open(f"{Logger.directory}/checkpoint4.pkl", 'wb') as f:
                pickle.dump([types, type_hierarchy, predicates, actions, PddlGenerator], f)
    elif start_from == 5:
        with open(os.path.join(start_dir, "checkpoint4.pkl"), 'rb') as f:
            types, type_hierarchy, predicates, actions, PddlGenerator2 = pickle.load(f)
        PddlGenerator.copy(PddlGenerator2)

    # extract goal and initial state
    if start_from <= 5:
        goal, state, objects = task_extraction(llm_gpt, domain_task, type_hierarchy, predicates, shorten_message=shorten_message, remaining_attempts=max_attempts, feedback=feedback)
        PddlGenerator.generate()
        if checkpoints:
            with open(f"{Logger.directory}/checkpoint5.pkl", 'wb') as f:
                pickle.dump([types, type_hierarchy, predicates, actions, objects, goal, state, PddlGenerator], f)
    elif start_from == 6:
        with open(os.path.join(start_dir, "checkpoint5.pkl"), 'rb') as f:
            types, type_hierarchy, predicates, actions, objects, goal, state, PddlGenerator2 = pickle.load(f)
        PddlGenerator.copy(PddlGenerator2)

    # Solve the generated PDDL
    PddlGenerator.generate() # Generate the final PDDL file
    plan = planning(mute=mute_solver)

    return plan

if __name__ == "__main__":
    print("To demo the NL2Plan pipeline, instead run the main.py script in the root directory.")