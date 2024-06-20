import os, re, itertools, copy

from .utils.pddl_output_utils import parse_new_predicates, parse_params, combine_blocks
from .utils.pddl_types import Predicate, Action
from .utils.paths import action_construction_prompts as prompt_dir
from .utils.logger import Logger
from .utils.pddl_generator import PddlGenerator
from .utils.human_feedback import human_feedback
from .pddl_syntax_validator import PDDL_Syntax_Validator
from .hierarchy_construction import Hierarchy
from .llm_model import LLM_Chat, get_llm

@Logger.section("4 Action Construction")
def action_construction(
        llm_conn: LLM_Chat, 
        action_descs: dict[str, str], 
        domain_desc_str: str, 
        type_hierarchy: Hierarchy,
        unsupported_keywords: list[str] = [],
        feedback: str | None = None,
        max_attempts: int = 8,
        shorten_message: bool = False,
        max_iters: int = 2,
        mirror_symmetry: bool = False
    ) -> tuple[list[Action], list[Predicate]]:
    """
    Construct actions from a given domain description using an LLM_Chat language model.

    Args:
        llm_conn (LLM_Chat): The LLM_Chat language model connection.
        actions (dict[str, str]): A dictionary of actions to construct, where the keys are action names and the values are action descriptions.
        domain_desc_str (str): The domain description string.
        type_hierarchy (Hierarchy): The type hierarchy.
        unsupported_keywords (list[str]): A list of unsupported keywords. 
        feedback (bool): Whether to request feedback from the language model. Defaults to True.
        max_attempts (int): The maximum number of messages to send to the language model. Defaults to 8.
        shorten_message (bool): Whether to shorten the messages sent to the language model. Defaults to False.
        max_iters (int): The maximum number of iterations to construct each action. Defaults to 2.
        mirror_symmetry (bool): Whether to mirror any symmetrical predicates used process. Defaults to False.

    Returns:
        list[Action]: A list of constructed actions.
        list[Predicate]: A list of predicates.
    """
    llm_conn.reset_token_usage()

    action_list = "\n".join([f"- {name}: {desc}" for name, desc in action_descs.items()])

    with open(os.path.join(prompt_dir, "main.txt")) as f:
        act_constr_template = f.read().strip()
    act_constr_template = act_constr_template.replace('{domain_desc}', domain_desc_str)
    act_constr_template = act_constr_template.replace('{type_hierarchy}', str(type_hierarchy))
    act_constr_template = act_constr_template.replace('{action_list}', action_list)

    with open(os.path.join(prompt_dir, "feedback.txt")) as f:
        feedback_template = f.read().strip()
    feedback_template = feedback_template.replace('{domain_desc}', domain_desc_str)
    feedback_template = feedback_template.replace('{type_hierarchy}', str(type_hierarchy))
    feedback_template = feedback_template.replace('{action_list}', action_list)

    syntax_validator = PDDL_Syntax_Validator(type_hierarchy, unsupported_keywords=unsupported_keywords)

    predicates = []
    for iter in range(max_iters):
        actions = []
        Logger.print(f"Starting iteration {iter + 1} of action construction", subsection=False)
        current_preds = len(predicates)
        for action_name, action_desc in action_descs.items():
            action, new_predicates = construct_action(
                llm_conn, act_constr_template, action_name, action_desc, predicates, syntax_validator, feedback_template, 
                max_iters=max_attempts, feedback=feedback, shorten_message=shorten_message, mirror_symmetry=mirror_symmetry
            )
            actions.append(action)
            predicates.extend(new_predicates) 
        if len(predicates) == current_preds:
            Logger.print("No new predicates created. Stopping action construction.", subsection=False)
            break
    else:
        Logger.print("Reached maximum iterations. Stopping action construction.", subsection=False)

    predicates = prune_predicates(predicates, actions) # Remove predicates that are not used in any action
    types = type_hierarchy.types()
    pruned_types = prune_types(types, predicates, actions) # Remove types that are not used in any predicate or action

    Logger.print("Constructed actions:\n", "\n".join([str(action) for action in actions]))
    PddlGenerator.reset_actions()
    for action in actions:
        PddlGenerator.add_action(action)
    predicate_str = "\n".join([pred["clean"].replace(":", " ; ") for pred in predicates])
    PddlGenerator.set_predicates(predicate_str)
    Logger.print(f"PREDICATES: {predicate_str}")
    
    in_tokens, out_tokens = llm_conn.token_usage()
    Logger.add_to_info(Action_Construction_Tokens=(in_tokens, out_tokens))

    return actions, predicates, pruned_types

def construct_action(
        llm_conn: LLM_Chat, 
        act_constr_prompt: str,
        action_name: str,
        action_desc: str,
        predicates: list[Predicate],
        syntax_validator: PDDL_Syntax_Validator,
        feedback_template: str = None,
        max_iters=8,
        shorten_message=False,
        feedback=True,
        mirror_symmetry=False
    ) -> tuple[Action, list[Predicate]]:
    """
    Construct an action from a given action description using a LLM_Chat language model.

    Args:
        llm_conn (LLM_Chat): The LLM_Chat language model connection.
        act_constr_prompt (str): The action construction prompt.
        action_name (str): The name of the action.
        action_desc (str): The action description.
        predicates list[Predicate]: A list of predicates.
        syntax_validator (PDDL_Syntax_Validator): The PDDL syntax validator.
        feedback_template (str): The feedback template. Has to be specified if feedback used. Defaults to None.
        max_iters (int): The maximum number of iterations to construct the action. Defaults to 8.
        shorten_message (bool): Whether to shorten the messages sent to the language model. Defaults to False.
        feedback (bool): Whether to request feedback from the language model. Defaults to True.
        mirror_symmetry (bool): Whether to mirror any symmetrical predicates in the action preconditions. Defaults to False.

    Returns:
        Action: The constructed action.
        new predicates list[Predicate]: A list of new predicates.
    """

    act_constr_prompt = act_constr_prompt.replace('{action_desc}', action_desc)
    act_constr_prompt = act_constr_prompt.replace('{action_name}', action_name)
    if len(predicates) == 0:
        predicate_str = "No predicate has been defined yet"
    else:
        predicate_str = ""
        for i, pred in enumerate(predicates): predicate_str += f"{i+1}. {pred['name']}: {pred['desc']}\n"            
    act_constr_prompt = act_constr_prompt.replace('{predicate_list}', predicate_str)

    if feedback_template is not None:
        feedback_template = feedback_template.replace('{action_desc}', action_desc)
        feedback_template = feedback_template.replace('{action_name}', action_name)
    elif feedback:
        raise ValueError("Feedback template is required when feedback is enabled.")

    messages = [{'role': 'user', 'content': act_constr_prompt}]

    received_feedback_at = None
    for iter in range(1, max_iters + 1 + (feedback is not None)):
        Logger.print(f'Generating PDDL of action: `{action_name}` | # of messages: {len(messages)}', subsection=False)

        msgs_to_send = messages if not shorten_message else shorten_messages(messages)
        Logger.log("Messages to send:\n", "\n".join([m["content"] for m in msgs_to_send]))
        llm_output = llm_conn.get_response(prompt=None, messages=msgs_to_send)
        messages.append({'role': 'assistant', 'content': llm_output})
        Logger.print("LLM Output:\n", llm_output)

        try:
            new_predicates = parse_new_predicates(llm_output)
            validation_info = syntax_validator.perform_validation(llm_output, curr_predicates = predicates, new_predicates = new_predicates)
            no_error, error_type, _, error_msg = validation_info
        except Exception as e:
            no_error = False
            error_msg = str(e)
            error_type = str(e.__class__.__name__)

        if no_error or error_type == "all_validation_pass":
            if received_feedback_at is None and feedback is not None:
                received_feedback_at = iter
                error_type = "feedback"
                if feedback.lower() == "human":
                    action = parse_action(llm_output, action_name)
                    new_predicates = parse_new_predicates(llm_output)
                    preds = "\n".join([f"\t- {pred['clean']}" for pred in new_predicates])
                    msg  = f"\n\nThe action {action_name} has been constructed.\n\n"
                    msg += f"Action desc: \n\t{action_desc}\n\n"
                    msg += f"Parameters: \n\t{action['parameters']}\n\n"
                    msg += f"Preconditions: \n{action['preconditions']}\n\n"
                    msg += f"Effects: \n{action['effects']}\n\n"
                    msg += f"New predicates: \n{preds}\n"
                    error_msg = human_feedback(msg)
                else:
                    error_msg = get_llm_feedback(llm_conn, feedback_template, llm_output, predicates, new_predicates)
                if error_msg is None:
                    break # No feedback and no error, so we can stop iterating
            else:
                break # No error and feedback finished, so we can stop iterating

        Logger.print(f"Error of type {error_type} for action {action_name} iter {iter}:\n{error_msg}", subsection=False)

        with open(os.path.join(prompt_dir, "error.txt")) as f:
            error_template = f.read().strip()
        error_prompt = error_template.replace('{action_name}', action_name)
        error_prompt = error_prompt.replace('{action_desc}', action_desc)
        error_prompt = error_prompt.replace('{predicate_list}', predicate_str)
        error_prompt = error_prompt.replace('{error_msg}', error_msg)

        messages.append({'role': 'user', 'content': error_prompt})
    else:
        Logger.print(f"Reached maximum iterations. Stopping action construction for {action_name}.", subsection=False)

    action = parse_action(llm_output, action_name)
    new_predicates = parse_new_predicates(llm_output)
    # Remove re-defined predicates
    new_predicates = [pred for pred in new_predicates if pred['name'] not in [p["name"] for p in predicates]]

    if mirror_symmetry:
        action = mirror_action(action, predicates + new_predicates)

    return action, new_predicates

def shorten_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Only keep the latest LLM output and correction feedback
    """
    if len(messages) == 1:
        return [messages[0]]
    else:
        short_message = [messages[0]] + messages[-2:]
        assert short_message[1]['role'] == 'assistant'
        assert short_message[2]['role'] == 'user'
        return short_message

def parse_action(llm_output: str, action_name: str) -> Action:
    """
    Parse an action from a given LLM output.

    Args:
        llm_output (str): The LLM output.
        action_name (str): The name of the action.

    Returns:
        Action: The parsed action.
    """
    #parameters = llm_output.split("Parameters:")[1].split("```")[1].strip()
    parameters = parse_params(llm_output)
    try:
        preconditions = llm_output.split("Preconditions\n")[1].split("##")[0].split("```")[1].strip(" `\n")
    except:
        raise Exception("Could not find the 'Preconditions' section in the output. Provide the entire response, including all headings even if some are unchanged.")
    try:
        effects = llm_output.split("Effects\n")[1].split("##")[0].split("```")[1].strip(" `\n")
    except:
        raise Exception("Could not find the 'Effects' section in the output. Provide the entire response, including all headings even if some are unchanged.")
    return {"name": action_name, "parameters": parameters, "preconditions": preconditions, "effects": effects}

def get_llm_feedback(llm_conn: LLM_Chat, feedback_template: str, llm_output: str, predicates: list[Predicate], new_predicates: list[Predicate]) -> str | None:
    all_predicates = predicates + [pred for pred in new_predicates if pred['name'] not in [p["name"] for p in predicates]]
    action_params = combine_blocks(llm_output.split("Parameters")[1].split("##")[0])
    action_preconditions = llm_output.split("Preconditions")[1].split("##")[0].split("```")[1].strip(" `\n")
    action_effects = llm_output.split("Effects")[1].split("##")[0].split("```")[-2].strip(" `\n")
    predicate_list = "\n".join([f"- {pred['name']}: {pred['desc']}" for pred in all_predicates])

    feedback_prompt = feedback_template.replace('{action_params}', action_params)
    feedback_prompt = feedback_prompt.replace('{action_preconditions}', action_preconditions)
    feedback_prompt = feedback_prompt.replace('{action_effects}', action_effects)
    feedback_prompt = feedback_prompt.replace('{predicate_list}', predicate_list)

    Logger.print("Requesting feedback from LLM", subsection=False)
    Logger.log("Feedback prompt:\n", feedback_prompt)
    feedback = llm_conn.get_response(prompt=feedback_prompt).strip()
    Logger.log("Received feedback:\n", feedback)
    if "no feedback" in feedback.lower() or len(feedback.strip()) == 0:
        Logger.print(f"No Received feedback:\n {feedback}")
        return None
    
    return feedback

def prune_predicates(predicates: list[Predicate], actions: list[Action]) -> list[Predicate]:
    """
    Remove predicates that are not used in any action.

    Args:
        predicates (list[Predicate]): A list of predicates.
        actions (list[Action]): A list of actions.

    Returns:
        list[Predicate]: The pruned list of predicates.
    """
    used_predicates = []
    for pred in predicates:
        for action in actions:
            # Add a space or a ")" to avoid partial matches 
            names = [f"{pred['name']} ", f"{pred['name']})"]
            for name in names:
                if name in action['preconditions'] or name in action['effects']:
                    used_predicates.append(pred)
                    break
    return used_predicates

def mirror_action(action: Action, predicates: list[Predicate]):
    """
    Mirror any symmetrical predicates used in the action preconditions. 

    Example:
        Original action:
        (:action drive
            :parameters (
                ?truck - truck
                ?from - location
                ?to - location
            )
            :precondition
                (and
                    (at ?truck ?from)
                    (connected ?to ?from)
                )
            :effect
                (at ?truck ?to )
            )
        )
        
        Mirrored action:
        (:action drive
            :parameters (
                ?truck - truck
                ?from - location
                ?to - location
            )
            :precondition
                (and
                    (at ?truck ?from)
                    ((connected ?to ?from) or (connected ?from ?to))
                )
            :effect
                (at ?truck ?to )
            )
        )
    """
    mirrored = copy.deepcopy(action)
    for pred in predicates:
        if pred["name"] not in action["preconditions"]:
            continue # The predicate is not used in the preconditions
        param_types = list(pred["params"].values())
        for type in set(param_types): 
            # For each type
            if not param_types.count(type) > 1:
                continue # The type is not repeated
            # The type is repeated
            occs = [i for i, x in enumerate(param_types) if x == type]
            perms = list(itertools.permutations(occs))
            if len(occs) > 2:
                Logger.print(f"[WARNING] Mirroring predicate with {len(occs)} occurences of {type}.", subsection=False)
            uses = re.findall(f"\({pred['name']}.*\)", action["preconditions"]) # Find all occurrences of the predicate used in the preconditions
            for use in uses:
                versions = [] # The different versions of the predicate
                args = [use.strip(" ()").split(" ")[o+1] for o in occs] # The arguments of the predicate
                template = use
                for i, arg in enumerate(args): # Replace the arguments with placeholders
                    template = template.replace(arg, f"[MIRARG{i}]", 1)
                for perm in perms:
                    ver = template
                    for i, p in enumerate(perm):
                        # Replace the placeholders with the arguments in the permutation
                        ver = ver.replace(f"[MIRARG{i}]", args[p])
                    if ver not in versions:
                        versions.append(ver) # In case some permutations are the same (repeated args)
                combined = "(" + " or ".join(versions) + ")"
                mirrored["preconditions"] = mirrored["preconditions"].replace(use, combined)
    return mirrored

def prune_types(types: list[str], predicates: list[Predicate], actions: list[Action]):
    """
    Prune types that are not used in any predicate or action.

    Args:
        types (list[str]): A list of types.
        predicates (list[Predicate]): A list of predicates.
        actions (list[Action]): A list of actions.

    Returns:
        list[str]: The pruned list of types.
    """
    used_types = []
    for type in types:
        for pred in predicates:
            if type in pred['params'].values():
                used_types.append(type)
                break
        else:
            for action in actions:
                if type in action['parameters'].values():
                    used_types.append(type)
                    break
                if type in action['preconditions'] or type in action['effects']: # If the type is included in a "forall" or "exists" statement
                    used_types.append(type)
                    break
    return used_types

if __name__ == "__main__":
    
    Logger.start("demo")
    PddlGenerator.start()

    llm_conn = get_llm(engine='gpt-3.5-turbo-0125')

    domain_desc = "The AI agent here is a logistics planner that has to plan to transport packages within the locations in a city through a truck and between cities through an airplane. Also, there is no limit to how many packages a truck or plane can carry (so in theory a truck or plane can carry an infinite number of packages)."
    action_descs = {
        'drive': "Move a truck from a location to another in the same city. Example: truck1 moves from loc1 to loc2 in city1.",
        'load': "Load a vehicle with a package at the same location. Example: truck1 loads package1 at loc1.",
        'unload': "Remove a package from a vehicle and place it at the vehicle's location. Example: truck1 unloads package1.",
    }

    hier = "- object: object is always root, everything is an object"
    hier += "\n\t- city: The geographical area where the transportation takes place."
    hier += "\n\t- location: The origin and destination points for the packages. Trucks can move between these."    
    hier += "\n\t\t- airport: The only locations planes can move between."
    hier += "\n\t- package: The items that need to be transported from one location to another."
    hier += "\n\t- vehicle: The means of transportation for the packages, such as trucks or planes."
    hier += "\n\t\t- truck: A vehicle used to transport packages."
    hier += "\n\t\t- plane: A vehicle used to transport packages."
    type_hierarchy = Hierarchy(hier)
    print(type_hierarchy)

    actions, predicates = action_construction(llm_conn, action_descs, domain_desc, type_hierarchy, shorten_message=True, feedback="human")

    PddlGenerator.generate()