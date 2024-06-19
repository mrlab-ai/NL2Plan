import os

from .utils.paths import state_goal_extraction_prompts as prompt_dir
from .utils.logger import Logger
from .utils.pddl_generator import PddlGenerator
from .utils.pddl_types import Predicate, ParameterList
from .utils.pddl_output_utils import combine_blocks
from .utils.human_feedback import human_feedback
from .hierarchy_construction import Hierarchy
from .llm_model import LLM_Chat, get_llm
from .action_construction import shorten_messages

@Logger.section("5 Extract Goal & State")
def task_extraction(
        llm_conn: LLM_Chat, 
        domain_desc_str: str, 
        type_hierarchy: Hierarchy, 
        predicates: list[Predicate],
        messages: list[dict[str, str]] | None = None,
        error: Exception | None = None,
        remaining_attempts: int = 8,
        shorten_message: bool = False,
        feedback: str | None = None,
    ) -> tuple[str, str, str]:
    """Extract the goal, state, and objects from the LLM output given the domain description, type hierarchy, and predicates."""
    llm_conn.reset_token_usage()
    
    if messages is None:
        predicate_str = "\n".join([f"- {pred['name']}: {pred['desc']}" for pred in predicates])

        with open(os.path.join(prompt_dir, "main.txt")) as f:
            goal_state_extr_template = f.read().strip()
        goal_state_extr_prompt = goal_state_extr_template.replace('{domain_desc}', domain_desc_str)
        goal_state_extr_prompt = goal_state_extr_prompt.replace('{type_hierarchy}', str(type_hierarchy))
        goal_state_extr_prompt = goal_state_extr_prompt.replace('{predicates}', predicate_str)
        goal_state_extr_prompt = goal_state_extr_prompt.replace('{actions}', PddlGenerator.action_descs())

        Logger.log("PROMPT:\n", goal_state_extr_prompt)
        messages = [{'role': 'user', 'content': goal_state_extr_prompt}]
    elif error is not None:
        with open(os.path.join(prompt_dir, "error.txt")) as f:
            goal_corr_template = f.read().strip()
        goal_corr_prompt = goal_corr_template.replace('{error_msg}', str(error))
        goal_corr_prompt = goal_corr_prompt.replace('{task}', "goal and state extraction")
        messages.append({'role': 'user', 'content': goal_corr_prompt})
        Logger.log("Error Correction Prompt:\n", goal_corr_prompt)
    else:
        raise ValueError("If messages are provided, error must also be provided.")

    messages_to_send = messages if not shorten_message else shorten_messages(messages)
    llm_output = llm_conn.get_response(messages=messages_to_send)
    messages.append({'role': 'assistant', 'content': llm_output})
    if error is None:
        Logger.print("LLM Output:\n", llm_output)
    else:
        Logger.print("Error Correction Response:\n", llm_output)

    if not "## Object Instances" in llm_output or \
       not "## State" in llm_output or \
       not "## Goal" in llm_output:
        Logger.print("Error before extraction: Could not find all necessary sections in the LLM output.")
        error = "Could not find all necessary sections in the LLM output. Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task."
        return task_extraction(
            llm_conn, domain_desc_str, type_hierarchy, predicates,
            messages, error, remaining_attempts - 1, shorten_message, feedback
        )

    try:
        objects, objects_str = parse_objects(llm_output, type_hierarchy, predicates)
    except ValueError as error:
        if remaining_attempts < 0:
            Logger.print("WARNING - Ran out of extraction attempts")
            objects, objects_str = parse_objects(llm_output, type_hierarchy, predicates, check_errors=False)
        else:
            Logger.print(f"Error during object construction ({remaining_attempts} attempts left):\n{error}")
            return task_extraction(
                llm_conn, domain_desc_str, type_hierarchy, predicates,
                messages, error, remaining_attempts - 1, shorten_message, feedback
            )
    Logger.print("Extracted objects: \n", objects_str)
    PddlGenerator.set_objects(objects_str)

    try:
        state = parse_state(llm_output, type_hierarchy, predicates, objects)
    except ValueError as error:
        if remaining_attempts < 0:
            Logger.print("WARNING - Ran out of extraction attempts")
            state = parse_state(llm_output, type_hierarchy, predicates, objects, check_errors=False)
        else:
            Logger.print(f"Error during state construction ({remaining_attempts} attempts left):\n{error}")
            return task_extraction(
                llm_conn, domain_desc_str, type_hierarchy, predicates,
                messages, error, remaining_attempts - 1, shorten_message, feedback
            )
    Logger.print("Extracted state: \n", state)
    PddlGenerator.set_init(state)

    try:
        goal = parse_goal(llm_output, type_hierarchy, predicates, objects)
    except ValueError as error:
        if remaining_attempts <= 0:
            Logger.print("WARNING - Ran out of extraction attempts")
            goal = parse_goal(llm_output, type_hierarchy, predicates, objects, check_errors=False)
        else:
            Logger.print(f"Error during goal construction ({remaining_attempts} attempts left):\n{error}")
            return task_extraction(
                llm_conn, domain_desc_str, type_hierarchy, predicates,
                messages, error, remaining_attempts - 1, shorten_message, feedback
            )
    Logger.print(f"Extracted goal: \n", goal)
    PddlGenerator.set_goal(goal)

    if feedback is not None:
        if feedback.lower() == "human":
            msg  =  "The goal and state have been extracted. Please review them and provide feedback.\n\n"
            msg += f"Objects:\n{objects_str}\n\nState:\n{state}\n\nGoal:\n{goal}\n\n"
            feedback_msg = human_feedback(msg)
        else:
            feedback_msg = get_llm_feedback(llm_conn, type_hierarchy, predicates, domain_desc_str, objects_str, state, goal)
        if feedback_msg is not None:
            Logger.print("Received feedback:\n", feedback_msg)
            return task_extraction(
                llm_conn, domain_desc_str, type_hierarchy, predicates,
                messages, feedback_msg, remaining_attempts, shorten_message, feedback=None
            )
        
    in_tokens, out_tokens = llm_conn.token_usage()
    Logger.add_to_info(Task_Extraction_Tokens=(in_tokens, out_tokens))

    return goal, state, objects

def parse_objects(llm_output: str, type_hierarchy: Hierarchy, predicates: list[Predicate], check_errors: bool = True) -> tuple[dict[str, str], str]:
    """Extract the objects from the LLM output and return them as a string."""
    objects_head = extract_heading(llm_output, "Object Instances")
    objects_raw = combine_blocks(objects_head)
    objects_clean = clear_comments(objects_raw, comments=[':','//','#',';','(']) # Remove comments
    objects = {obj.split(" - ")[0].strip(" `"): obj.split(" - ")[1].strip(" `").lower() for obj in objects_clean.split("\n") if obj.strip()}

    if check_errors:
        errors = []
        for obj, type in objects.items():
            if type not in type_hierarchy.types():
                errors.append(f"Type `{type}` of object `{obj}` not found in the type hierarchy. Correct it to an existing type.")
            if obj in type_hierarchy.types():
                errors.append(f"Object `{obj}` is reusing a type name. Rename it.")
            if obj in [p["name"] for p in predicates]:
                errors.append(f"Object `{obj}` is reusing a predicate name. Rename it.")

        if errors:
            raise ValueError("\n".join([f" - {error}" for error in errors]))
    
    objects_str = "\n".join([f"{obj} - {type}" for obj, type in objects.items()])
    return objects, objects_str

def parse_state(llm_output: str, type_hierarchy: Hierarchy, predicates: list[Predicate], objects: dict[str, str], check_errors: bool = True) -> str:
    """Extract the state (PDDL-init) from the LLM output and return it as a string."""
    state_head = extract_heading(llm_output, "State")
    state_raw = combine_blocks(state_head)
    state_clean = clear_comments(state_raw)

    states = []
    for line in state_clean.split("\n"):
        line = line.strip("- `()")
        if not line: # Skip empty lines
            continue
        name = line.split(" ")[0]
        if name == "not":
            neg = True
            name = line.split(" ")[1].strip("()") # Remove the `not` and the parentheses
            params = line.split(" ")[2:]
        else:
            neg = False
            params = line.split(" ")[1:] if len(line.split(" ")) > 1 else []
        states.append({"name": name, "params": params, "neg": neg})
    
    if check_errors:
        errors = check_predicates(states, type_hierarchy, predicates, objects)
        if errors:
            raise ValueError("\n".join([f" - {error}" for error in errors]))

    inner_str = [f"({state['name']} {' '.join(state['params'])})" for state in states] # The main part of each predicate
    full_str = [f"(not {inner})" if state["neg"] else inner for state, inner in zip(states, inner_str)] # Add the `not` if needed
    state_str = "\n".join(full_str) # Combine the states into a single string
    return state_str

def parse_goal(llm_output: str, type_hierarchy: Hierarchy, predicates: list[Predicate], objects: dict[str, str], check_errors: bool = True) -> str:
    """Extract the goal (PDDL-goal) from the LLM output and return it as a string."""
    goal_str = extract_heading(llm_output, "Goal")
    if goal_str.count("```") != 2:
        raise ValueError("Could not find exactly one block in the goal section of the LLM output. The goal has to be specified in a single block and as valid PDDL using the `and` and `not` operators. Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task.")
    goal_raw = goal_str.split("```")[1].strip() # Only a single block in the goal
    goal_clean = clear_comments(goal_raw)

    goal_pure = goal_clean.replace("and", "").replace("AND", "").replace("not", "").replace("NOT", "")
    goals = []
    for line in goal_pure.split("\n"):
        line = line.strip(" ()")
        if not line: # Skip empty lines
            continue
        name = line.split(" ")[0]
        params = line.split(" ")[1:] if len(line.split(" ")) > 1 else []
        goals.append({"name": name, "params": params})

    if check_errors:
        errors = check_predicates(goals, type_hierarchy, predicates, objects)
        if errors:
            raise ValueError("\n".join([f" - {error}" for error in errors]))

    return goal_clean # Since the goal uses `and` and `not` recombining it is difficult 

def check_predicates(to_check: list[dict[str, str]], type_hierarchy: Hierarchy, predicates: list[Predicate], objects: dict[str, str]) -> list[str]:
    """
    Check the validity of predicates in a given state.

    Args:
        to_check (list[dict[str, str]]): List of states or goals to check. Each state or goal is a dictionary with "name" and "params" keys.
        predicates (list[Predicate]): List of available predicates.
        objects (dict[str, str]): Dictionary of available objects.
        type_hierarchy (Hierarchy): Object type hierarchy.

    Returns:
        list[str]: List of error messages for invalid predicates.
    """
    
    errors = []
    pred_names = [p["name"] for p in predicates]
    for state in to_check:
        name = state["name"]

        # Check if the predicate exists
        if name not in pred_names:
            errors.append(f"Predicate `{name}` not found. You can only use existing predicates.")
            continue
        pred = predicates[pred_names.index(name)]

        # Check if the number of objects is correct
        if len(state["params"]) != len(pred["params"]):
            errors.append(f"Predicate `{name}` expects {len(pred['params'])} objects but {len(state['params'])} were provided.")
            continue

        for i, obj in enumerate(state["params"]):
            # Check if the object exists
            if obj not in objects:
                errors.append(f"Object `{obj}` used for predicate `{name}` is not a created object. Create it if needed, or use an existing object.")
                continue 

            # Check if the object is of the correct type
            type = objects[obj]
            if not type_hierarchy.is_subtype(type, list(pred["params"].values())[i]):
                errors.append(f"Object `{obj}` is not of the correct type for predicate `{name}`. `{name}` is a `{type}` but `{name}` expects a `{list(pred['params'].values())[i]}`.")
    return errors

def clear_comments(text: str, comments = [':','//','#',';']) -> str:
    """Remove comments from the text."""
    for comment in comments:
        text = "\n".join([line.split(comment)[0] for line in text.split("\n")])
    return text

def extract_heading(llm_output: str, heading: str):
    """Extract the text between the heading and the next second level heading in the LLM output."""
    if heading not in llm_output:
        print("#"*10, "LLM Output", "#"*10)
        print(llm_output)
        print("#"*30)
        raise ValueError(f"Could not find heading {heading} in the LLM output. Likely this is caused by a too long response and limited context length. If so, try to shorten the message and exclude objects which aren't needed for the task.")
    heading_str = llm_output.split(heading)[1].split("\n## ")[0].strip() # Get the text between the heading and the next heading
    return heading_str

def get_llm_feedback(
        llm_conn: LLM_Chat, type_hierarchy: Hierarchy, predicates: list[Predicate], 
        domain_desc_str: str, objects_str: str, state_str: str, goal_str: str
    ) -> str | None:
    predicate_list = "\n".join([f"- {pred['name']}: {pred['desc']}" for pred in predicates])

    with open(os.path.join(prompt_dir, "feedback.txt")) as f:
        feedback_template = f.read().strip()
    feedback_prompt = feedback_template.replace('{objects}', objects_str)
    feedback_prompt = feedback_prompt.replace('{state}', state_str)
    feedback_prompt = feedback_prompt.replace('{goal}', goal_str)
    feedback_prompt = feedback_prompt.replace('{predicate_list}', predicate_list)
    feedback_prompt = feedback_prompt.replace('{type_hierarchy}', str(type_hierarchy))
    feedback_prompt = feedback_prompt.replace('{domain_desc}', domain_desc_str)

    Logger.print("Requesting feedback from LLM", subsection=False)
    Logger.log("FEEDBACK PROMPT:\n", feedback_prompt)
    feedback = llm_conn.get_response(prompt=feedback_prompt).strip()

    if "no feedback" in feedback.lower() or len(feedback.strip()) == 0:
        Logger.print("FEEDBACK:\n", "No feedback.")
        Logger.log(feedback)
        return None
    Logger.print("FEEDBACK:\n", feedback)
    
    return feedback

if __name__ == "__main__":
    Logger.start("demo")
    PddlGenerator.start()

    llm_gpt = get_llm(engine='gpt-3.5-turbo-0125')
    domain_desc = "The AI agent here is a logistics planner that has to plan to transport packages within the locations in a city through a truck and between cities through an airplane. Also, there is no limit to how many packages a truck or plane can carry (so in theory a truck or plane can carry an infinite number of packages)."

    hier = "- object: object is always root, everything is an object"
    hier += "\n\t- city: The geographical area where the transportation takes place."
    hier += "\n\t- location: The origin and destination points for the packages. Trucks can move between these."    
    hier += "\n\t\t- airport: The only locations planes can move between."
    hier += "\n\t- package: The items that need to be transported from one location to another."
    hier += "\n\t- vehicle: The means of transportation for the packages, such as trucks or planes."
    hier += "\n\t\t- truck: A vehicle used to transport packages."
    hier += "\n\t\t- plane: A vehicle used to transport packages."
    type_hierarchy = Hierarchy(hier)

    predicates = [
        {
            "name": "at",
            "desc": "true if the object ?o is at the location ?l",
            "params": {"?o" : "object", "?l" : "location"}
        },
        {
            "name": "in",
            "desc": "true if the package ?p is in the vehicle ?v",
            "params": {"?p" : "package", "?v" : "vehicle"}
        },
        {
            "name": "in-city",
            "desc": "true if the location ?x is in the city ?y",
            "params": {"?x" : "location", "?y" : "city"}
        },
        {
            "name": "connected",
            "desc": "true if the location ?x is connected to the location ?y by a road",
            "params": {"?x" : "location", "?y" : "location"}
        },
    ]
    
    goal_desc = "Currently, I've got 3 packages in Olike, two of which should go to different addresses in Awar (Abo Street 3 and Acil Street 7) and the last should go to a woman in Frear. Each city has a truck and an airport, but the only plane available is currently in Klock."
    
    desc = domain_desc + "\n\n" + goal_desc

    print("Domain description:\n", desc, "\n")
    print("Type hierarchy:\n", type_hierarchy, "\n")
    print("Predicates:\n", predicates, "\n")

    task_extraction(llm_gpt, desc, type_hierarchy, predicates, feedback="human")
    PddlGenerator.generate()