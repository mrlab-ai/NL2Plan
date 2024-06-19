import os

from .utils.paths import action_extraction_prompts as prompt_dir
from .utils.logger import Logger
from .utils.human_feedback import human_feedback
from .hierarchy_construction import Hierarchy
from .llm_model import LLM_Chat, get_llm

@Logger.section("3 Action Extraction")
def action_extraction(llm_conn: LLM_Chat, domain_desc_str: str, type_hierarchy: Hierarchy, feedback: str | None = None) -> dict[str, str]:
    """
    Extracts actions from a given domain description using a GPT_Chat language model.

    Args:
        llm_conn (LLM_Chat): The language model.
        domain_desc_str (str): The domain description string.
        type_hierarchy (Hierarchy): The type hierarchy.

    Returns:
        dict[str, str]: A dictionary of extracted actions, where the keys are action names and the values are action descriptions.
    """
    llm_conn.reset_token_usage()

    with open(os.path.join(prompt_dir, "main.txt")) as f:
        act_extr_template = f.read().strip()
    act_extr_prompt = act_extr_template.replace('{domain_desc}', domain_desc_str)
    act_extr_prompt = act_extr_prompt.replace('{type_hierarchy}', str(type_hierarchy))

    with open(os.path.join(prompt_dir, "feedback.txt")) as f:
        feedback_template = f.read().strip()
    feedback_template = feedback_template.replace('{domain_desc}', domain_desc_str)
    feedback_template = feedback_template.replace('{type_hierarchy}', str(type_hierarchy))

    Logger.log("PROMPT:\n", act_extr_prompt)
    llm_output = llm_conn.get_response(prompt=act_extr_prompt)
    Logger.print("LLM Output:\n", llm_output)

    actions = parse_actions(llm_output)

    if feedback is not None:
        if feedback.lower() == "human":
            action_strs = "\n".join([f"- {name}: {desc}" for name, desc in actions.items()])
            feedback_msg = human_feedback(f"\n\nThe actions extracted are:\n{action_strs}\n")
        else:
            feedback_msg = get_llm_feedback(llm_conn, actions, feedback_template)
        if feedback_msg is not None:
            messages = [
                {'role': 'user', 'content': act_extr_prompt},
                {'role': 'assistant', 'content': llm_output},
                {'role': 'user', 'content': feedback_msg}
            ]
            llm_response = llm_conn.get_response(messages=messages)
            Logger.print("LLM Response:\n", llm_response)
            actions = parse_actions(llm_response)

    # Log results
    action_strs = [f"{name}: {desc}" for name, desc in actions.items()]
    Logger.print(f"Extracted {len(actions)} actions: \n - ", "\n - ".join(action_strs))

    in_tokens, out_tokens = llm_conn.token_usage()
    Logger.add_to_info(Action_Extraction_Tokens=(in_tokens, out_tokens))

    return actions

def parse_actions(llm_output: str) -> dict[str, str]:
    """
    Parses the output of the action extraction prompt to extract the action names and descriptions.

    Args:
        llm_output (str): The output of the action extraction prompt.

    Returns:
        dict[str, str]: A dictionary of extracted actions, where the keys are action names and the values are action descriptions.
    """

    splits = llm_output.split("```")
    action_outputs = [splits[i].strip() for i in range(1, len(splits), 2)] # Every other split *should* be an action

    actions = {}
    for action in action_outputs:
        name = action.split("\n")[0].strip()
        desc = action.split("\n", maxsplit=1)[1].strip() # Works even if there is no blank line
        actions[name] = desc

    return actions

def get_llm_feedback(llm_conn: LLM_Chat, actions: dict[str, str], feedback_template: str) -> str | None:
    """
    Gets feedback on the extracted actions.

    Args:
        llm_conn (LLM_Chat): The LLM_Chat language model connection.
        actions (dict[str, str]): A dictionary of extracted actions, where the keys are action names and the values are action descriptions.
        feedback_template (str): The feedback template to use.

    Returns:
        str | None: The feedback on the extracted actions.
    """
    action_str = "\n".join([f"- {name}: {desc}" for name, desc in actions.items()])
    feedback_prompt = feedback_template.replace('{actions}', action_str)

    Logger.log("FEEDBACK PROMPT:\n", feedback_prompt)
    feedback = llm_conn.get_response(prompt=feedback_prompt)

    if "no feedback" in feedback.lower() or len(feedback.strip()) == 0:
        Logger.print("FEEDBACK:\n", "No feedback.")
        Logger.log(feedback)
        return None
    
    Logger.print("FEEDBACK:\n", feedback)
    feedback = "## Feedback" + feedback + "\nStart with a \"## Response\" header, then go through all the actions, even those kept from before, under a \"## Actions\" header as before."
    feedback += "\n\n## Response\n"
    return feedback

if __name__ == "__main__":
    Logger.start("demo")

    llm_gpt = get_llm(engine='gpt-3.5-turbo-0125')
    desc = "The AI agent here is a logistics planner that has to plan to transport packages within the locations in a city through a truck and between cities through an airplane. Also, there is no limit to how many packages a truck or plane can carry (so in theory a truck or plane can carry an infinite number of packages)."
    hier = "- object: object is always root, everything is an object"
    hier += "\n\t- city: The geographical area where the transportation takes place."
    hier += "\n\t- location: The origin and destination points for the packages. Trucks can move between these."    
    hier += "\n\t\t- airport: The only locations planes can move between."
    hier += "\n\t- package: The items that need to be transported from one location to another."
    hier += "\n\t- vehicle: The means of transportation for the packages, such as trucks or planes."
    hier += "\n\t\t- truck: A vehicle used to transport packages."
    hier += "\n\t\t- plane: A vehicle used to transport packages."
    type_hierarchy = Hierarchy(hier)
            
    action_extraction(llm_gpt, desc, hier, feedback=True)