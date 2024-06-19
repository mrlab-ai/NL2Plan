import os, datetime

from .paths import results_dir, prompt_dir
from NL2Plan.llm_model import get_llm

def main(domain: str, domain_task_desc: str, llm: str = "gpt-4-1106-preview"):
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(os.path.join(prompt_dir, "prompt.txt"), "r") as file:
        prompt = file.read()
    prompt = prompt.replace("{domain_task_desc}", domain_task_desc)

    # Load the LLM
    llm = get_llm(llm)
    llm.reset_token_usage()

    # Generate the plan
    resp = llm.get_response(prompt=prompt)
    print(resp)
    print("-"*100)

    # Save the plan and prompt
    dir = os.path.join(results_dir, "CoT", domain, time)
    os.makedirs(dir, exist_ok=True)
    with open(os.path.join(dir, "resp.txt"), "w") as file:
        file.write(resp)
    with open(os.path.join(dir, "prompt.txt"), "w") as file:
        file.write(prompt)

    # Save the token usage
    in_tokens, out_tokens = llm.token_usage()
    with open(os.path.join(dir, "token_usage.txt"), "w") as file:
        file.write(f"Input tokens: {in_tokens}\nOutput tokens: {out_tokens}")

    return resp