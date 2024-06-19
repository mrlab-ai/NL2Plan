# NL2Plan

This is the code for the paper **NL2Plan: Robust LLM-Driven Planning from Minimal Text Descriptions** which can be found [here](https://arxiv.org/abs/2405.04215). 

NL2Plan is an offline, domain-agnostic planning system able to produce plans for tasks described in natural language through use of an LLM. It does this via creating an intermediate [PDDL](https://planning.wiki/guide/whatis/pddl) representation of the task, which is then solved with a classical planner. This uses the LLM for its strength, general world knowledge and textual understanding, and the planner for long term decision making.

This repo also offers a baseline zero-shot [chain-of-thought](https://arxiv.org/abs/2201.11903) method. This plans directly with an LLM in a single step.

## Setup
The setup is only tested on Ubuntu 22.04, but should work on other Linux machines. Installation for Windows or Mac users might require further adapations. 

### Python Environment
The repo has primarily been tested for Python 3.10 and 3.11. 

You can set up  a Python environment using either [Conda](https://conda.io) or [venv](https://docs.python.org/3/library/venv.html) and install the dependencies via the following steps.

**Conda**
```
conda create -n NL2Plan python=3.10
conda activate NL2Plan
pip install -r requirements.txt
```

**venv**
```
python3.10 -m venv env
source env/bin/activate
pip install -r requirements.txt
``` 

These environments can then be exited with `conda deactivate` and `deactivate` respectively. The instructions below assume that a suitable environemnt is active. 

### Scorpion Planner
NL2Plan uses the [Scorpion](https://github.com/jendrikseipp/scorpion) planning system. To set it up, run the following from the root of this repo. 

```
# Install dependencies
sudo apt install cmake g++ git make python3
# Pull the repo
git clone https://github.com/jendrikseipp/scorpion.git
# Build Scorpion
./scorpion/build.py
```

### API Keys
NL2Plan requires access to an LLM. It supports either OpenAI:s GPT-models, or any self-hosted Ollama model. To configure these, provide the necessary API-key or adress in an environement variable. You only need to configure those which you will use.

**OpenAI**
```
export OPENAI_API_KEY='YOUR-KEY' # e.g. OPENAI_API_KEY='sk-123456'
``` 

**Ollama**
``` 
export OLLAMA_URL='YOUR-URL' # e.g. OLLAMA_URL='http://localhost:11434/'
```

## Quickstart
You can easily run NL2Plan or Zero-Shot CoT on the existing example domains and tasks. To run these on Blocksworld, you can use:

**NL2Plan**
``` 
python main.py NL2Plan --domain blocksworld --task 1
```

**Zero-Shot CoT**
``` 
python main.py CoT --domain blocksworld --task 1
```

The results will be printed in the terminal and stored in the `results` directory. Note that the `results` directory also contains some further logs which aren't printed.

The available example domains are:
- blocksworld
- household
- isr
- isr-assisted
- logistics
- tyreworld

With the exception of logistics, each has three tasks of increasing difficulty. Logistics has only one example task.

## Custom Domains and Tasks
To add a new domain called `MyDomain` add the folder `MyDomain` as a subfolder of the `domains` folder.

```
mkdir domains/MyDomain
```

Within this folder, create a file called `desc.txt`. This is the file with joint domain information. Add any joint information here.

```
touch domains/MyDomain/desc.txt
```

Within the same folder, you can also create any numer of task files. Each should follow the naming format `taskI.txt` where `I` is an integer. These contain task-specific information. Add any such information.

```
touch domains/MyDomain/task1.txt
touch domains/MyDomain/task2.txt
``` 

In this way, you can also add new tasks to existing domains. 

To run NL2Plan on your domain and task, replace the domain arguments.

```
python main.py NL2Plan --domain MyDomain --task 1
python main.py NL2Plan --domain MyDomain --task 2
```

While solving a task, the requested domain and task files are concatenated to form a joint description, which is then passed to either NL2Plan or Zero-Shot CoT.

## Configuration

There are several command line arguments which can be configured when running NL2Plan or Zero-Shot CoT. 

For example, to run NL2Plan on task 2 of the Blocksworld domain, with GPT-3.5, no feedback steps and only at most 2 attempts for Action Construction and Task Extraction, run the following:
```
python main.py NL2Plan --domain blocksworld --task 2 --llm gpt-3.5-turbo-1106 --feedback none --max_attempts 2
``` 

Note that the first argument (NL2Plan or CoT) is mandatory and chooses between NL2Plan and Zero-Shot CoT solver.

Joint arguments (for both NL2Plan and Zero-Shot CoT):
- **domain**: Specifies the domain. All subfolders of the "domains" directory are valid. Default logistics. Example: `--domain blocksworld`. 
- **task**: Specifies the task. Integer. Default 1. Example: `--task 1`. 
- **llm**: Specifies the LLM to use. Default "gpt-4-1106-preview". Example: `--llm gpt-4`. 

NL2Plan arguments:
- **feedback**: Which feedback source to use. "llm", "human" or "none" (disables feedback). Default "llm". Example: `--feedback human`. 
- **act_constr_iters**: Specifies the number of complete iterations of the Action Construction step. Integer. Default 2. Example `--act_constr_iters 1`.
- **full_history**: If set, provides the LLM with the full message history during Action Construction and Task Extraction steps. If not set, only the example, last solution and feedback are given. Boolean. Default not set. Example: `--full_history`. 
- **max_attempts**: The maximum number of attempts for the Action Construction (per action) and Task Extraction steps. It re-tries upon receiving feedback. Integer. Default 8. Example `--max_attempts 2`.
- **mute_solver**: If set, the internal solver information will not be displayed during the Planning step. Boolean. Default not set. Example: `--mute_solver`. 
- **start_from**: Specifies from which NL2Plan step the method should start. If set, "start_dir" also must be set. 1 specifies start at the Type Extraction step (the first step) and 5 specifies start at the Task Extraction step (the last LLM-step). Integer. Default 1. Example: `--start_from 5`. 
- **start_dir**: Specifies the path to the directory of the "checkpointI.pkl" file to use when "start_from" is set. Default None. Example: `--start_dir results/blocksworld/2024-...`
- **no_checkpoints**: If set, disables saving of checkpoints. This means that you cannot later use "start_from" and "start_dir" to continue or re-start later. Default not set. Example: `--no_checkpoints`.  

## Parsing Errors
Each LLM has different tendencies in their responses. As such, swapping to a new LLM might lead to parsing issues.

If you encounter this you can try one of the previously tested LLMs (primarily GPT-4 and GPT-4-1106-preview), try to modify the parsing code, or [contact us](#contact) for support.  

## Credit
Parts of this repo, most notably the Action Construction step, are based on code from [Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning](https://github.com/GuanSuns/LLMs-World-Models-for-Planning) by Guan et al.

## Contact
Please contact `elliot.gestrin@liu.se` for questions, comments or feedback about the NL2Plan project. 