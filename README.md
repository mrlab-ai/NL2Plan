# NL2Plan

This is the code for the paper **NL2Plan: Assistive PDDL Modeling from Minimal Text Descriptions**.

NL2Plan is able to generate PDDL domain and problem files through a six-step process which can then be solved with a classical planner. We also offer a baseline zero-shot PDDL generation approach, PDDL-0.

## Setup
We recommend that you use Docker to run NL2Plan, since it's simpler and compatible with most operating systems. However, if you are using a Linux system or WSL on Windows you can also configure the dependencies manually.

### Using Docker
#### Installing Docker
If you do not have Docker and Docker-Compose installed, install these. For support, you can see the following guides:

- Linux: [installing Docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04) and [installing Docker-Compose](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-22-04). If you have another Linux version than Ubuntu 22.04 see [this more complete documentation](https://docs.docker.com/compose/install/).
- Windows: For windows, we recommend using [WSL](https://ubuntu.com/desktop/wsl) and follow the Linux instructions. If you do not have WSL install, you can use [this guide](https://learn.microsoft.com/en-us/windows/wsl/install). It is also possible to install Docker natively, see [this more complete documentation](https://docs.docker.com/compose/install/).
- Mac: See [this complete documentation](https://docs.docker.com/compose/mac-install/). Note that NL2Plan has not currently been tested on Mac. If you encounter any issues, [contact us](#contact).

#### Building NL2Plan
With Docker and Docker Compose installed, you can build NL2Plan with:
```
sudo docker compose build
```
Though note that due to the large amount of dependencies, this can take a while to build. However, once built, NL2Plan runs quickly.

#### Api Key
NL2Plan requires access to an LLM. The easiest option is to use OpenAI's GPT models. For this, specify your API key in [.env](./.env). You can get an API key from: https://platform.openai.com/account/api-keys

#### User Configuration
By default, NL2Plan will run as a root user. However, this can cause generated files to be annoying to remove/modify due to being created with root privilege. You can change the user in [.env](./.env). There, we also show you how to find your user ID.

If you encounter permission errors when running NL2Plan, there are two things you can try:
1) Delete any existing files in [`results`](./results/) and [`pddl0-results`](./pddl0-results/). If you previously ran NL2Plan as another user, this can cause conflicts. Note that this might require admin privileges, for example `sudo rm` on Linux.
2) Use the default user. Commenting away or removing the line `user: "${UID}:${GID}"` in [docker-compose.yml](./docker-compose.yml) will make Docker use the default root user.

If errors persist, please [contact us](#contact).

### Building without Docker
If you prefer to build everything without Docker, follow the instructions in [BUILD.md](./BUILD.md). Note that these instructions assume you use Linux or WSL on Windows.

## Quickstart
Once the dependencies are installed, you can easily run NL2Plan or PDDL-0 on the existing example domains and tasks. To run these methods on the first Blocksworld task, you can use:

**NL2Plan**
```
sudo docker compose run nl2plan NL2Plan --domain blocksworld --task 1
```

**PDDL-0**
```
sudo docker compose run nl2plan PDDL0 --domain blocksworld --task 1
```

If you're not using Docker, replace `docker-compose run nl2plan` with `python main.py`.

When starting NL2Plan or PDDL-0, a subfolder is created within the corresponding folder:
- NL2Plan: `nl2plan-code/results`
- PDDL-0: `nl2plan-code/pddl0-results`

While running, terminal print-outs will be made showing the current status. Additionally, full logs will be available in the corresponding directory. Upon completion, the files `domain.pddl` and `problem.pddl` will contain the generated domain and problem respectively. If a plan can be found, candidate plans will be saved in `plan.txt.I` files, where `I` is an integer, and higher numbers corresponds to better plans.

## When Running
### NL2Plan
NL2Plan works through 6 steps:
- **Type Extraction**: Extracts types from the description.
- **Type Hierarchy**: Organizes the types into a hierarchy.
- **Action Extraction**: Extracts actions from the description.
- **Action Construction**: Constructs actions and predicates. After this, the entire PDDL domain has been constructed.
- **Problem Extraction**: Extracts objects and initial state from the description.
- **Planning**: Generates a plan using a classical planner, revising the domain and problem files if unsolvable.

Each step includes several LLM-calls, either to perform a job such as deciding on the types or to provide feedback on the choices of the earlier instance.

When you start NL2Plan, a subfolder is created in the [`results`](./results/) directory of the domain for the task. By default, the subfolder is named with the current time. Within this folder, you can see the progress of the NL2Plan process as it generates the files `XXX.log` and the PDDL files. You can open the logs to see the progress live, or you can view the terminal print-outs which contain the most important information.

Once NL2Plan has finished, the generated PDDL files will be saved in `domain.pddl` and `problem.pddl`. If a plan can be found, the candidate plans (if any) will be saved in `plan.txt.I` files, where `I` is an integer, and higher numbers corresponds to better plans. The best plan will also be printed at the end of the terminal print-outs and printed at the end of `6_Planning.log`.

Note that since NL2Plan uses feedback and validators it's expected behavior that the NL2Plan will have to perform the same step multiple times. Additionally, it's expected behavior that you will see error messages in the terminal. If the program doesn't crash, the error messages are likely due to the validators checking the syntax of the generated PDDL files and can be safely ignored. Additionally, while attempting to solve generated PDDL files, the planner might take up to two minutes to find a plan.

If you encounter any issues, please see the [Troubleshooting](#troubleshooting) section below or [contact us](#contact).

### PDDL-0
PDDL-0 generates a PDDL domain and problem files and then iterates them until they pass a validation suite. As such, it's expected behavior that error messages will be shown. If the program doesn't crash, the error messages are likely due to the validators checking the syntax of the generated PDDL files and can be safely ignored.

When you start PDDL-0, a subfolder is created in the [`pddl0-results`](./pddl0-results/) directory of the domain for the task. By default, the subfolder is named with the current time. Within this folder, you can see the progress of the PDDL-0 process as it generates the files `messages.txt` and the PDDL files. You can open the logs to see the progress live, or you can view the terminal print-outs which contain the most important information.

If you encounter any issues, please see the [Troubleshooting](#troubleshooting) section below or [contact us](#contact).

## Default Domains
The currently available domains are:
- [blocksworld](./domains/blocksworld/desc.txt)
- [dimicy](./domains/dimicy/desc.txt)
- [dungeoncrawl](./domains/dungeoncrawl/desc.txt)
- [household](./domains/household/desc.txt)
- [isr](./domains/isr/desc.txt)
- [isr-assisted](./domains/isr-assisted/desc.txt)
- [logistics](./domains/logistics/desc.txt)
- [robility](./domains/robility/desc.txt)
- [rod-rings](./domains/rod-rings/desc.txt)
- [splitfish](./domains/splitfish/desc.txt)
- [tyreworld](./domains/tyreworld/desc.txt)

Most of which have 6 tasks available. You can select a domain with the `--domain` flag and a task with the `--task` flag, see examples above and below.

### Novel Domains
The domains:
- [dimicy](./domains/dimicy/desc.txt)
- [dungeoncrawl](./domains/dungeoncrawl/desc.txt)
- [robility](./domains/robility/desc.txt)
- [rod-rings](./domains/rod-rings/desc.txt)
- [splitfish](./domains/splitfish/desc.txt)

Are novel domains created for this paper. As such, no LLMs have been trained on their PDDL formulations. To keep these domains novel, we do not provide the PDDL formulations in this repository. However, you can still run NL2Plan on the natural language descriptions of the domains and tasks. If you wish to see or use the PDDL formulations, please [contact us](#contact).

## Custom Domains and Tasks
To add a new domain called `MyDomain`, add the folder `MyDomain` as a subfolder of the `domains` folder.

```
mkdir domains/MyDomain
```

Within this folder, create a file called `desc.txt`. This is the file with joint domain information, such as the dynamics of the domain. Add any such information.

```
touch domains/MyDomain/desc.txt
```

Within the same folder, you can also create any number of task files. Each should follow the naming format `taskI.txt` where `I` is an integer. These contain task-specific information. Add any such information.

```
touch domains/MyDomain/task1.txt
touch domains/MyDomain/task2.txt
```

In this way, you can also add new tasks to existing domains.

To run NL2Plan on your domain and task, replace the domain arguments.

```
sudo docker compose run nl2plan NL2Plan --domain MyDomain --task 1
sudo docker compose run nl2plan NL2Plan --domain MyDomain --task 2
```

While solving a task, the requested domain and task files are concatenated to form a joint description, which is then passed to either NL2Plan or PDDL-0.

## Configuration

There are several command line arguments which can be configured when running NL2Plan or PDDL-0.

For example, to run NL2Plan on task 2 of the Blocksworld domain with GPT-3.5 while calling the used subfolder "blocksworld_2", you could use the following command:
```
sudo docker compose run nl2plan NL2Plan --domain blocksworld --task 2 --llm gpt-3.5-turbo --instance_name blocksworld_2
```

Note that the first argument (NL2Plan or PDDL0) is mandatory and chooses between NL2Plan and PDDL-0.

Joint arguments (for both NL2Plan and PDDL-0):
- **domain**: Specifies the domain. All subfolders of the "domains" directory are valid. Default logistics. Example: `--domain blocksworld`.
- **task**: Specifies the task. Integer. Default 1. Example: `--task 1`.
- **llm**: Specifies the LLM to use. You can find the available LLMs at the [OpenAI website](https://platform.openai.com/docs/models). String. Default "gpt-4o". Example: `--llm gpt-4`.
- **instance_name**: Specifies the directory name which results are saved in. If not set, it will use the current time. String. Example `--instance_name blocksworld_1`.

NL2Plan arguments:
- **act_constr_iters**: Specifies the maximum number of complete iterations of the Action Construction step. Integer. Default 2. Example `--act_constr_iters 1`.
- **max_step_4_5_6_attempts**: The maximum number of attempts for the Action Construction (per action), Problem Extraction to correct syntax errors and Planning correct unsolvability. Resets after feedback. Integer. Default 5. Example `--max_step_4_5_6_attempts 2`.
- **max_step_4_attempts**: The maximum number of attempts for the Action Construction (per action) to correct syntax errors. Must be at least 1. Resets after feedback. Overrides "max_step_4_5_6_attempts". Integer. Example `--max_step_4_attempts 2`.
- **max_step_5_attempts**: The maximum number of attempts for the Problem Extraction to correct syntax errors. Must be at least 1. Resets after feedback. Overrides "max_step_4_5_6_attempts". Integer. Example `--max_step_5_attempts 2`.
- **max_step_6_attempts**: The maximum number of attempts for the Planning to correct syntax errors and unsolvability. Resets after feedback. Setting it to 0 disables revisions of Domain and Problem. Overrides "max_step_4_5_6_attempts". Integer. Example `--max_step_6_attempts 2`.
- **start_from**: Specifies from which NL2Plan step the method should start. If set, "start_dir" also must be set. 1 specifies start at the Type Extraction step (the first step) and 5 specifies start at the Problem Extraction step (the last LLM-step). Integer. Default 1. Example: `--start_from 5`.
- **start_dir**: Specifies the path to the directory of the "checkpointI.pkl" file to use when "start_from" is set. Default None. Example: `--start_dir results/blocksworld/2024-...`
- **no_checkpoints**: If set, disables saving of checkpoints. This means that you cannot later use "start_from" and "start_dir" to continue or re-start later. Default not set. Example: `--no_checkpoints`.

PDDL-0 arguments:
- **max_attempts**: The maximum numbers of attempts to correct syntax errors. Integer. Default 5. Example `--max_attempts 2`.
- **disable_validation**: Disables validation and is equivalent to setting "max_attempts" to 1. Default not set. Example `--disable_validation`.

## Prompts
The prompts for NL2Plan are located in the [`NL2Plan/prompts`](./NL2Plan/prompts/) folder. Each step has its own subfolder, and within these are the prompt files used for that step. You can modify these files to change the behavior of NL2Plan.

The prompts for PDDL-0 are located in the [`PDDL0/prompts`](./PDDL0/prompts/) folder. You can modify these files to change the behavior of PDDL-0.

## Troubleshooting
If you see errors when running NL2Plan or PDDL-0, but the program does not crash, this is likely expected behavior. NL2Plan and PDDL-0 involves validators that check the syntax of the generated PDDL files. As such, if the LLM makes any mistakes while coding, these will be shown in the terminal.

If you encounter permission errors when running NL2Plan with Docker, see [### User Configuration](#user-configuration) above.

If you get the error `ValueError: OPENAI_API_KEY is not set.`, this means that NL2Plan can't find your OpenAI API key. If you're using Docker, add your key to [.env](./.env), remember to uncomment the line. If you're running NL2Plan without Docker, you need to export the key in each terminal running NL2Plan, or add it to your bashrc. You can export it using `export OPENAI_API_KEY=sk-123...`, replacing the key with your own.

If you get the error `ConnectionError: Failed to connect to the LLM after 5 retries`, this likely means that your OpenAI API key is incorrect. Please double-check your API-key and reconfigure it as described in the previous paragraph. If the key is correct, this might be a server-side error from OpenAI. If so, try again later. Should the error still persists, please [contact us](#contact).

If the program stalls, this might be expected behavior. After PDDL domain and problem files are generated a planner is used to solve them, which can take up to two minutes. When running NL2Plan, this might occur several times for each task as it attempts to find recommendations for the LLM. If you wish, you can disable revisions of the domain and problem files by setting `--max_step_6_attempts 0` which will reduce the number of planner calls to one per task. If using PDDL-0, only one planner call is made per task.

If the program outputs `The supplied domain/problem file appear to violate part of the PDDL language specification.  Specifically:` this is due to one of the validators used in the validation suite. It's expected behavior and likely does not mean that the domain or problem files are incorrect. If there are actual errors, they will be printed below this message if running NL2Plan or found in the "messages.txt" file if using PDDL-0. Any errors will be revised by the LLM. This message can be safely ignored.

If you get the error `KeyError: 'Could not automatically map {LLM} to a tokeniser.` this is due to the LLM requested, "{LLM}", not being available. Please check the LLMs available on the [OpenAI website](https://platform.openai.com/docs/models) and select one of those. Our recommendations are:
- "gpt-4o"
- "gpt-4o-mini"
- "gpt-4-turbo"
- "gpt-3.5-turbo"
In decreasing order of LLM quality. You can set the LLM with the `--llm` flag. See [### Configuration](#configuration) above.

If you get the error `There was a problem when trying to write in your cache folder (/.cache/huggingface/hub)`, this is due to Docker permissions. You can safely ignore this error.

If you encounter any other issues or they persist, please [contact us](#contact).

## Credit
Parts of this repo, most notably the Action Construction step, are based on code from [Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning](https://github.com/GuanSuns/LLMs-World-Models-for-Planning) by Guan et al.

## Contact
For questions, issues or feedback, please contact us at `elliot.gestrin@liu.se`.

## Cite Us
```
@inproceedings{gestrin-et-al-icaps2024wshaxp,
  author       = {Elliot Gestrin and Marco Kuhlmann and Jendrik Seipp},
  title        = {{NL2Plan}: Robust {LLM}-Driven Planning from Minimal Text Descriptions},
  booktitle    = {ICAPS 2024 Workshop on Human-Aware and Explainable Planning (HAXP)},
  year         = 2024,
}
```