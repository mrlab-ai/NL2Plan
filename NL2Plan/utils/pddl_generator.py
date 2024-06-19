import os

from .logger import Logger
from .paths import results_dir
from .pddl_types import Action

class PddlGeneratorClass:
    def __init__(self):
        self.started = False

    def start(self, experiment = None, domain = None):
        self.started = True

        # Get experiment name if not specified
        if experiment is None:
            if not Logger.started:
                raise FileNotFoundError("Logger not started and no experiment specified. Start logger or specify experiment when starting PDDLGenerator.")
            experiment = Logger.name
        if domain is None:
            if not Logger.started:
                raise FileNotFoundError("Logger not started and no domain specified. Start logger or specify domain when starting PDDLGenerator.")
            domain = Logger.domain
        
        # Initialize files
        os.makedirs(os.path.join(results_dir, experiment), exist_ok=True)
        self.domain_file = os.path.join(results_dir, experiment, "domain.pddl")
        self.problem_file = os.path.join(results_dir, experiment, "problem.pddl")

        # Initialize parts
        self.domain = domain
        self.types = ""
        self.predicates = ""
        self.actions = []
        self.objects = ""
        self.init = ""
        self.goal = ""

    def add_action(self, action: Action):
        if not self.started:
            print("Warning: PDDLGenerator not started. Discarding action.")
            return
        self.actions.append(action)

    def reset_actions(self):
        self.actions = []

    def set_types(self, types: str):
        if not self.started:
            print("Warning: PDDLGenerator not started. Discarding types.")
            return
        self.types = types.strip()
    
    def set_predicates(self, predicates: str):
        if not self.started:
            print("Warning: PDDLGenerator not started. Discarding predicates.")
            return
        self.predicates = predicates

    def set_objects(self, objects: str):
        if not self.started:
            print("Warning: PDDLGenerator not started. Discarding objects.")
            return
        self.objects = objects
    
    def set_init(self, init: str):
        if not self.started:
            print("Warning: PDDLGenerator not started. Discarding init.")
            return
        self.init = init

    def set_goal(self, goal: str):
        if not self.started:
            print("Warning: PDDLGenerator not started. Discarding goal.")
            return
        self.goal = goal

    def generate(self):
        if not self.started:
            raise ValueError("PDDLGenerator not started. Start PDDLGenerator before generating.")
        domain = self.generate_domain(self.domain, self.types, self.predicates, self.actions)
        problem = self.generate_problem(self.domain, self.objects, self.init, self.goal)
        with open(self.domain_file, "w") as f:
            f.write(domain)
        with open(self.problem_file, "w") as f:
            f.write(problem)

    def generate_domain(self, domain: str, types: str, predicates: str, actions: list[Action]):
        if not self.started:
            raise ValueError("PDDLGenerator not started. Start PDDLGenerator before generating domain.")
        
        # Write domain file
        desc = ""
        desc += f"(define (domain {domain})\n"
        desc += self.indent(f"(:requirements\n   :strips :typing :equality :negative-preconditions :disjunctive-preconditions\n   :universal-preconditions :conditional-effects\n)", 1) + "\n\n"
        desc += f"   (:types \n{self.indent(types)}\n   )\n\n"
        desc += f"   (:predicates \n{self.indent(predicates)}\n   )"
        desc += self.action_descs(actions)
        desc += "\n)"
        desc = desc.replace("AND","and").replace("OR","or") # The python PDDL package can't handle capital AND and OR
        return desc
    
    def action_descs(self, actions = None) -> str:
        if actions is None:
            actions = self.actions
        desc = ""
        for action in actions:
            desc += "\n\n" + self.indent(self.action_desc(action),1)
        return desc

    def generate_problem(self, domain: str, objects: str, init: str, goal: str):
        if not self.started:
            raise ValueError("PDDLGenerator not started. Start PDDLGenerator before generating problem.")
        
        # Write problem file
        desc = "(define\n"
        desc += f"   (problem {domain}_problem)\n"
        desc += f"   (:domain {domain})\n\n"
        desc += f"   (:objects \n{self.indent(objects)}\n   )\n\n"
        desc += f"   (:init\n{self.indent(init)}\n   )\n\n"
        desc += f"   (:goal\n{self.indent(goal)}\n   )\n\n"
        desc += ")"
        desc = desc.replace("AND","and").replace("OR","or") # The python PDDL package can't handle capital AND and OR
        return desc

    def indent(self, string: str, level: int = 2):
        return "   " * level + string.replace("\n", f"\n{'   ' * level}")
    
    def action_desc(self, action: Action):
        param_str = "\n".join([f"{name} - {type}" for name, type in action['parameters'].items()]) # name includes ?
        desc  = f"(:action {action['name']}\n"
        desc += f"   :parameters (\n{self.indent(param_str,2)}\n   )\n"
        desc += f"   :precondition\n{self.indent(action['preconditions'],2)}\n"
        desc += f"   :effect\n{self.indent(action['effects'],2)}\n"
        desc +=  ")"
        return desc
    
    def copy(self, other: "PddlGeneratorClass"):
        self.started = other.started
        #self.domain_file = other.domain_file
        #self.problem_file = other.problem_file
        #self.domain = other.domain
        self.types = other.types
        self.predicates = other.predicates
        self.actions = other.actions
        self.objects = other.objects
        self.init = other.init
        self.goal = other.goal
    
PddlGenerator = PddlGeneratorClass()