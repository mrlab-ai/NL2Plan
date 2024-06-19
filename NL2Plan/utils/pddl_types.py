from typing import TypedDict, NewType
from collections import OrderedDict

ParameterList = NewType('ParameterList', OrderedDict[str, str]) # {param_name: param_type}
ObjectList = NewType('ObjectList', dict[str, str]) # {obj_name: obj_type}

class Predicate(TypedDict):
    name: str # name of the predicate
    desc: str # description of the predicate
    raw: str # raw LLM output
    params: ParameterList # parameters of the predicate
    clean: str # clean version of the predicate, with comment

class Action(TypedDict):
    name: str
    desc: str
    raw: str
    parameters: ParameterList
    preconditions: str
    effects: str