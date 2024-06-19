from copy import deepcopy

from .utils.pddl_output_utils import parse_params, parse_new_predicates, parse_predicates, read_object_types

class PDDL_Syntax_Validator:
    def __init__(self, obj_hierarchy,
                 error_types=None, messed_output_len=20, unsupported_keywords=None):
        default_error_types = [
            #'messed_output_len',
            'unsupported_keywords',
            'invalid_param_types',
            'invalid_predicate_name',
            'invalid_predicate_format',
            'invalid_predicate_usage',
            #'new_action_creation' # Was not used in experiments
        ]
        default_unsupported = ['forall', 'when', 'exists', 'implies']
        self.error_types = default_error_types if error_types is None else error_types
        self.unsupported_keywords = default_unsupported if unsupported_keywords is None else unsupported_keywords
        self.messed_output_len = messed_output_len
        self.obj_types = [t.lower() for t in obj_hierarchy.types()]
        self.obj_hierarchy = obj_hierarchy

    def perform_validation(self, llm_output, **kwargs):
        for error_type in self.error_types:
            self.check_header_specification(llm_output, **kwargs)
            self.check_keyword_usage(llm_output, **kwargs)
            if error_type == 'messed_output_len':
                validation_info = self.check_messed_output(llm_output, **kwargs)
            elif error_type == 'unsupported_keywords':
                validation_info = self.check_unsupported_keywords(llm_output, **kwargs)
            elif error_type == 'invalid_param_types':
                validation_info = self.check_param_types(llm_output, **kwargs)
            elif error_type == 'invalid_predicate_name':
                validation_info = self.check_predicate_names(llm_output, **kwargs)
            elif error_type == 'invalid_predicate_format':
                validation_info = self.check_predicate_format(llm_output, **kwargs)
            elif error_type == 'invalid_predicate_usage':
                validation_info = self.check_predicate_usage(llm_output, **kwargs)
            elif error_type == 'new_action_creation':
                validation_info = self.check_new_action_creation(llm_output, **kwargs)
            else:
                raise NotImplementedError
            if not validation_info[0]:
                return validation_info
        return True, 'all_validation_pass', None, None
    
    def check_header_specification(self, llm_output, **kwargs):
        """
        This function checks whether the header is correctly specified
        """
        for header in ['Parameters', 'Preconditions', 'Effects', 'New Predicates']:
            if header not in llm_output:
                feedback_message = f'The header `{header}` is missing in the PDDL model. Please include the header `{header}` in the PDDL model.'
                return False, 'header_specification', header, feedback_message
        for header in ['Parameters', 'Preconditions', 'Effects']:
            if llm_output.split(f"{header}")[1].split("##")[0].count('```\n') < 2:
                feedback_message = f'The header `{header}` is missing in the formalised code block. Please include a "```" section in the {header} section.'
                return False, 'header_specification', header, feedback_message
        return True, 'header_specification', None, None

    def check_unsupported_keywords(self, llm_output, **kwargs):
        """
        A simple function to check whether the pddl model uses unsupported logic keywords
        """
        for keyword in self.unsupported_keywords:
            if f'({keyword} ' in llm_output:
                feedback_message = f'The precondition or effect contain the keyword `{keyword}` that is not supported in a standard STRIPS style model. Please express the same logic in a simplified way. You can come up with new predicates if needed (but note that you should use existing predicates as much as possible).'
                return False, 'has_unsupported_keywords', keyword, feedback_message
        return True, 'has_unsupported_keywords', None, None
    
    def check_keyword_usage(self, llm_output, **kwargs):
        if not "Action Effects" in llm_output:
            return True, 'has_unsupported_keywords', None, None
        heading = llm_output.split("Action Effects")[1].split("```\n")[1].strip()
        for keyword in ['forall', 'exists', "if "]:
            if keyword in heading:
                feedback_message = f'The keyword `{keyword}` is not supported in the action effects.'
                return False, 'invalid_effect_keyword', keyword, feedback_message

    def check_messed_output(self, llm_output, **kwargs):
        """
        Though this happens extremely rarely, the LLM (even GPT-4) might generate messed-up outputs (basically
            listing a large number of predicates in preconditions and effects)
        """
        assert 'Preconditions' in llm_output, llm_output
        precond_str = llm_output.split('Preconditions')[1].split('```\n')[1].strip(" `\n:")
        if len(precond_str.split('\n')) > self.messed_output_len:
            feedback_message = f'You seem to have generated an action model with an unusually long list of preconditions. Please include only the relevant preconditions/effects and keep the action model concise.'
            return False, 'messed_output_feedback', None, feedback_message

        return True, 'messed_output_feedback', None, None

    def check_param_types(self, llm_output, **kwargs):
        params_info = parse_params(llm_output, include_internal=True)
        for param_name in params_info:
            param_type = params_info[param_name]
            if param_type not in self.obj_types:
                feedback_message = f'There is an invalid object type `{param_type}` for the parameter {param_name}. If you need to use a new type, you can emulate it with an "is_{{type}} ?o - object" precondition. Please revise the PDDL model to fix this error. '
                return False, 'invalid_object_type', param_name, feedback_message
        return True, 'invalid_object_type', None, None

    def check_predicate_names(self, llm_output, **kwargs):
        curr_predicates = kwargs['curr_predicates']
        curr_pred_dict = {pred['name'].lower(): pred for pred in curr_predicates}
        new_predicates = parse_new_predicates(llm_output)

        # check name clash with obj types
        invalid_preds = list()
        for new_pred in new_predicates:
            if new_pred['name'].lower() in self.obj_types:
                invalid_preds.append(new_pred['name'])
        if len(invalid_preds) > 0:
            feedback_message = f'The following predicate(s) have the same name(s) as existing object types:'
            for pred_i, pred_name in enumerate(list(invalid_preds)):
                feedback_message += f'\n{pred_i + 1}. {pred_name}'
            feedback_message += '\nPlease rename these predicates.'
            return False, 'invalid_predicate_names', None, feedback_message

        # check name clash with existing predicates
        duplicated_predicates = list()
        for new_pred in new_predicates:
            # check if the name is already used
            if new_pred['name'].lower() in curr_pred_dict:
                curr = curr_pred_dict[new_pred['name'].lower()]
                if len(curr['params']) != len(new_pred['params']) or any([t1 != t2 for t1, t2 in zip(curr['params'], new_pred['params'])]):
                    # if the params are the same, then it's not a problem
                    duplicated_predicates.append((new_pred['raw'], curr_pred_dict[new_pred['name'].lower()]['raw']))
        if len(duplicated_predicates) > 0:
            feedback_message = f'The following predicate(s) have the same name(s) as existing predicate(s):'
            for pred_i, duplicated_pred_info in enumerate(duplicated_predicates):
                new_pred_full, existing_pred_full = duplicated_pred_info
                feedback_message += f'\n{pred_i + 1}. {new_pred_full.replace(":", ",")}; existing predicate with the same name: {existing_pred_full.replace(":", ",")}'
            feedback_message += '\n\nYou should reuse existing predicates whenever possible. If you are reusing existing predicate(s), you shouldn\'t list them under \'New Predicates\'. If existing predicates are not enough and you are devising new predicate(s), please use names that are different from existing ones.'
            feedback_message += '\n\nPlease revise the PDDL model to fix this error.\n\n'
            feedback_message += 'Parameters:'
            return False, 'invalid_predicate_names', None, feedback_message

        return True, 'invalid_predicate_names', None, None

    def check_predicate_format(self, llm_output, **kwargs):
        """
        Though this happens rarely, the LLM (even GPT-4) might forget to define the object type of some parameters in new predicates
        """
        new_predicates = parse_new_predicates(llm_output)
        for new_pred in new_predicates:
            new_pred_def = new_pred['raw'].split(': ')[0]
            new_pred_def = new_pred_def.strip(" ()`")   # discard parentheses and similar
            split_predicate = new_pred_def.split(' ')[1:]   # discard the predicate name
            split_predicate = [e for e in split_predicate if e != '']

            for i, p in enumerate(split_predicate):
                if i % 3 == 0:
                    if '?' not in p:
                        feedback_message = f'There are syntax errors in the definition of the new predicate {new_pred_def}. Please revise its definition and output the entire PDDL action model again. Note that you need to strictly follow the syntax of PDDL.'
                        return False, 'invalid_predicate_format', None, feedback_message
                    else:
                        if i + 1 >= len(split_predicate) or split_predicate[i+1] != '-':
                            feedback_message = f'There are syntax errors in the definition of the new predicate {new_pred_def}. Please revise its definition and output the entire PDDL action model again. Note that you need to define the object type of each parameter and strictly follow the syntax of PDDL.'
                            return False, 'invalid_predicate_format', None, feedback_message
                        if i + 2 >= len(split_predicate):
                            feedback_message = f'There are syntax errors in the definition of the new predicate {new_pred_def}. Please revise its definition and output the entire PDDL action model again. Note that you need to define the object type of each parameter and strictly follow the syntax of PDDL.'
                            return False, 'invalid_predicate_format', None, feedback_message
                        param_obj_type = split_predicate[i+2]
                        if param_obj_type not in self.obj_types:
                            feedback_message = f'There is an invalid object type `{param_obj_type}` for the parameter {p} in the definition of the new predicate {new_pred_def}. Please revise its definition and output the entire PDDL action model again.'
                            return False, 'invalid_predicate_format', None, feedback_message
        return True, 'invalid_predicate_format', None, None
    
    def check_new_action_creation(self, llm_output, **kwargs):
        """
        This action checks if the LLM attempts to create a new action (so two or more actions defined in the same response)
        """
        if llm_output.count('## Action Parameters') > 1 or llm_output.count('## Preconditions') > 1 or llm_output.count('## Effects') > 1 or llm_output.count('## New Predicates') > 1:
            # Note that the '##' check also covers the case with three #s for the headings
            feedback_message = "It's not possible to create new actions at this time. Please only define the requested action."
            return False, 'new_action_creation', None, feedback_message
        return True, 'new_action_creation', None, None

    def _is_valid_type(self, target_type, curr_type):
        return self.obj_hierarchy.is_subtype(curr_type, target_type)

    def _check_predicate_usage_pddl(self, pddl_snippet, predicate_list, action_params, part='preconditions'):
        """
        This function checks three types of errors:
            - check if the num of params given matches the num of params in predicate definition
            - check if there is any param that is not listed under `Parameters:`
            - check if the param type matches that in the predicate definition
        """
        def get_ordinal_suffix(_num):
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(_num % 10, 'th') if _num not in (11, 12, 13) else 'th'

        pred_names = {predicate_list[i]['name']: i for i in range(len(predicate_list))}
        pddl_elems = [e for e in pddl_snippet.split(' ') if e != '']
        idx = 0
        while idx < len(pddl_elems):
            if pddl_elems[idx] == '(' and idx + 1 < len(pddl_elems):
                if pddl_elems[idx + 1] in pred_names:
                    curr_pred_name = pddl_elems[idx + 1]
                    curr_pred_params = list()
                    target_pred_info = predicate_list[pred_names[curr_pred_name]]
                    # read params
                    idx += 2
                    while idx < len(pddl_elems) and pddl_elems[idx] != ')':
                        curr_pred_params.append(pddl_elems[idx])
                        idx += 1
                    # check if the num of params are correct
                    n_expected_param = len(target_pred_info['params'])
                    if n_expected_param != len(curr_pred_params):
                        feedback_message = f'In the {part}, the predicate `{curr_pred_name}` requires {n_expected_param} parameters but {len(curr_pred_params)} parameters were provided. Please revise the PDDL model to fix this error.'
                        return False, 'invalid_predicate_usage', None, feedback_message
                    # check if there is any unknown param
                    for curr_param in curr_pred_params:
                        if curr_param not in action_params:
                            feedback_message = f'In the {part} and in the predicate `{curr_pred_name}`, there is an unknown parameter `{curr_param}`. You should define all parameters (i.e., name and type) under the `Parameters` list. Please revise the PDDL model to fix this error (and other potentially similar errors).'
                            return False, 'invalid_predicate_usage', None, feedback_message
                    # check if the object types are correct
                    target_param_types = [target_pred_info['params'][t_p] for t_p in target_pred_info['params']]
                    for param_idx, target_type in enumerate(target_param_types):
                        curr_param = curr_pred_params[param_idx]
                        claimed_type = action_params[curr_param]

                        if not self._is_valid_type(target_type, claimed_type):
                            feedback_message = f'There is a syntax error in the {part.lower()}, the {param_idx+1}-{get_ordinal_suffix(param_idx+1)} parameter of `{curr_pred_name}` should be a `{target_type}` but a `{claimed_type}` was given. Please use the correct predicate or devise new one(s) if needed (but note that you should use existing predicates as much as possible).'
                            return False, 'invalid_predicate_usage', None, feedback_message
            idx += 1
        return True, 'invalid_predicate_usage', None, None

    def check_predicate_usage(self, llm_output, **kwargs):
        """
        This function performs very basic check over whether the predicates are used in a valid way.
            This check should be performed at the end.
        """
        # parse predicates
        new_predicates = parse_new_predicates(llm_output)
        curr_predicates = deepcopy(kwargs['curr_predicates'])
        curr_predicates.extend(new_predicates)
        curr_predicates = parse_predicates(curr_predicates)

        # get action params
        params_info = parse_params(llm_output, include_internal=True)

        # check preconditions
        precond_str = llm_output.split('Preconditions')[1].split('```\n')[1].strip()
        precond_str = precond_str.replace('\n', ' ').replace('(', ' ( ').replace(')', ' ) ')
        validation_info = self._check_predicate_usage_pddl(precond_str, curr_predicates, params_info, part='preconditions')
        if not validation_info[0]:
            return validation_info

        if llm_output.split('Effects')[1].count('```\n') < 2:
            # no effects, probably. 
            return True, 'invalid_predicate_usage', None, None
        eff_str = llm_output.split('Effects')[1].split('```\n')[1].strip()
        eff_str = eff_str.replace('\n', ' ').replace('(', ' ( ').replace(')', ' ) ')
        return self._check_predicate_usage_pddl(eff_str, curr_predicates, params_info, part='effects')


def main():
    kwargs = {
        'curr_predicates': list()
    }
    obj_hierarchy = {
        "furnitureAppliance": [],
        "householdObject": ["smallReceptacle"]
    }

    pddl_snippet = """
Apologies for the confusion. Since the predicates are already defined, I will not list them under 'New Predicates'. Here is the revised PDDL model.

Parameters:
1. ?x - householdObject: the object to put in/on the furniture or appliance
2. ?y - furnitureAppliance: the furniture or appliance to put the object in/on

Preconditions:
```
(and
    (robot-at ?y)
    (robot-holding ?x)
    (pickupable ?x)
    (object-clear ?x)
    (or
        (not (openable ?y))
        (opened ?y)
    )
)
```

Effects:
```
(and
    (not (robot-holding ?x))
    (robot-hand-empty)
    (object-on ?x ?y)
    (if (openable ?y) (closed ?y))
)
```

New Predicates:
1. (closed ?y - furnitureAppliance): true if the furniture or appliance ?y is closed
2. (openable ?y - householdObject): true if the furniture or appliance ?y can be opened
3. (furnitureappliance ?x - furnitureAppliance): true if xxxxxxxxx
    """

    pddl_validator = PDDL_Syntax_Validator(obj_hierarchy)
    print(pddl_validator.check_unsupported_keywords(pddl_snippet, **kwargs))
    print(pddl_validator.check_messed_output(pddl_snippet, **kwargs))
    print(pddl_validator.check_param_types(pddl_snippet, **kwargs))
    print(pddl_validator.check_predicate_names(pddl_snippet, **kwargs))
    print(pddl_validator.check_predicate_format(pddl_snippet, **kwargs))
    print(pddl_validator.check_predicate_usage(pddl_snippet, **kwargs))
    # print(pddl_validator.perform_validation(pddl_snippet, **kwargs))


if __name__ == '__main__':
    main()
