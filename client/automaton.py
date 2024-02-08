import os
from typing import Callable
from transitions import Machine, State
from jinja2 import Template as JinjaTemplate
from PIL import Image
import uuid
from action import *


class Template(JinjaTemplate):
    def __init__(self, source, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source = source

    def __str__(self):
        return self.source

    def __repr__(self):
        return f"Template({self.source!r})"

class BaseState:
    prompt_template_name = None

    def __init__(self, automaton):
        self.automaton = automaton
        if self.prompt_template_name is not None:
            with open(os.path.join(automaton.prompt_tepmlate_dir, f"{self.prompt_template_name}_{automaton.operation_system}_{automaton.language}.txt"), "r", encoding="utf8") as f:
                prompt_template = f.read()
                self.prompt_template = Template(prompt_template)

    def before(self):
        pass

class Prepare(BaseState):
    prompt_template_name = None

class Planning(BaseState):
    prompt_template_name = "planner_agent"

    def before(self):
        print("[Automaton:Planning] Run Planning")

        if self.automaton.vncwidget is not None:
            self.automaton.vncwidget.automaton_state_changed("planning")

        render_dict = {
            "advice": self.automaton.advice,
            **self.automaton.base_info
        }
        prompt = self.prompt_template.render(render_dict, variable=True)
        self.automaton.current_screen = self.automaton.vncwidget.get_now_screenshot()
        self.automaton.ask_llm(prompt, self.automaton.current_screen, self.after_plan)

    def after_plan(self, actions):
        sub_task_list = []
        for action in actions:
            if isinstance(action, PlanAction):
                sub_task_list.append(action.element)

        if len(sub_task_list) == 0:
            print("[Automaton:Planning] No Sub Task Planning Found")
            if self.automaton.auto_transitions:
                self.automaton.replanning()
        else:
            self.automaton.sub_task_list = sub_task_list
            self.automaton.current_task_index = 0
            self.automaton.current_task = self.automaton.sub_task_list[self.automaton.current_task_index]
            self.automaton.update_sub_task_display()

            print("[Automaton:Planning] After Planning")
            if self.automaton.auto_transitions:
                self.automaton.start_acting()

class Acting(BaseState):
    prompt_template_name = "actor_agent"

    def __init__(self, automaton):
        super().__init__(automaton)
        self.auto_execute_actions = automaton.auto_execute_actions

    def before(self):
        print("[Automaton:Acting] Run Acting, current_task:", self.automaton.current_task)

        if self.automaton.vncwidget is not None:
            self.automaton.vncwidget.automaton_state_changed("acting")

        render_dict = {
            "current_task": self.automaton.current_task,
            "advice": self.automaton.advice,
            "sub_task_list": self.automaton.sub_task_list,
            **self.automaton.base_info
        }
        prompt = self.prompt_template.render(render_dict, variable=True)
        self.automaton.before_action_screen = self.automaton.vncwidget.get_now_screenshot()
        self.automaton.ask_llm(prompt, self.automaton.before_action_screen, self.ask_llm_recall_func)

    def ask_llm_recall_func(self, actions):
        if len(actions) == 0:
            print("[Automaton:Acting] No Action Found")
            if self.automaton.auto_transitions:
                self.automaton.reacting()
        else:
            self.automaton.action_list = actions
            print("[Automaton:Acting] Submit Actions")
            request_id = uuid.uuid4().hex
            for action in self.automaton.action_list:
                action.request_id = request_id
            if self.auto_execute_actions:
                self.automaton.vncwidget.execute_actions(request_id, self.automaton.action_list, self.after_action)

    def after_action(self):
        print("[Automaton:Acting] After Acting")
        if self.automaton.auto_transitions:
            self.automaton.start_evaluating()

class Evaluating(BaseState):
    prompt_template_name = "evaluator_agent"

    def before(self):
        print("[Automaton:Evaluating] Refore Evaluating")

        self.automaton.after_action_screen = self.automaton.vncwidget.get_now_screenshot()

        if self.automaton.vncwidget is not None:
            self.automaton.vncwidget.automaton_state_changed("evaluating")

        render_dict = {
            "current_task": self.automaton.current_task,
            "sub_task_list": self.automaton.sub_task_list,
            **self.automaton.base_info
        }
        prompt = self.prompt_template.render(render_dict, variable=True)
        self.automaton.ask_llm(prompt, self.automaton.after_action_screen, self.after_evaluate)

    def after_evaluate(self, actions, evaluate_update_subtask_index=True):
        found_result = None
        for result in actions:
            if isinstance(result, EvaluateSubTaskAction):
                found_result = result
                break

        self.automaton.advice = None
        if found_result is None:
            print("[Automaton:Evaluating] No EvaluateSubTaskAction Found")
            if self.automaton.auto_transitions:
                self.automaton.reevaluate()

        else:
            if found_result.situation is not None:
                if found_result.situation == "sub_task_success":
                    print("[Automaton:Evaluating] After Evaluating, to next_subtask")
                    if evaluate_update_subtask_index:
                        self.automaton.current_task_index += 1
                        self.automaton.update_sub_task_display()
                    if self.automaton.current_task_index == len(self.automaton.sub_task_list):
                        if self.automaton.auto_transitions:
                            self.automaton.to_finish()
                    else:
                        self.automaton.current_task = self.automaton.sub_task_list[self.automaton.current_task_index]
                        if self.automaton.auto_transitions:
                            self.automaton.next_subtask()
                elif found_result.situation == "need_retry":
                    print("[Automaton:Evaluating] After Evaluating, to need retry this subtask")
                    self.automaton.advice = found_result.advice
                    if self.automaton.auto_transitions:
                        self.automaton.retry_current_subtask()

                elif found_result.situation == "need_reformulate":
                    print("[Automaton:Evaluating] After Evaluating, to need_reformulate")
                    self.automaton.advice = found_result.advice
                    if self.automaton.auto_transitions:
                        self.automaton.need_reformulate()
            else:
                print("[Automaton:Evaluating] After Evaluating, Unreasonable circumstances, reevaluate")
                if self.automaton.auto_transitions:
                    self.automaton.reevaluate()

class Finish(BaseState):
    prompt_template_name = None

    def before(self):
        if self.automaton.vncwidget is not None:
            self.automaton.vncwidget.automaton_state_changed("finish")
        print("Finish Game")


class Automaton:
    def __init__(self, config):

        self.prompt_tepmlate_dir = config['prompt_tepmlate_dir']
        self.language = config.get('language', 'en')
        self.operation_system = config.get('operation_system', 'linux')
        self.auto_transitions = config.get('auto_transitions', True)
        self.auto_execute_actions = config.get('auto_execute_actions', True)

        self.vncwidget = None

        self.prepare = Prepare(self)
        self.planning = Planning(self)
        self.acting = Acting(self)
        self.evaluating = Evaluating(self)
        self.finish = Finish(self)

        states = [
            State(name='prepare'),
            State(name='planning', on_enter=self.planning.before),
            State(name='acting', on_enter=self.acting.before),
            State(name='evaluating', on_enter=self.evaluating.before),
            State(name='finish', on_enter=self.finish.before),
        ]

        self.machine = Machine(model=self, states=states, initial=states[0])

        self.machine.add_transition(trigger='to_prepare', source='*', dest='prepare')
        
        self.machine.add_transition(trigger='start_planning', source='*', dest='planning')
        self.machine.add_transition(trigger='start_acting', source='*', dest='acting')
        self.machine.add_transition(trigger='start_evaluating', source='*', dest='evaluating')

        self.machine.add_transition(trigger='next_subtask', source='evaluating', dest='acting')
        self.machine.add_transition(trigger='retry_current_subtask', source='evaluating', dest='acting')
        self.machine.add_transition(trigger='need_reformulate', source='evaluating', dest='planning')

        self.machine.add_transition(trigger='replanning', source='planning', dest='planning')
        self.machine.add_transition(trigger='reacting', source='acting', dest='acting')
        self.machine.add_transition(trigger='reevaluate', source='evaluating', dest='evaluating')

        self.machine.add_transition(trigger='to_finish', source='*', dest='finish')

        self.sub_task_list = []
        self.current_task_index = 0
        self.current_task = None

        self.before_action_screen = None
        self.action_list = []
        self.after_action_screen = None

        self.advice = None
        self.base_info = {}

    def set_auto_transitions(self, auto_transitions: bool):
        self.auto_transitions = auto_transitions

    def set_auto_execute_actions(self, auto_execute_actions: bool):
        self.auto_execute_actions = auto_execute_actions

    def link_to_vncwidget(self, vncwidget):
        self.vncwidget = vncwidget

    def ask_llm(self, prompt: str, image: Image, ask_llm_recall_func: Callable):
        self.vncwidget.set_send_prompt_display(prompt)
        self.vncwidget.ask_llm_sync(prompt, image, ask_llm_recall_func)

    def start(self, task_prompt: str = None, video_width: int = 640, video_height: int = 480):

        self.base_info = {
            "video_width": video_width,
            "video_height": video_height,
            "task_prompt": task_prompt,
        }
        self.current_screen = self.vncwidget.get_now_screenshot()

        self.sub_task_list = []
        self.current_task_index = 0
        self.current_task = None

        self.before_action_screen = None
        self.action_list = []
        self.after_action_screen = None

        self.advice = None

        self.start_planning()

    def get_state(self):
        return self.state

    def set_state(self, state_name):
        self.state = state_name
        print(f"[Automaton] State Changed to {state_name}")
        if state_name == "prepare":
            self.to_prepare()
        elif state_name == "planning":
            self.start_planning()
        elif state_name == "acting":
            self.start_acting()
        elif state_name == "evaluating":
            self.start_evaluating()
        elif state_name == "finish":
            self.to_finish()

    def set_parse_action(self, actions):
        has_plan = False
        has_evalate = False
        for action in actions:
            if isinstance(action, PlanAction):
                has_plan = True
                break
            if isinstance(action, EvaluateSubTaskAction):
                has_evalate = True
                break

        if has_plan:
            self.planning.after_plan(actions)

        elif has_evalate:
            self.evaluating.after_evaluate(actions, evaluate_update_subtask_index = False)

    def update_sub_task_display(self):
        if self.vncwidget is not None:
            self.vncwidget.update_sub_task_display(self.sub_task_list, self.current_task_index)

    def set_current_task_index(self, current_task_index):
        self.current_task_index = current_task_index
        self.current_task = self.sub_task_list[self.current_task_index]
        self.update_sub_task_display()

    def get_current_task(self):
        return self.current_task