import asyncio
from collections import OrderedDict, defaultdict
import enum
import queue
from abc import ABC, abstractmethod
from base import *
import numpy as np
import zlib
import re
import base64
from datetime import datetime
from typing import Union, Dict, List
from functools import lru_cache, partial
from itertools import chain
from dataclasses import  asdict

import requests
import queue

from keysymdef import keysymdef  # type: ignore
vaild_keysymdef = [x[0] for x in keysymdef]
vaild_keysymdef_lower_map = {x.lower(): x for x in vaild_keysymdef}

from bleu import compute_bleu

@dataclass
class ActionAttributeScore:
    attribute_name: str
    pred:str
    label:str

@dataclass
class ActionValueScore:
    attribute_name: str
    score: float
    metric: str
    pred: str or int = None 
    label: str or int = None 

# similarity 由 ActionAttributeScore 和 ActionValueScore 构成
@dataclass
class ActionSimilarity:
    score_point:int
    scores: List[ActionAttributeScore or ActionValueScore]

    def get_score(self):
        if self.score_point == 0:
            return 0.0
        
        action_score = 0.0
        for score in self.scores:
            if isinstance(score, ActionAttributeScore):
                if score.pred == score.label:
                    action_score += 1.0
            elif isinstance(score, ActionValueScore):
                action_score += score.score

        return action_score


class ActionMeta(type):
    def __new__(cls, name, bases, attrs):
        ordered_save_attrs = []
        for base in bases:
            if hasattr(base, 'save_attributes'):
                ordered_save_attrs.extend(base.save_attributes)

        if 'save_attributes' in attrs:
            for attr in attrs['save_attributes']:
                if attr not in ordered_save_attrs:
                    ordered_save_attrs.append(attr)

        attrs['save_attributes'] = ordered_save_attrs
        return super().__new__(cls, name, bases, attrs)

    def from_json(cls, json_dict):
        action_type = json_dict.get("action_type", None)
        if action_type is None:
            return None
        action_class = globals().get(action_type, None)
        if action_class is None:
            return None
        json_dict.pop("action_type")
        try:
            action = action_class(**json_dict)
        except TypeError:
            return None
        except AttributeError:
            return None
        except IncompleteActionDataError:
            return None
        except KeyError:
            return None
        return action


class Action(metaclass=ActionMeta):
    # MicroAction class
    base_attributes = ["action_time", "before_action_obs", "after_action_obs"]
    save_attributes = ["action_time", "before_action_obs", "after_action_obs"]
    base64_attributes = ["before_action_obs", "after_action_obs"]
    is_required_update = True
    request_id = None

    def __init__(self, action_time=None, before_action_obs=None, after_action_obs=None):
        self.action_time = action_time
        self.before_action_obs = before_action_obs
        self.after_action_obs = after_action_obs

    @property
    def action_type(self):
        return type(self).__name__

    @abstractmethod
    async def step(self, vnc):
        pass

    def save_action(self):
        # return a dict save_attributes
        dic = {}
        for attr in self.save_attributes:
            dic[attr] = getattr(self, attr)
        dic["action_type"] = type(self).__name__
        return dic

    def __str__(self):
        attrs = []
        for attr in self.save_attributes:
            value = getattr(self, attr)
            if attr in ["before_action_obs", "after_action_obs"]:
                if isinstance(value, np.ndarray):
                    attrs.append(f"{attr}=shape{value.shape}")
            elif isinstance(value, str):
                attrs.append(f"{attr}='{value}'")
            else:
                attrs.append(f"{attr}={value}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"
    
    def __repr__(self):
        return self.to_ideal_display_format()

    def to_ideal_dict_format(self):
        dic = OrderedDict()
        dic["action_type"] = type(self).__name__
        for attr in type(self).save_attributes:
            if attr not in self.base_attributes:
                value = getattr(self, attr)
                if value is not None:
                    if isinstance(value, enum.Enum):
                        value = value.name
                    elif isinstance(value, np.ndarray):
                        continue
                    dic[attr] = value
        return dic

    def to_ideal_json_format(self):
        dic = self.to_ideal_dict_format()
        return json.dumps(dic, cls=ObjEncoder, ensure_ascii=False)
    
    def to_ideal_display_format(self):
        attrs = []
        for attr in self.save_attributes:
            if attr not in self.base_attributes:
                value = getattr(self, attr)
                if value is not None:
                    if isinstance(value, enum.Enum):
                        value = value.name
                    elif isinstance(value, np.ndarray):
                        continue
                    attrs.append(str(value))
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    @property
    def score_point(self):
        return 0
    
    @abstractmethod
    def similarity(self, pred_action):
        pass

class MouseActionType(enum.Enum):
    down = 0
    up = 1
    scroll_up = 2
    scroll_down = 3
    move = 4
    drag = 5
    click = 6
    double_click = 7


class MouseAction(Action):
    
    save_attributes = ("mouse_action_type", "mouse_button", "mouse_position", "scroll_repeat", "clickable_area")
    is_required_update = False
    
    def __init__(self, mouse_action_type: MouseActionType = None, mouse_button: VNCMouseButton = None, mouse_position: Union[Position, Dict] = None, scroll_repeat: int = None, clickable_area:ClickableArea=None, **kwargs):
        self.mouse_action_type = mouse_action_type
        if isinstance(mouse_action_type, str):
            self.mouse_action_type = MouseActionType[mouse_action_type.lower()]
        self.mouse_button = mouse_button
        if isinstance(mouse_button, str):
            self.mouse_button = VNCMouseButton[mouse_button.lower()]
        self.mouse_position = mouse_position
        if isinstance(mouse_position, dict):
            self.mouse_position = Position(**mouse_position)
        self.scroll_repeat = scroll_repeat

        if clickable_area is not None and isinstance(clickable_area, dict):
            self.clickable_area = ClickableArea.from_json(clickable_area)
        else:
            self.clickable_area = clickable_area

        super().__init__(**kwargs)

    async def step(self, vnc):
        if self.mouse_action_type == MouseActionType.down and self.mouse_button is not None:
            vnc.mouse.down(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.up and self.mouse_button is not None:
            vnc.mouse.up(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.scroll_up:
            if self.scroll_repeat is None:
                self.scroll_repeat = 1
            vnc.mouse.scroll_up(repeat=self.scroll_repeat)
        elif self.mouse_action_type == MouseActionType.scroll_down:
            if self.scroll_repeat is None:
                self.scroll_repeat = 1
            vnc.mouse.scroll_down(repeat=self.scroll_repeat)
        elif self.mouse_action_type == MouseActionType.move and self.mouse_position is not None:
            vnc.mouse.move(self.mouse_position.width, self.mouse_position.height)
        elif self.mouse_action_type == MouseActionType.drag and self.mouse_position is not None:
            # mouse_position is the end position
            vnc.mouse.down(self.mouse_button.value)
            vnc.mouse.move(self.mouse_position.width, self.mouse_position.height)
            vnc.mouse.up(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.click and self.mouse_position is not None:
            vnc.mouse.move(self.mouse_position.width, self.mouse_position.height)
            vnc.mouse.click(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.click and self.mouse_position is None:
            vnc.mouse.click(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.double_click and self.mouse_position is not None:
            vnc.mouse.move(self.mouse_position.width, self.mouse_position.height)
            vnc.mouse.click(self.mouse_button.value)
            vnc.mouse.click(self.mouse_button.value)
        elif self.mouse_action_type == MouseActionType.double_click and self.mouse_position is None:
            vnc.mouse.click(self.mouse_button.value)
            vnc.mouse.click(self.mouse_button.value)
        else:
            raise IncompleteActionDataError("MouseAction is incomplete: {}".format(self))

    def set_clickable_area(self, area:ClickableArea):
        self.clickable_area = area
        if self.mouse_position is not None and self.clickable_area is not None:
            if self.mouse_position not in self.clickable_area:
                raise MousePositionNotInClickableAreaWarning("MouseAction warning: mouse_position{} not in clickable_area{}".format(self.mouse_position, self.clickable_area))

    @property
    def score_point(self):
        score_point = 2 # 采分点 action_type and mouse_action_type
        # self is label_action
        if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.drag]:
            score_point += 2 # if mouse_button same and if mouse_position in clickable_area
        elif self.mouse_action_type == MouseActionType.move:
            score_point += 1 # mouse_position in clickable_area
        elif self.mouse_action_type == MouseActionType.scroll_up or self.mouse_action_type == MouseActionType.scroll_down:
            score_point += 0 # not care about scroll_repeat

        return score_point
    
    def similarity(self, pred_action: Action):
        scores = [
            ActionAttributeScore("action_type", pred_action.action_type, self.action_type),
        ]

        if isinstance(pred_action, MouseAction):
            scores.append(ActionAttributeScore("mouse_action_type", pred_action.mouse_action_type.name, self.mouse_action_type.name))

            # mouse_button
            if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.drag]:
                if pred_action.mouse_button is not None:
                    scores.append(ActionAttributeScore("mouse_button", pred_action.mouse_button.name, self.mouse_button.name))
                else:
                    scores.append(ActionAttributeScore("mouse_button", "", self.mouse_button.name))

            # mouse_position
            if self.mouse_action_type in [MouseActionType.down, MouseActionType.up, MouseActionType.click, MouseActionType.double_click, MouseActionType.move, MouseActionType.drag]:
                if pred_action.mouse_position is not None:
                    is_mouse_position_in_clickable_area = pred_action.mouse_position in self.clickable_area
                    if is_mouse_position_in_clickable_area:
                        scores.append(ActionValueScore("mouse_position", 1.0, "clickable_area"))
                    else:
                        scores.append(ActionValueScore("mouse_position", 0.0, "clickable_area"))

        return ActionSimilarity(self.score_point, scores)

class KeyboardActionType(enum.Enum):
    down = 0
    up = 1
    press = 2
    text = 3

class KeyboardAction(Action):
    save_attributes = ["keyboard_action_type", "keyboard_key", "keyboard_text"]
    convert_dict = {
        "win": "Super_L",
        "windows": "Super_L",
        "windowskey": "Super_L",
        "windows key": "Super_L",
        "Windows key": "Super_L",
        "winkey": "Super_L",
        "ctrl": "Control_L",
        "alt": "Alt_L",
        "shift": "Shift_L",
        "tab": "Tab",
        "enter": "Return",
        "esc": "Escape",
        "backspace": "BackSpace",
        "delete": "Delete",
        "up": "Up",
        "down": "Down",
        "printscreen": "3270_PrintScreen",
        "prtscn": "3270_PrintScreen"
    }

    use_remote_clipboard = False
    remote_clipboard_host = "localhost"
    remote_clipboard_port = 8001
    remote_clipboard_secret_token = None

    @classmethod
    def set_remote_clipboard(cls, config):
        cls.use_remote_clipboard = config.get("use_remote_clipboard", False)
        cls.remote_clipboard_host = config.get("remote_clipboard_host","localhost")
        cls.remote_clipboard_port = config.get("remote_clipboard_port", 8001)
        cls.remote_clipboard_secret_token = config.get("remote_clipboard_secret_token", None)

    def __init__(self, keyboard_action_type: KeyboardActionType = None, keyboard_key: str = None, keyboard_text: str = None, **kwargs):
        self.keyboard_action_type = keyboard_action_type
        if isinstance(keyboard_action_type, str):
            if keyboard_action_type == "input":
                self.keyboard_action_type = KeyboardActionType.text
            else:
                self.keyboard_action_type = KeyboardActionType[keyboard_action_type]
        if keyboard_action_type is None:
            # 自动判断
            if keyboard_key is not None:
                self.keyboard_action_type = KeyboardActionType.press
            elif keyboard_text is not None:
                self.keyboard_action_type = KeyboardActionType.text
            else:
                raise IncompleteActionDataError("KeyboardAction is incomplete: {}".format(self))

        self.keyboard_key = keyboard_key
        self.keyboard_text = keyboard_text
        if keyboard_text is None and kwargs.get("keyboard_input", None) is not None:
            self.keyboard_text = kwargs.pop("keyboard_input")

        if self.keyboard_action_type in [KeyboardActionType.down, KeyboardActionType.up, KeyboardActionType.press] and isinstance(keyboard_key, str):
            # 检查是否是组合键
            if "+" in self.keyboard_key:
                keyboard_key_list = self.keyboard_key.split("+")
                self.keyboard_key = []
                for key in keyboard_key_list:
                    if key == " ":
                        self.keyboard_key.append("space")
                        continue
                    key = key.strip()
                    if key.lower() in self.convert_dict.keys():
                        self.keyboard_key.append(self.convert_dict[key.lower()])
                    else:
                        self.keyboard_key.append(key.lower())
            else:
                if self.keyboard_key == " ":
                    self.keyboard_key = "space"
                self.keyboard_key = self.keyboard_key.strip()
                if self.keyboard_key.lower() in self.convert_dict.keys():
                    self.keyboard_key = self.convert_dict[self.keyboard_key.lower()]
                elif isinstance(self.keyboard_key, str) and self.keyboard_key.lower() in vaild_keysymdef_lower_map.keys():
                    self.keyboard_key = vaild_keysymdef_lower_map[self.keyboard_key.lower()]
                else:
                    raise IncompleteActionDataError("KeyboardAction is incomplete: {}".format(self))
            # print("Converted keyboard_key", self.keyboard_key)
        elif self.keyboard_action_type == KeyboardActionType.text and isinstance(keyboard_text, str):
            self.keyboard_text = keyboard_text
        super().__init__(**kwargs)

    async def step(self, vnc):
        if self.keyboard_action_type == KeyboardActionType.down and self.keyboard_key is not None:
            vnc.keyboard.down(self.keyboard_key)
        elif self.keyboard_action_type == KeyboardActionType.up and self.keyboard_key is not None:
            vnc.keyboard.up(self.keyboard_key)
        elif self.keyboard_action_type == KeyboardActionType.press and self.keyboard_key is not None:
            if isinstance(self.keyboard_key, list):
                vnc.keyboard.press(*self.keyboard_key)
            else:
                vnc.keyboard.press(self.keyboard_key)
        elif self.keyboard_action_type == KeyboardActionType.text and self.keyboard_text is not None:

            if self.use_remote_clipboard:
                url = f"http://{self.remote_clipboard_host}:{self.remote_clipboard_port}/clipboard"
                data = {
                    "text": self.keyboard_text,
                    "token": self.remote_clipboard_secret_token
                }
                try:
                    r = requests.post(url, json=data)
                    print("remote clipboard server response:", r)
                    if r.status_code == 200:
                        vnc.keyboard.press('Control_L', 'v')
                    else:
                        print("remote clipboard server error:", r.text)
                except Exception as e:
                    print("remote clipboard server error:", e)
            else:
                vnc.keyboard.write(self.keyboard_text)
        else:
            raise IncompleteActionDataError("KeyboardAction is incomplete: {}".format(self))

    @property
    def keys_or_text(self):
        if self.keyboard_key is not None:
            # 如果是组合键，转换成字符串
            if isinstance(self.keyboard_key, list):
                return "+".join(self.keyboard_key)
            else:
                return self.keyboard_key
        elif self.keyboard_text is not None:
            return self.keyboard_text
        else:
            return None

    def to_ideal_json_format(self):
        # 希望LLM输出的良好json格式，不包含save_attributes
        dic = OrderedDict()
        dic["action_type"] = type(self).__name__
        for attr in self.save_attributes:
            if attr == "keyboard_key" and self.keyboard_key is not None:
                # 如果是组合键，转换成字符串
                if isinstance(self.keyboard_key, list):
                    dic[attr] = "+".join(self.keyboard_key)
                else:
                    dic[attr] = self.keyboard_key
            elif attr not in self.base_attributes:
                value = getattr(self, attr)
                if value is not None:
                    if isinstance(value, enum.Enum):
                        value = value.name
                    elif isinstance(value, np.ndarray):
                        continue
                    dic[attr] = value
        # return json.dumps(dic, cls=ObjEncoder, object_pairs_hook=OrderedDict)
        return json.dumps(dic)

    @property
    def score_point(self):
        return 2 # 采分点 action_type and mouse_action_type

    def similarity(self, pred_action: Action):
        scores = [
            ActionAttributeScore("action_type", pred_action.action_type, self.action_type),
        ]
        # 比较keyboard_key 或者 keyboard_text 的 bleu分数
        if isinstance(pred_action, KeyboardAction):
            pred_action_key_or_text = pred_action.keys_or_text
            label_key_or_text = self.keys_or_text
            if pred_action_key_or_text is not None and label_key_or_text is not None:
                if pred_action_key_or_text == label_key_or_text:
                    scores.append(ActionValueScore("keyboard_key_or_text", 1.0, "same_or_bleu", pred_action_key_or_text, label_key_or_text))
                else:
                    score_dict = compute_bleu(translation_corpus=[pred_action_key_or_text], reference_corpus=[[label_key_or_text,],])
                    scores.append(ActionValueScore("keyboard_key_or_text", score_dict[0], "same_or_bleu", pred_action_key_or_text, label_key_or_text))
        return ActionSimilarity(self.score_point, scores)
    
class WaitAction(Action):
    save_attributes = ["wait_time"]

    def __init__(self, wait_time: float = 0.5, **kwargs):
        self.wait_time = wait_time
        super().__init__(**kwargs)

    async def step(self, vnc):
        await asyncio.sleep(self.wait_time)

    @property
    def score_point(self):
        return 0

    def similarity(self, pred_action: Action):
        return ActionSimilarity(self.score_point, [])

class PlanAction(Action):
    # For planner agent
    save_attributes = ["element"]

    def __init__(self, element: str = None, **kwargs):
        self.element = element
        super().__init__(**kwargs)

    @property
    def score_point(self):
        score_point = 2 # 采分点 action_type and mouse_action_type
        return score_point

    def similarity(self, pred_action: Action):
        
        scores = [
            ActionAttributeScore("action_type", pred_action.action_type, self.action_type),
        ]
        # bleu分数
        if isinstance(pred_action, PlanAction):
            if pred_action.element == self.element:
                scores.append(ActionValueScore("element", 1.0, "same_or_bleu", pred_action.element, self.element))
            else:
                score_dict = compute_bleu(translation_corpus = [pred_action.element], reference_corpus=[[self.element,],])
                scores.append(ActionValueScore("element", score_dict[0], "bleu", pred_action.element, self.element))
            
        return ActionSimilarity(self.score_point, scores)
            
class EvaluateSubTaskAction(Action):
    # For evaluator agent
    save_attributes = ["situation", "advice"]  # situation has "sub_task_success", "need_retry" or "need_reformulate"

    def __init__(self, situation=None,  advice=None, **kwargs):
        super().__init__(**kwargs)
        self.situation = situation
        if situation in ["goal_success"]:
            self.situation = "sub_task_success"
        self.advice = advice

    @classmethod
    def check(self, action: Action):
        if isinstance(action, EvaluateSubTaskAction):
            if self.situation == "sub_task_success":
                return True
            elif self.situation in ["need_retry", "need_reformulate"] and self.advice is not None:
                return True
        return False
    

    @property
    def score_point(self):
        score_point = 1 # 采分点 situation
        return score_point

    def similarity(self, pred_action: Action):
        scores = []
        if isinstance(pred_action, EvaluateSubTaskAction):
            pred_situation = pred_action.situation
            if pred_situation == "sub_task_success":
                pred_situation = "sub_task_success"
            else:
                pred_situation = "sub_task_fail"
            label_situation = self.situation
            if label_situation == "sub_task_success":
                label_situation = "sub_task_success"
            else:
                label_situation = "sub_task_fail"
            scores.append(ActionAttributeScore("situation",  pred_situation, label_situation))
        return ActionSimilarity(self.score_point, scores)

class ObjEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Position) or isinstance(obj, ClickableArea):
            return obj.__dict__
        elif issubclass(obj.__class__, Action):
            return obj.save_action()
        elif isinstance(obj, np.ndarray):
            return base64.b64encode(zlib.compress(obj.tobytes())).decode('utf-8')  # TODO 统一压缩格式
        elif isinstance(obj, enum.Enum):
            return obj.value
        elif isinstance(obj, ActionAttributeScore) or isinstance(obj, ActionValueScore) or isinstance(obj, ActionSimilarity):
            return asdict(obj)
        return json.JSONEncoder.default(self, obj)

    def iterencode(self, obj, **kwargs):
        if isinstance(obj, dict):
            obj = OrderedDict(obj)
        return super().iterencode(obj, **kwargs)


def _try_find_json(last_stream_response):
    try:
        last_stream_response = last_stream_response.strip().replace("\\_", '_')# .replace('\\"', '"').replace("\n", "")
        action_json = json.loads(last_stream_response)
        return action_json
    except json.JSONDecodeError as e:
        return None

def _parse_json_to_action(maybe_action_json_str):
    actions = []
    if isinstance(maybe_action_json_str, list):
        for action in maybe_action_json_str:
            actions.append(Action.from_json(action))
    if isinstance(maybe_action_json_str, dict):
        actions.append(Action.from_json(maybe_action_json_str))
    return actions

def parse_action_from_text(last_stream_response):
    actions = []
    one_maybe_json = _try_find_json(last_stream_response)
    if one_maybe_json is not None:
        actions.extend(_parse_json_to_action(one_maybe_json))
    else:
        # search json block
        pattern = r'```json.*?```'
        res = re.findall(pattern, last_stream_response, re.S)
        if len(res) > 0:
            for one_maybe_json in res:
                one_maybe_json = one_maybe_json.replace("```json", "").replace("```", "").strip()
                maybe_action = _try_find_json(one_maybe_json)
                if maybe_action is not None:
                    actions.extend(_parse_json_to_action(maybe_action))
                else:
                    # 有可能是好几行的json，需要按行分开
                    for one_line in one_maybe_json.split("\n"):
                        maybe_action = _try_find_json(one_line)
                        if maybe_action is not None:
                            actions.extend(_parse_json_to_action(maybe_action))

    actions = [action for action in actions if action is not None]
    return actions

def find_non_json_span_from_text(last_stream_response):
    non_json_span = []
    one_maybe_json = _try_find_json(last_stream_response)
    if one_maybe_json is not None:
        return non_json_span
    
    # search json block
    json_block_span = []
    pattern = r'```json.*?```'
    for match in re.finditer(pattern, last_stream_response, re.DOTALL):
        start, end = match.span()
        json_block_span.append((start, end))

    # 按照json block的span，将非json的部分分开
    if len(json_block_span) == 0:
        non_json_span.append((0, len(last_stream_response)))
    else:
        last_end = 0
        for start, end in json_block_span:
            if start > last_end:
                non_json_span.append((last_end, start))
            last_end = end
        if last_end < len(last_stream_response):
            non_json_span.append((last_end, len(last_stream_response)))

    return non_json_span

def split_json_span_and_non_json_span_from_text(last_stream_response):
    json_spans = []
    non_json_spans = []
    one_maybe_json = _try_find_json(last_stream_response)
    if one_maybe_json is not None:
        json_spans.append((0, len(last_stream_response)))
        return json_spans, non_json_spans
    
    # search json block
    pattern = r'```json.*?```'
    for match in re.finditer(pattern, last_stream_response, re.DOTALL):
        start, end = match.span()
        json_spans.append((start, end))

    # 按照json block的span，将非json的部分分开
    if len(json_spans) == 0:
        non_json_spans.append((0, len(last_stream_response)))
    else:
        last_end = 0
        for start, end in json_spans:
            if start > last_end:
                non_json_spans.append((last_end, start))
            last_end = end
        if last_end < len(last_stream_response):
            non_json_spans.append((last_end, len(last_stream_response)))

    return json_spans, non_json_spans

@lru_cache(maxsize=1000)
def generate_alignments(n, m, i=0, j=0):
    # 生成所有可能的保持顺序的对齐方式
    if i == n or j == m:
        return [()]

    alignments = []

    # 匹配当前动作，然后为剩余动作生成对齐
    for suffix in generate_alignments(n, m, i + 1, j + 1):
        alignments.append(((i, j),) + suffix)

    # 跳过一个pred action
    for suffix in generate_alignments(n, m, i + 1, j):
        alignments.append(((i, None),) + suffix)

    # 跳过一个label action
    for suffix in generate_alignments(n, m, i, j + 1):
        alignments.append(((None, j),) + suffix)

    return alignments

def calculate_alignment_score(score_matrix, alignment):
    # 计算特定对齐方式的得分
    score = 0
    for i, j in alignment:
        if i is not None and j is not None:
            score += score_matrix[i][j]
    return score

def calculate_optimal_path(score_matrix):
    n, m = len(score_matrix), len(score_matrix[0])
    best_score = float('-inf')
    best_alignment = None

    for alignment in generate_alignments(n, m):
        score = calculate_alignment_score(score_matrix, alignment)
        if score > best_score:
            best_score = score
            best_alignment = alignment

    best_alignment = [(i, j) for i, j in best_alignment if i is not None and j is not None]
    return best_alignment, best_score

def compare_action_sequences(pred_seq: List[Action], label_seq: List[Action]):
    # Use greedy search to find the best alignment of actions
    score_matrix = np.zeros((len(label_seq), len(pred_seq)))
    info_matrix = [[None for _ in range(len(pred_seq))] for _ in range(len(label_seq))]

    for i in range(len(label_seq)):
        for j in range(len(pred_seq)):
            score_info = label_seq[i].similarity(pred_seq[j])
            info_matrix[i][j] = score_info
            score_matrix[i, j] = score_info.get_score()

    best_alignment, best_alignment_score = calculate_optimal_path(score_matrix)

    all_score_point = 0
    for item in label_seq:
        all_score_point += item.score_point

    best_alignment_similarity_info = []
    for i, j in best_alignment:
        best_alignment_similarity_info.append(info_matrix[i][j])

    if all_score_point == 0:
        print("Warning: all_score_point == 0")
        return float('nan'), best_alignment_similarity_info, best_alignment

    return best_alignment_score / all_score_point, best_alignment_similarity_info, best_alignment

