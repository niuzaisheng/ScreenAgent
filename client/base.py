import json
import enum
from dataclasses import dataclass
from key_map import qt2keysymdef_key_mapping

from PyQt5.QtCore import Qt

@dataclass
class Position:
    width: int
    height: int

    def __str__(self):
        return "Position({},{})".format(self.width, self.height)

@dataclass
class ClickableArea:
    upper_left_position: Position
    lower_right_position: Position

    def __post_init__(self):
        if self.upper_left_position.width > self.lower_right_position.width or self.upper_left_position.height > self.lower_right_position.height:
            raise Exception("upper_left_position should be smaller than lower_right_position")

    def __str__(self):
        return "ClickableArea({},{},{},{})".format(self.upper_left_position.width, self.upper_left_position.height, self.lower_right_position.width, self.lower_right_position.height)

    def __contains__(self, position: Position):
        return self.upper_left_position.width <= position.width <= self.lower_right_position.width and self.upper_left_position.height <= position.height <= self.lower_right_position.height

    def from_json(json_dict):
        upper_left_position = Position(json_dict["upper_left_position"]["width"], json_dict["upper_left_position"]["height"])
        lower_right_position = Position(json_dict["lower_right_position"]["width"], json_dict["lower_right_position"]["height"])
        return ClickableArea(upper_left_position, lower_right_position)

    def get_center_position(self):
        center_width = (self.upper_left_position.width + self.lower_right_position.width) // 2
        center_height = (self.upper_left_position.height + self.lower_right_position.height) // 2
        return Position(center_width, center_height)

class VNCMouseButton(enum.Enum):
    left = 0
    middle = 1
    right = 2

class PygameMouseButton(enum.Enum):
    left = 1
    middle = 2
    right = 3

def convert_mouse_button(pygame_mouse_button: PygameMouseButton):
    if pygame_mouse_button == PygameMouseButton.left.value:
        return VNCMouseButton.left
    elif pygame_mouse_button == PygameMouseButton.middle.value:
        return VNCMouseButton.middle
    elif pygame_mouse_button == PygameMouseButton.right.value:
        return VNCMouseButton.right
    else:
        raise Exception("Unknown pygame mouse button")

def convert_mouse_button_qt(qt_mouse_button):
    if qt_mouse_button == Qt.LeftButton:
        return VNCMouseButton.left
    elif qt_mouse_button == Qt.MidButton:
        return VNCMouseButton.middle
    elif qt_mouse_button == Qt.RightButton:
        return VNCMouseButton.right
    else:
        raise Exception("Unknown pygame mouse button")

def convert_qt2keysymdef_key_mapping(qt_key):
    return qt2keysymdef_key_mapping[qt_key]

class IncompleteActionDataError(Exception):
    pass

class MousePositionNotInClickableAreaWarning(Exception):
    pass
