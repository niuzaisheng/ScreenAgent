
import uuid
import queue

from asyncio import open_connection
from functools import partial

import numpy as np
from PIL import Image

from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, QMetaObject, Q_ARG, pyqtSlot
from PyQt5.QtWidgets import QWidget, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QTextEdit, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor, QPalette
from qasync import asyncSlot, asyncClose

from asyncvnc import Client
from base import *
from action import *

class VNCFrame(QLabel):
    def __init__(self, parent, action_queue):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.action_queue = action_queue  # Send action to the vnc server

        self.is_in_focus = False
        self.out_focus_style()

    def in_focus_style(self):
        palette = self.palette()
        palette.setColor(QPalette.WindowText, QColor(Qt.green))
        self.setPalette(palette)
        self.setStyleSheet("border: 5px solid green;")

    def out_focus_style(self):
        palette = self.palette()
        palette.setColor(QPalette.WindowText, QColor(Qt.black))
        self.setPalette(palette)
        self.setStyleSheet("border: 5px solid red;")

    def mousePressEvent(self, event):
        if self.is_in_focus:
            cursor_pos = self.get_local_cursor_pos()
            if cursor_pos is not None:
                action = MouseAction(mouse_action_type=MouseActionType.down, mouse_button=convert_mouse_button_qt(event.button()), mouse_position=cursor_pos)
                self.action_queue.put(action)

    def mouseMoveEvent(self, event):
        if self.get_local_cursor_pos():
            self.is_in_focus = True
        else:
            self.is_in_focus = False

        if self.is_in_focus:
            if cursor_pos:=self.get_local_cursor_pos():
                action = MouseAction(mouse_action_type=MouseActionType.move, mouse_position=cursor_pos)
                self.action_queue.put(action)

    def mouseReleaseEvent(self, event):
        if self.is_in_focus:
            cursor_pos = self.get_local_cursor_pos()
            if cursor_pos is not None:
                action = MouseAction(mouse_action_type=MouseActionType.up, mouse_button=convert_mouse_button_qt(
                    event.button()), mouse_position=cursor_pos)
                self.action_queue.put(action)

    def wheelEvent(self, event):
        if self.is_in_focus:
            scroll_repeat = int(event.angleDelta().y() / 120)
            if scroll_repeat > 0:
                mouse_action_type = MouseActionType.scroll_up
            elif scroll_repeat < 0:
                mouse_action_type = MouseActionType.scroll_down
            else:
                return
            action = MouseAction(
                mouse_action_type=mouse_action_type, scroll_repeat=abs(scroll_repeat))
            self.action_queue.put(action)

    def keyPressEvent(self, event):
        if self.is_in_focus:
            keyboard_key = convert_qt2keysymdef_key_mapping(event.key())
            action = KeyboardAction(
                keyboard_action_type=KeyboardActionType.down, keyboard_key=keyboard_key)
            self.action_queue.put(action)

    def keyReleaseEvent(self, event):
        if self.is_in_focus:
            keyboard_key = convert_qt2keysymdef_key_mapping(event.key())
            action = KeyboardAction(
                keyboard_action_type=KeyboardActionType.up, keyboard_key=keyboard_key)
            self.action_queue.put(action)

    def get_local_cursor_pos(self):
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        if cursor_pos.x() < 0 or cursor_pos.y() < 0 or cursor_pos.x() > self.width() or cursor_pos.y() > self.height():
            return None
        else:
            cursor_pos = Position(cursor_pos.x(), cursor_pos.y())
            return cursor_pos

    def update_screen(self, qimage):
        self.setPixmap(QPixmap.fromImage(qimage))

class VNCWidget(QMainWindow):

    def __init__(self, remote_vnc_server_config, llm_client, automaton):
        super().__init__()

        self.remote_vnc_server_config = remote_vnc_server_config

        self.task_list_path = remote_vnc_server_config["task_list"]

        self.action_queue = queue.Queue()
        self.task_prompt_list = []

        self.video_width = 640
        self.video_height = 480

        self.llm_client_gpt = llm_client

        # 执行完之后任务回调位置记录
        self._request_recall_func_cache = {} # 包含ask llm的request和excuting action的request
        self._request_action_num_counter = {} # 包含excuting action的计数

        self.system_prompt_template_name = ""
        self.system_prompt = ""
        self.task_prompt = ""
        self.send_prompt = ""
        self.last_message = ""

        # link automaton to vncwidget
        self.automaton = automaton
        automaton.link_to_vncwidget(self)

        # refresh screen timer
        # Create a QTimer to limit sampling rate
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(1)
        self.refresh_timer.timeout.connect(self.render)
        self.refresh_timer.stop()
        self.refreshing_screen=False # need for refresh flag

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setMouseTracking(True)

        main_layout = QHBoxLayout(central_widget)
        self.main_layout = main_layout

        # 第一栏 left layout
        left_layout = QVBoxLayout()
        self.left_layout = left_layout

        left_layout.addWidget(QLabel("Connect Control"))
        reconnect_button = QPushButton("Re-connect")
        reconnect_button.clicked.connect(self.reconnect)
        left_layout.addWidget(reconnect_button)

        # left layout - Prompt Selection List
        left_layout.addWidget(QLabel("Task Selection"))
        self.task_prompt_selection = QListWidget(self)
        # double click to select
        self.task_prompt_selection.itemDoubleClicked.connect(self.select_task_prompt)
        left_layout.addWidget(self.task_prompt_selection)

        self.load_prompts_from_file()
        reload_button = QPushButton("Reload Task List")
        reload_button.clicked.connect(self.load_prompts_from_file)
        left_layout.addWidget(reload_button)

        left_layout.addWidget(QLabel("Automaton Control"))

        self.start_automaton_button = QPushButton("Start Automaton")
        self.start_automaton_button.clicked.connect(self.start_automaton)
        left_layout.addWidget(self.start_automaton_button)

        self.automaton_control_buttons = []
        for state in self.automaton.machine.states:
            button = QPushButton(state)
            button.clicked.connect(partial(self.set_automaton_state, state))
            self.automaton_control_buttons.append(button)
            left_layout.addWidget(button)

        self.set_auto_transitions_checkbox = QCheckBox("Enable Auto Transitions")
        self.set_auto_transitions_checkbox.setChecked(True)
        def set_auto_transitions():
            self.automaton.set_auto_transitions(self.set_auto_transitions_checkbox.isChecked())
        self.set_auto_transitions_checkbox.stateChanged.connect(set_auto_transitions)
        left_layout.addWidget(self.set_auto_transitions_checkbox)

        left_layout.addWidget(QLabel("Sub Tasks"))
        self.sub_task_display = QListWidget(self)
        left_layout.addWidget(self.sub_task_display)
        self.sub_task_display.itemDoubleClicked.connect(self.set_current_task_index)

        main_layout.addLayout(left_layout)

        # 第二栏 Middle layout
        middle_layout = QVBoxLayout()
        self.middle_layout = middle_layout

        middle_layout.addWidget(QLabel(f"Task Prompt", self))
        self.task_prompt_display = QTextEdit(self)
        self.task_prompt_display.setReadOnly(False)
        self.task_prompt_display.setPlainText("Please first select a task prompt from the task selection")
        # Set a fixed height for the QTextEdit
        self.task_prompt_display.setFixedHeight(30)
        middle_layout.addWidget(self.task_prompt_display)

        middle_layout.addWidget(QLabel(f"Send Prompt", self))
        self.send_prompt_display = QTextEdit(self)
        self.send_prompt_display.setReadOnly(False)
        middle_layout.addWidget(self.send_prompt_display)

        self.vnc_frame = VNCFrame(self, self.action_queue)
        self.vnc_frame.setGeometry(0, 0, self.video_width, self.video_height)
        self.middle_layout.addWidget(self.vnc_frame)

        self.main_layout.addLayout(middle_layout)

        # Right layout
        right_layout = QVBoxLayout()

        right_layout.addWidget(QLabel("VLM Response and Action Display"))

        self.LLM_response = ""
        self.send_image = None
        self.parse_action_list = []

        clear_LLM = QPushButton("Clear All")
        clear_LLM.clicked.connect(self.reset)
        right_layout.addWidget(clear_LLM)

        right_layout.addWidget(QLabel("VLM Original Response (read only)"))
        self.LLM_response_display = QTextEdit(self)
        self.LLM_response_display.setReadOnly(True)
        self.LLM_response_display.setFixedHeight(100)
        right_layout.addWidget(self.LLM_response_display)

        # LLM Response Editer
        right_layout.addWidget(QLabel("VLM Response Editor"))

        self.LLM_response_editer = QTextEdit(self)
        self.LLM_response_editer.setReadOnly(False)
        self.LLM_response_editer.setFixedHeight(300)
        right_layout.addWidget(self.LLM_response_editer)

        parse_action_button = QPushButton("Try to parse actions in editor")

        def parse_action_from_LLM_Response_TextEdit():
            self.parse_action_list.clear()
            self.parse_action_display.clear()
            self._parse_action_display_action_map.clear()
            actions = parse_action_from_text(self.LLM_response_editer.toPlainText())
            for action in actions:
                self.parse_action_list.append(action)
                self.parse_action_display.addItem(action.to_ideal_display_format())
                self._parse_action_display_action_map[action] = self.parse_action_display.count() - 1

            if self.automaton is not None:
                self.automaton.set_parse_action(actions)

        parse_action_button.clicked.connect(parse_action_from_LLM_Response_TextEdit)
        right_layout.addWidget(parse_action_button)

        right_layout.addWidget(QLabel("Parsed Actions"))
        self.parse_action_display = QListWidget(self)
        self.parse_action_display.setWordWrap(True)
        right_layout.addWidget(self.parse_action_display)
        self._parse_action_display_action_map = {} # action to parse_action_display item index

        self.set_auto_execute_actions_checkbox = QCheckBox("Enable Auto Execute Actions")
        self.set_auto_execute_actions_checkbox.setChecked(True)
        def set_auto_execute_actions():
            self.automaton.set_auto_execute_actions(self.set_auto_execute_actions_checkbox.isChecked())
        self.set_auto_execute_actions_checkbox.stateChanged.connect(set_auto_execute_actions)
        right_layout.addWidget(self.set_auto_execute_actions_checkbox)

        run_action = QPushButton("Run Actions")
        def run_action_func():
            for action in self.parse_action_list:
                self.action_queue.put(action)
                self.action_queue.put(WaitAction(1))
        run_action.clicked.connect(run_action_func)
        right_layout.addWidget(run_action)

        main_layout.addLayout(right_layout)

        self.setWindowTitle("ScreenAgent VNC Viewer")
        self.setGeometry(0, 0, 100, 100)

        self.setMouseTracking(True)

        self.vnc = None
        self.connect_vnc()
        self.reset()

    def load_prompts_from_file(self):
        with open(self.task_list_path, "r", encoding="utf8") as f:
            task_prompt_list = f.readlines()
            self.task_prompt_list = [x.strip() for x in task_prompt_list]
        self.task_prompt_selection.clear()
        self.task_prompt_selection.addItems(self.task_prompt_list)

    @asyncSlot()
    async def reconnect(self):
        self.action_queue.queue.clear()
        await self.vnc.disconnect()
        self.connect_vnc()

    @asyncSlot()
    async def connect_vnc(self):
        self.statusBar().showMessage("Connecting")

        host = self.remote_vnc_server_config["host"]
        port = self.remote_vnc_server_config["port"]
        password = self.remote_vnc_server_config["password"]
        self._reader, self._writer = await open_connection(host, port)
        self.vnc = await Client.create(reader=self._reader, writer=self._writer, password=password)
        self.video_height = self.vnc.video.height
        self.video_width = self.vnc.video.width
        self.now_screenshot = np.zeros((self.video_height, self.video_width, 4), dtype='uint8')

        self.vnc_frame.setFixedSize(self.video_width, self.video_height)
        self.vnc_frame.setMouseTracking(True)

        self.task_prompt_display.setFixedWidth(self.vnc_frame.width())
        self.send_prompt_display.setFixedWidth(self.vnc_frame.width())

        self.refresh_timer.start()
        self.statusBar().showMessage("Prepare")

    def reset(self):
        self.LLM_response = ""
        self.LLM_response_display.clear()
        self.LLM_response_editer.clear()
        self.parse_action_list.clear()
        self.parse_action_display.clear()
        self._parse_action_display_action_map.clear()
        self.action_queue.queue.clear()
        self.saved_image_name = None

    def set_status_text(self):
        all_status_text = []
        all_status_text.append(self.last_message)
        if action_queue_size:=self.action_queue.qsize():
            all_status_text.append(f"{action_queue_size} Actions Waiting to Execute.")
        if self.vnc is not None:
            if local_cursor_pos:=self.vnc_frame.get_local_cursor_pos():
                all_status_text.append(f"Cursor Position: {str(local_cursor_pos)}")

        self.statusBar().showMessage(" ".join(all_status_text))

    def get_now_screenshot(self):
        return Image.fromarray(self.now_screenshot).convert('RGB')

    def set_automaton_state(self, state):
        self.automaton.set_state(state)

    def automaton_state_changed(self, state):
        for button in self.automaton_control_buttons:
            if button.text() == state:
                button.setStyleSheet("background-color: green")
            else:
                button.setStyleSheet("")

    async def update_screen(self):
        try:
            self.now_screenshot = await self.vnc.screenshot()
        except Exception as e:
            print("[update_screen] error:", e)

        rgba_array = self.now_screenshot
        if rgba_array is not None:
            qimage = QImage(rgba_array.tobytes(), self.video_width, self.video_height, QImage.Format_RGBA8888)
            self.vnc_frame.update_screen(qimage)

    @asyncSlot()
    async def render(self):
        self.refresh_timer.stop()

        self.wait_for_screen_refreshed=True

        if self.refreshing_screen==True:
            self.refresh_timer.start()
            return

        self.refreshing_screen=True
        await self.update_screen()
        self.set_status_text()
        recall_functions = []

        try:
            while not self.action_queue.empty():
                action = self.action_queue.get()
                action.action_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                action.before_action_obs = self.now_screenshot
                await action.step(self.vnc)

                action_request_id = action.request_id
                if action.request_id is not None:
                    self._request_action_num_counter[action_request_id] -= 1
                    if self._request_action_num_counter[action_request_id] == 0:
                        # The action is executed and executes the callback
                        request_recall_func = self._request_recall_func_cache[action_request_id]
                        recall_functions.append(request_recall_func) # Add the callback function to the pending queue
                        self.clear_request_cache(action_request_id) # Clear action request cache

                if action in self._parse_action_display_action_map:
                    # hightlight in parse_action_display
                    index = self._parse_action_display_action_map[action]
                    item = self.parse_action_display.item(index)
                    item.setForeground(QColor("green"))

                del action

        except Exception as e:
            print("[render] error:", e)

        self.set_status_text()
        self.refreshing_screen=False
        self.refresh_timer.start()

    def select_task_prompt(self, item):
        self.task_prompt = item.text()
        self.task_prompt_display.setText(self.task_prompt)

    def start_automaton(self):
        self.automaton.start(task_prompt=self.task_prompt, video_width=self.video_width, video_height=self.video_height)

    def execute_actions(self, request_id, actions, recall_func):
        # receive a list of actions from other components, execute them
        self._request_recall_func_cache[request_id] = recall_func
        self._request_action_num_counter[request_id] = len(actions)
        for action in actions:
            self.action_queue.put(action)

    def set_send_prompt_display(self, send_prompt):
        self.send_prompt = send_prompt
        self.send_prompt_display.setText(self.send_prompt)

    def set_llm_response(self, last_stream_response):
        self.LLM_response = last_stream_response
        self.LLM_response_display.setText(self.LLM_response)
        self.LLM_response_editer.setText(self.LLM_response)

    def ask_llm_sync(self, prompt, image, ask_llm_recall_func):
        self.send_prompt = prompt
        print(f"[func:ask_llm_sync] send_prompt: {self.send_prompt}")
        request_id = uuid.uuid4().hex
        self._request_recall_func_cache[request_id] = ask_llm_recall_func
        self.parse_action_display.clear()
        self.llm_client_gpt.send_request_to_server(self.send_prompt, image, request_id, self._send_to_main_ask_llm_sync_recall_func)

    def _send_to_main_ask_llm_sync_recall_func(self, response, fail_message, request_id):
        QMetaObject.invokeMethod(self, "ask_llm_sync_recall_func", Qt.QueuedConnection, Q_ARG(str, response), Q_ARG(str, fail_message), Q_ARG(str, request_id))

    @pyqtSlot(str, str, str)
    def ask_llm_sync_recall_func(self, response, fail_message, request_id):
        print(f"[func:ask_llm_sync_recall_func] llm response: {response}")
        if response:
            self.set_llm_response(response)
            actions = parse_action_from_text(response)
            self.parse_action_list = actions

            self.last_message = f"Parsed {len(actions)} actions"
            if len(actions) == 0:
                self.last_message = f"No action found."
            else:
                self.update_parse_action_display()
        else:
            self.last_message = "Ask LLM failed: " + fail_message
            self.clear_request_cache(request_id)

        request_recall_func = self._request_recall_func_cache[request_id]
        request_recall_func(actions)

    # sub task
    def update_sub_task_display(self, sub_task_list, now_sub_task_index):
        self.sub_task_display.clear()
        for i, sub_task in enumerate(sub_task_list):
            self.sub_task_display.addItem(f"{i}. {sub_task}")

        item = self.sub_task_display.item(now_sub_task_index)
        if item is not None:
            item.setForeground(QColor("green"))

    def set_current_task_index(self):
        current_row = self.sub_task_display.currentRow()
        self.automaton.set_current_task_index(current_row)

    def get_current_task(self):
        if self.automaton is not None:
            return self.automaton.get_current_task()

    def clear_request_cache(self, request_id):
        self._request_recall_func_cache.pop(request_id, None)
        self._request_action_num_counter.pop(request_id, None)

    def update_parse_action_display(self):
        self.parse_action_display.clear()
        self._parse_action_display_action_map.clear()
        for i, action in enumerate(self.parse_action_list):
            self.parse_action_display.addItem(action.to_ideal_display_format())
            self._parse_action_display_action_map[action] = i

    @asyncClose
    async def closeEvent(self, event):
        self.statusBar().showMessage("Closing")
        self.refresh_timer.stop()
        if self.vnc is not None:
            await self.vnc.disconnect()
        print("[func:closeEvent]")
        exit(0)
