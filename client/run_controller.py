import asyncio
import sys
import gym_asyncvnc_qt

import argparse
import yaml

import qasync
import functools
from qasync import QApplication
from automaton import Automaton


def close_future(future, loop):
    loop.call_later(10, future.cancel)
    future.cancel()

async def run_client():

    loop = asyncio.get_event_loop()
    future = asyncio.Future()

    app = QApplication(sys.argv)
    if hasattr(app, "aboutToQuit"):
        getattr(app, "aboutToQuit").connect(functools.partial(close_future, future, loop))



    from fastchat_interface.gpt4_client import LanguageModelClient as LanguageModelClientGPT
    from fastchat_interface.llava_llm_client import LLaVALanguageModelClient
    from fastchat_interface.cogagent_llm_client import CogAgentLanguageModelClient



    # llm_client_gpt = LanguageModelClientGPT("gpt-4-vision-preview", "http://10.147.19.10:15000/forward")
    # llm_client_gpt = LanguageModelClientGPT("gpt-4-vision-preview", "http://49.140.25.220:15000/forward")
    # llm_client_gpt = LLaVALanguageModelClient("llava-v1.5-13b", "http://localhost:40001/worker_generate")
    # llm_client_gpt = CogAgentLanguageModelClient("CogAgent", "http://localhost:40001/worker_generate")
    llm_client_gpt = CogAgentLanguageModelClient("ScreenAgent", "http://localhost:41000/worker_generate")
    # llm_client_fuyu = FuyuLanguageModelClient("adept/fuyu-8b", "http://localhost:40000/worker_generate")
    llm_client_fuyu = None

    language = "en"
    # language = "zh"
    # operation_system = "win"
    operation_system="linux"
    automaton = Automaton(language, operation_system, asyncio_loop=loop)

    use_remote_clipboard = True

    # win
    ip, password, port, remote_clipboard_port = "10.147.19.10", "Ohch3Quugh", 5929, None
    # ip, password, port, remote_clipboard_port = "49.140.25.220", "Ohch3Quugh", 5929, 8001

    remote_clipboard_ip = ip

    if use_remote_clipboard:
        from action import KeyboardAction
        KeyboardAction.set_remote_clipboard(remote_clipboard_ip, remote_clipboard_port)
       
    widget = gym_asyncvnc_qt.VNCWidget(ip=ip, port=port, password=password, save_base_dir=save_base_dir,
                                       llm_client_gpt=llm_client_gpt, llm_client_fuyu=llm_client_fuyu,
                                       automaton=automaton)
    widget.show()
    await future


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Read YAML config file.')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()

    # read config file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    try:
        qasync.run(run_client(config))
    except asyncio.exceptions.CancelledError:
        sys.exit(0)
