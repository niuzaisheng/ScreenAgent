import asyncio
import asyncvnc
import sys

import argparse
import yaml

import qasync
import functools
from qasync import QApplication

import controller_core
from automaton import Automaton


def close_future(future, loop):
    loop.call_later(10, future.cancel)
    future.cancel()

async def run_client(config):

    loop = asyncio.get_event_loop()
    future = asyncio.Future()

    app = QApplication(sys.argv)
    if hasattr(app, "aboutToQuit"):
        getattr(app, "aboutToQuit").connect(functools.partial(close_future, future, loop))

    llm_api_client = None
    api_config = config['llm_api']

    if api_config.get("GPT4V", None):
        from interface_api.gpt4_client import LanguageModelClient
        llm_api_client = LanguageModelClient(api_config)

    elif api_config.get("LLaVA", None):
        from interface_api.llava_llm_client import LanguageModelClient
        llm_api_client = LanguageModelClient(api_config)

    elif api_config.get("CogAgent", None):
        from interface_api.cogagent_llm_client import LanguageModelClient
        llm_api_client = LanguageModelClient("CogAgent", api_config)

    elif api_config.get("ScreenAgent", None):
        from interface_api.cogagent_llm_client import LanguageModelClient
        llm_api_client = LanguageModelClient("ScreenAgent", api_config)

    else:
        raise ValueError("No LLM API config found")

    assert llm_api_client is not None, "No LLM API client found"

    automaton_config = config['automaton']
    automaton = Automaton(automaton_config)

    remote_vnc_server_config = config['remote_vnc_server']

    if remote_vnc_server_config.get('use_remote_clipboard'):
        from action import KeyboardAction
        KeyboardAction.set_remote_clipboard(remote_vnc_server_config)
        
    widget = controller_core.VNCWidget(remote_vnc_server_config, llm_client = llm_api_client, automaton = automaton)
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
