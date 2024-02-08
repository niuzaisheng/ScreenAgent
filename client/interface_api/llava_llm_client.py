import requests
import base64
from PIL import Image
from io import BytesIO
import threading

from interface_api.conversation import conv_templates

DEFAULT_IMAGE_TOKEN = "<image>"

def encode_image_to_base64_with_pil(image:Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

class LanguageModelClient:
    def __init__(self, server_name, llm_server_config):

        self.model_name = llm_server_config[server_name].get("model_name", "LLaVA-1.5")
        self.target_url = llm_server_config[server_name].get("target_url", "http://localhost:40000/worker_generate")
        self.temperature = llm_server_config.get("temperature", 1.0)
        self.top_p = llm_server_config.get("top_p", 0.9)
        self.max_tokens = llm_server_config.get("max_tokens", 500)

    def send_request_to_server(self, prompt, image:Image, request_id=None, ask_llm_recall_func=None):

        conv = conv_templates["vicuna_v1"].copy()
        image_base64 = None
        if image is not None:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            conv.append_message(conv.roles[0], inp)
            image_base64 = encode_image_to_base64_with_pil(image)
        else:
            conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_base64 = encode_image_to_base64_with_pil(image)
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'max_new_tokens': self.max_tokens,
            "image":image_base64
        }

        if ask_llm_recall_func is None:
            response = requests.post(self.target_url, json=payload)
            if response.status_code == 200:
                response = response.json()
                text = response["text"]
                return text
            else:
                return None
        else:
            def thread_func():
                text = None
                response = None
                fail_message = None
                try:
                    response = requests.post(self.target_url, json=payload)
                    if response.status_code == 200:
                        response = response.json()
                        text = response["text"]
                        
                except requests.exceptions.RequestException as e:
                    fail_message = f"An error occurred: {e}"

                ask_llm_recall_func(text, fail_message, request_id)

            thread = threading.Thread(target=thread_func)
            thread.start()
