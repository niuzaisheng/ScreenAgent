import base64
import requests
from PIL import Image
from io import BytesIO
import threading

def encode_image_to_base64_with_pil(image: Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class LanguageModelClient:
    def __init__(self, llm_server_config):
        
        self.model_name = llm_server_config["GPT4V"].get("model_name", "gpt-4-vision-preview")
        self.openai_api_key = llm_server_config["GPT4V"].get("openai_api_key")
        self.target_url = llm_server_config["GPT4V"].get("target_url", "https://api.openai.com/v1/chat/completions")
        self.temperature = llm_server_config.get("temperature", 0.9)
        self.top_p = llm_server_config.get("top_p", 0.9)
        self.max_tokens = llm_server_config.get("max_tokens", 500)

    def send_request_to_server(self, prompt, image: Image, request_id=None, ask_llm_recall_func=None):

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        image_base64 = encode_image_to_base64_with_pil(image)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens
        }

        if ask_llm_recall_func is None:
            response = requests.post(self.target_url, headers=headers, json=payload)
            if response.status_code == 200:
                response = response.json()["choices"][0]["message"]["content"]
                return response
            else:
                raise Exception(f"Server returned status code {response.status_code}, response: {response.text}")
        else:
            def thread_func():
                response = None
                fail_message = None
                try:
                    response = requests.post(self.target_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        response = response.json()["choices"][0]["message"]["content"]
                        print(response)
                    else:
                        fail_message = f"Server returned status code {response.status_code}"
                except requests.exceptions.RequestException as e:
                    fail_message = f"An error occurred: {e}"

                ask_llm_recall_func(response, fail_message, request_id)

            thread = threading.Thread(target=thread_func)
            thread.start()
