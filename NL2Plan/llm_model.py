import os
import tiktoken
import requests
from retry import retry
from openai import OpenAI
from .utils.logger import Logger

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', None))

@retry(tries=2, delay=60)
def connect_openai(engine, messages, temperature, max_tokens,
                   top_p, frequency_penalty, presence_penalty, stop):
    return client.chat.completions.create(
        model=engine,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )


class LLM_Chat:
    # Simple abstract class for the LLM chat
    def __init__(self, *args, **kwargs):
        pass

    def get_response(self, prompt=None, messages=None):
        raise NotImplementedError
    
    def token_usage(self) -> tuple[int, int]:
        raise NotImplementedError
    
    def reset_token_usage(self):
        raise NotImplementedError

class GPT_Chat(LLM_Chat):
    def __init__(self, engine, stop=None, max_tokens=4e3, temperature=0, top_p=1,
                 frequency_penalty=0.0, presence_penalty=0.0, seed=0):
        self.engine = engine
        self.temperature = temperature
        self.top_p = top_p
        self.freq_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.context_length = {
            "gpt-3.5-turbo-0125": 16e3, # 16k tokens
            "gpt-3.5-turbo-instruct": 4e3, # 4k tokens
            "gpt-4-1106-preview": 128e3, # 128k tokens
            "gpt-4-turbo-2024-04-09": 128e3, # 128k tokens
            "gpt-4": 8192, # ~8k tokens
            "gpt-4-32k": 32768, # ~32k tokens
            "gpt-4o": 32768, # ~32k tokens
        }[engine]
        self.max_tokens = max_tokens if max_tokens is not None else self.context_length
        self.tok = tiktoken.get_encoding("cl100k_base") # For GPT3.5+
        self.in_tokens = 0
        self.out_tokens = 0
        print(f"Seed is not used for OpenAI models. Discarding seed {seed}")

    def get_response(self, prompt=None, messages=None, end_when_error=False, max_retry=5, est_margin = 200):
        if prompt is None and messages is None:
            raise ValueError("prompt and messages cannot both be None")
        if messages is not None:
            messages = messages
        else:
            messages = [{'role': 'user', 'content': prompt}]

        # Calculate the number of tokens to request. At most self.max_tokens, and prompt + request < self.context_length
        current_tokens = int(sum([len(self.tok.encode(m['content'])) for m in messages])) # Estimate current usage
        requested_tokens = int(min(self.max_tokens, self.context_length - current_tokens - est_margin)) # Request with safety margin
        Logger.log(f"Requesting {requested_tokens} tokens from {self.engine} (estimated {current_tokens - est_margin} prompt tokens with a safety margin of {est_margin} tokens)")
        self.in_tokens += current_tokens

        # Request the response
        n_retry = 0
        conn_success = False
        while not conn_success:
            n_retry += 1
            if n_retry >= max_retry:
                break
            try:
                print(f'[INFO] connecting to the LLM ({requested_tokens} tokens)...')
                response = connect_openai(
                    engine=self.engine,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=requested_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.freq_penalty,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop
                )
                llm_output = response.choices[0].message.content # response['choices'][0]['message']['content']
                conn_success = True
            except Exception as e:
                print(f'[ERROR] LLM error: {e}')
                if end_when_error:
                    break
        if not conn_success:
            raise ConnectionError(f'Failed to connect to the LLM after {max_retry} retries')
        
        response_tokens = len(self.tok.encode(llm_output)) # Estimate response tokens
        self.out_tokens += response_tokens

        return llm_output
    
    def token_usage(self) -> tuple[int, int]:
        return self.in_tokens, self.out_tokens
    
    def reset_token_usage(self):
        self.in_tokens = 0
        self.out_tokens = 0

class OLLAMA_Chat(LLM_Chat):
    def __init__(self, engine, stop=None, max_tokens=8e3, temperature=0, top_p=1,
                 frequency_penalty=0.0, presence_penalty=0.0, seed=0):
        self.engine = engine
        self.url = os.environ.get("OLLAMA_URL", ).strip("/").replace("api/generate", "api/chat")
        if not self.url.endswith("/"):
            self.url += "/"
        if not self.url.endswith("api/chat/"):
            self.url += "api/chat/"
        self.temperature = temperature
        self.seed = seed
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.in_tokens = 0
        self.out_tokens = 0
        self.tok = tiktoken.get_encoding("cl100k_base") # For GPT3.5+

    def get_response(self, prompt=None, messages=None):
        if prompt is None and messages is None:
            raise ValueError("prompt and messages cannot both be None")
        if messages is not None:
            messages = messages
        else:
            messages = [{'role': 'user', 'content': prompt}]

        self.in_tokens += sum([len(self.tok.encode(m['content'])) for m in messages])

        to_send = {
            "model": self.engine,
            "messages": messages,
            "stream": False,
            "options": {
                "seed": self.seed,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty
            }
        }

        resp = requests.post(self.url, json=to_send)
        if resp.status_code != 200:
            print(f"Failed to connect to OLLAMA at {self.url}: {resp.status_code}. \n\t{resp.text}")
            raise ConnectionError(f"Failed to connect to OLLAMA at {self.url}: {resp.status_code}. \n\t{resp.text}")
        ans = resp.json()["message"]["content"]

        self.out_tokens += len(self.tok.encode(ans))

        return ans

    def token_usage(self) -> tuple[int, int]:
        print("WARNING: Ollama token usage is currently estimated with GPT tokenization.")
        return self.in_tokens, self.out_tokens
    
    def reset_token_usage(self):
        self.in_tokens = 0
        self.out_tokens = 0

def get_llm(engine, **kwargs) -> LLM_Chat:
    if "gpt-" in engine:
        return GPT_Chat(engine, **kwargs)
    else:
        return OLLAMA_Chat(engine, **kwargs)

if __name__ == '__main__':
    model = get_llm("gpt-3.5-turbo-0125")
    print(model.get_response("Hello world!"))