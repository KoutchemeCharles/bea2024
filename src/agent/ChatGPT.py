""" 
Interface to OpenAI models.
https://platform.openai.com/docs/api-reference/chat

"""
import openai
from warnings import warn
from src.agent.RemoteAgent import RemoteAgent
from openai import OpenAI

class ChatGPT(RemoteAgent):

    def __init__(self, config, seed=42) -> None:
        super().__init__(config)
        self.client = OpenAI(
            # api_key defaults to os.environ.get("OPENAI_API_KEY")
            max_retries=5,
            timeout=300.0, # 5 minutes timeout
        )
        self.seed = seed
        self.name = self.config.name 

    def query_with_prompt(self, prompt, **gen_kwargs):
        try:
            completions = self.client.completions.create(
                model=self.config.name,
                prompt = prompt,
                **gen_kwargs
            )

            return [choice.text for choice in completions.choices]
        
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.BadRequestError as e:
            print("Request sent to server is apparently bad")
            print(e.status_code)
            print(e.response)
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)

        raise ValueError("Error while querying openai")
    
    def query_with_message(self, messages, **gen_kwargs):
        if type(messages[0]) == list:
            messages = messages[0]
            warn("You passed in argument as multiple list of messages but not suported yet, only generating for one")
        try:
            completions = self.client.chat.completions.create(
                model=self.config.name,
                messages=messages,
                **gen_kwargs
            )
            return [choice.message.content
                    for choice in completions.choices]
        
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
        except openai.BadRequestError as e:
            print("Request sent to server is apparently bad")
            print(e.status_code)
            print(e.response)
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)

        raise ValueError("Error while querying openai")
            
    