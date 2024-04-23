""" 
Accessing HuggingFace models through their Inference API
"""

from huggingface_hub import HfApi, ModelFilter
from easyllm.clients import huggingface
from src.agent.RemoteAgent import RemoteAgent

class HFRemoteAgent(RemoteAgent):

    def __init__(self, config) -> None:
        super().__init__(config)

        api = HfApi()
        models = api.list_models(
            filter=ModelFilter(
                model_name=self.config.name
            )
        )
        if not models:
            err_mess = f"Unknown huggingface model {self.config.name}"
            raise ValueError(err_mess)
        
        huggingface.prompt_builder = lambda messages: messages
        if self.config.name.startswith("meta-llama/Llama-2-"):
            huggingface.prompt_builder = "llama2"
        
    # if the model can be easily accessed through huggingface
    def query_with_messages(self, messages, stop=None, **gen_kwargs):
        response = huggingface.ChatCompletion.create(
                model=self.config.name,
                stop=stop,
                messages = messages,
                **gen_kwargs
            )

        return messages, [choice["message"]["content"] 
                for choice in response["choices"]]
    
    def query_with_prompt(self, prompt, stop=None, **gen_kwargs):
        response = huggingface.Completion.create(
            model=self.config.name,
            stop=stop,
            prompt = prompt,
            **gen_kwargs
        )

        return prompt, [choice["text"]
                for choice in response["choices"]]