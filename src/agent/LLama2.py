
from src.agent.Agent import Agent


class Llama2(Agent):
    def __init__(self, config) -> None:
        super().__init__(config)

    def load_tokenizer(self):
        # https://discuss.huggingface.co/t/llama2-pad-token-for-batched-inference/48020/2
        tokenizer = super().load_tokenizer()
        #tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.add_special_tokens({"pad_token":"[PAD]"}) 
        tokenizer.padding_side = "left" 
        
        return tokenizer
    