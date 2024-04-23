""" Base Wrapper class around HuggingFace 
Instruction-tuned and chat models.

"""

import torch 
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig,
    StoppingCriteriaList
)
from src.utils.StoppingCriteria import StopWordsStoppingCriteria

class Agent():
    """
    
    Could have used pipelines for this, but I'd rather
    perform all the steps manually to ensure control over
    what is happening.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        

    def encode(self, inputs, **kwargs):
        return self.tokenizer(inputs, return_tensors="pt",
                              truncation=True, padding=True,
                              pad_to_multiple_of=8,
                              **kwargs)
    

    def decode(self, output_ids):
        outputs = self.tokenizer.batch_decode(output_ids, 
                                              skip_special_tokens=True)
        return list(outputs)
    

    def query(self, all_inputs, stop=[], **gen_kwargs):
        # Makking sure to translate the arguments properly
        if "n" in gen_kwargs and "num_return_sequences" not in gen_kwargs:
            gen_kwargs["num_return_sequences"] = gen_kwargs.pop("n")
        if "max_tokens" in gen_kwargs:
            gen_kwargs["max_new_tokens"] = gen_kwargs.pop("max_tokens")

        if gen_kwargs["top_p"] == 1.0 and gen_kwargs["temperature"] == 0:
            gen_kwargs.pop("temperature")
        elif gen_kwargs["temperature"] == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True

        gen_conf = GenerationConfig(**gen_kwargs)
        generate = self.get_generate(gen_conf, stop_words=stop)
        max_len = gen_kwargs["max_new_tokens"]
        model_inputs = self.encode(all_inputs, max_length=max_len)
        return self.decode(generate(model_inputs))


    def get_generate(self, gen_conf, stop_words=[]):
        """ 
        Return a generation function which completes the given input.
        This could be seen as some kind of custom "pipeline" 
        similar to the Transformers' one 

        Parameters
        ----------
        model: AutoModelForCausalLM
            Transformers Decoder only model
        generation_config: GenerationConfig
            Generation parameters to use for inference

        """
        
        # https://huggingface.co/docs/accelerate/usage_guides/big_modeling

        # some stuff that automatically need to be adapted:
        gen_conf.eos_token_id = self.tokenizer.eos_token_id
        gen_conf.pad_token_id = self.tokenizer.pad_token_id 
        sw_ids = _encode_stop_words(self.tokenizer, stop_words)
        # not that this will work even with device is on multiple gpus
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def generate(inputs):
            inputs = inputs.to(device)
            input_ids = inputs["input_ids"]
            stop_crit = _get_stopping_criteria_list(sw_ids, input_ids)

            with torch.no_grad():
                gen_outputs = self.model.generate(**inputs, 
                                                  generation_config=gen_conf,
                                                  stopping_criteria=stop_crit)
                return gen_outputs

        return generate
    
    def load_model(self, path=None):
        path = path if path is not None else self.config.name
        dtype = "auto"
        if self.config.instantiation.torch_dtype == "torch.bfloat16":
            dtype = torch.bfloat16
        elif self.config.instantiation.torch_dtype == "torch.float16":
            dtype = torch.float16
        
        return AutoModelForCausalLM.from_pretrained(path, 
                                                    device_map="auto", # Using accelerator for multi-gpu setups 
                                                    trust_remote_code=True,
                                                    torch_dtype=dtype)
    
    def load_tokenizer(self):
        # Padding to the left since we implement decoder models only
        tokenizer = AutoTokenizer.from_pretrained(self.config.name, 
                                                  padding_side='left')
        tokenizer.truncation_side = "left"
        if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    @property
    def name(self):
        return self.config.name 


def _encode_stop_words(tokenizer, stop_words):
    f = lambda w: tokenizer.encode(w, add_special_tokens=False)
    return list(map(f, stop_words))
            
            
def _get_stopping_criteria_list(sw_ids, input_ids):
    max_input_length = input_ids.size(1)
    stopping_criteria = None
    if sw_ids:
        stopping_criteria = StoppingCriteriaList()
        max_lengths = [max_input_length for l in range(len(input_ids))]
        ssc = StopWordsStoppingCriteria(max_lengths, sw_ids)
        stopping_criteria.append(ssc)

    return stopping_criteria
    
