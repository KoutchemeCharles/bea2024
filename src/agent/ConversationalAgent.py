""" Base Wrapper class around HuggingFace 
Instruction-tuned and chat models.

"""

from src.agent.Agent import Agent


class ConversationalAgent(Agent):
    """
    
    Could have used pipelines for this, but I'd rather
    perform all the steps manually to ensure control over
    what is happening.
    """

    def encode(self, all_messages, **kwargs):
        prompts = []
        for messages in all_messages:
            prompts.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True))
        return super().encode(prompts, **kwargs)
