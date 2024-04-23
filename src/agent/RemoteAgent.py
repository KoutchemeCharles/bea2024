""" 
Base class for interacting with various Language Models
through external APIs. 
"""

class RemoteAgent():
    """
    a fast agent can leverage easyllm to be queried instead of relying on 
    custom gpus.
    """

    def __init__(self, config) -> None:
        self.config = config 

    def query(self, inputs, **gen_kwargs):
        """ Batch querying """
        return [v for i in inputs for v in self.single_query(i, **gen_kwargs) ]
    
    def single_query(self, input_, **gen_kwargs):
        # Get the model generations 
        if self.config.is_chat:
            responses = self.query_with_message(input_, 
                                                 **gen_kwargs)
        else:
            responses = self.query_with_prompt(input_, **gen_kwargs)

        return responses