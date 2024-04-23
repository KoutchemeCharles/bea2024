from copy import deepcopy
from itertools import combinations

import numpy as np 
import pandas as pd
from datasets import Dataset

from src.Experiment import Experiment
from src.utils.extraction import locate 
from src.utils.distance import rougelcsum_dist
from src.utils.evaluation import  evaluate_functional_correctness
from src.utils.utils import duplicate, format_string


class Repair(Experiment):
    """
    Benchmarking Language Models Ability to fix students programs.
    """

    def __init__(self, config, test_run=False) -> None:
        super().__init__(config, "repair", test_run)    
        
    def run(self):
        
        print("config is", self.config)
        
        test_data = self.get_dataset("test")
        test_data = Dataset.from_pandas(test_data)

        gen_params = self.config.task.hyperparameters
        if self.test_run: gen_params["n"] = 2
        instr = self.config.task.toDict()["instructions"]

        gen_ds = test_data.map(_create_zero_shot_prompt,
                            fn_kwargs={"instructions": instr})
        # TODO: dynamic batch size based on the model size
        batch_size = 2
        if gen_params["n"] <= 2: batch_size = 8
        print("batch_size", batch_size)

        gen_ds = gen_ds.map(self.batch_query_and_process, 
                            batched=True, batch_size=batch_size, 
                            fn_kwargs={"gen_params": gen_params})
        results = self._evaluate(gen_ds, gen_params)
        gen_ds = gen_ds.remove_columns(["query_inputs"])
        
        self._save_results(results, gen_params, self.results_save_dir)


    def batch_query_and_process(self, samples, gen_params):
        query_inputs = samples["query_inputs"]
        if self.agent.config.source == "openai":
            prompts = ["\n".join([f"\n{q['role']}\n: {q['content']}" for q in qi]) 
                       for qi in query_inputs]
        else:
            prompts = [self.agent.tokenizer.apply_chat_template(qi, 
                                                                tokenize=False, 
                                                                add_generation_prompt=True)
                        for qi in query_inputs]
        
        samples["generation_prompt"] = prompts 

        # If the LLM cannot process messages (i.e. it's not a chat model)
        # then we can format the query as a 'normal' prompt to be completed
        assert self.agent.config.is_chat
            
        responses = self.agent.query(query_inputs, **gen_params)
        # split the responses based on the number of queries
        # and the number of generation asked.
        n_generations = len(responses) // len(query_inputs)
        # Format the output of the Dataset mapping function
        new_output = {k: duplicate(v, n_generations) 
                        for k, v in samples.items()}
        beacon1, beacon2, beacon3 = self.config.task.extract
        # if chatgpt, we don't get the full completion 
        if self.agent.config.source != "huggingface": beacon1 = "" 
        generations = extract_repaired_code(responses, beacon1, beacon2, beacon3)
        new_output["full_response"] = responses
        # new_output["generation_prompt"] = prompts # TODO: need to duplicate this one too 
        new_output["generation"] = generations
        # add a completion id to track the generations for evaluation  
        cids = list(range(n_generations)) * len(query_inputs)
        new_output["completion_id"] = cids
        
        return new_output 
   

    def _evaluate(self, eval_ds, gen_param):
        grade_fn = self.dataset_handler.grade_fn

        # Obtaining the results of performance per concept
        details = []
        columns = eval_ds.to_pandas().loc[:, 'input_str': 'tuple'].columns
        for c in columns:
            sub_ds = eval_ds.filter(lambda ex: bool(ex[c]))
            # sub_ds = sub_ds.remove_columns(["generation_correct", "generation_score"])
            if not len(sub_ds): continue
            sub_scores, sub_ds = compute_scores(grade_fn, sub_ds)

            details.append({
                "concept": c, 
                "n_exercises": len(set(sub_ds["problem_id"])),
                "n_problems": len(sub_ds),
                "scores": sub_scores,
                "eval_ds": sub_ds.to_dict()
            }) 
        
        # Obtaining the results for the full dataframe 
        all_scores, eval_ds = compute_scores(grade_fn, eval_ds)

        results = {
            "eval_ds": eval_ds.to_dict(),
            "scores": all_scores,
            "hyperparameters": gen_param,
            "concepts_scores": details
        }

        return results


def _create_zero_shot_prompt(sample, instructions):
    """ 
    Fills the prompt template using the information for
    the given query.
    """

    instr = deepcopy(instructions)
    instr[0]["content"] = format_string(sample, instr[0]["content"])
    sample["query_inputs"] = instr

    return sample


def extract_repaired_code(responses, extract_point, start_beacon, end_beacon):
    repairs = []
    for _, response in enumerate(responses):
        assert type(response) == str
        ix = 0 if (not extract_point or extract_point not in response) else response.rfind(extract_point)
        """if ix == -1:
            m = f"Beacon {extract_point} not in response: {response}"
            warn(m)"""
        # Anyway if the extract point is not there, it should be the last repair 
        ix = ix + len(extract_point)
        # Extract the element of interest given the beacons
        results = locate(start_beacon, end_beacon, response[ix:])
        repairs.append(results[-1] if results else "")

    return repairs


def compute_scores(grade_fn, eval_ds):
    """ 
    Computes our evaluation metrics given a model
    generations for the repair task. 
    """
    heval_k = list(range(1, len(set(eval_ds["completion_id"])) + 1))
    pass_at_k, eval_ds = (
        evaluate_functional_correctness(eval_ds, 
                                        grade_fn, 
                                        heval_k))
    rouge_at_k = (eval_ds.to_pandas()
                  .groupby("id")
                  .apply(compute_buggy_rouge_at_k)
                  .mean(axis=0)
                  .to_dict())

    d = {}
    d.update(pass_at_k)
    d.update(rouge_at_k)

    return d, eval_ds


def compute_buggy_rouge_at_k(prob_df):
    buggy = prob_df["source_code"].iloc[0]
    gencor = prob_df["generation_correct"]
    rouges = [rougelcsum_dist(buggy, gen, get_score=True) for gen in prob_df["generation"]]
    # averaging over all the correct generations
    l = [r for r, c in zip(rouges, gencor) if c] 
    normal_rouge = np.mean(l) if l else 0
    rouges = [r if c else 0 for r, c in zip(rouges, gencor)]   
    # measure introduced in the neurips benchmarking paper 
    rouge_at_k = {f"rouge@{k}": np.mean([max(score) 
                                         for score in combinations(rouges, k)]) 
                  for k in range(1, len(rouges) + 1)}
    rouge_at_k["rouge"] = normal_rouge
    
    return pd.Series(rouge_at_k)
  