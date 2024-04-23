""" 
Base class for generating feedback using a given language model.

"""

from src.Experiment import Experiment
from src.grading.Grading import Grading
from src.repair.Repair import (
    _create_zero_shot_prompt, extract_repaired_code
)
from src.utils.evaluation import evaluate_functional_correctness
from datasets import Dataset 


class NLE(Experiment):
    """ 
    Zero-shot prompting language model to generate natural language explanations
    of issues in students programs.

    """

    def __init__(self, config, test_run=False) -> None:
        super().__init__(config, "Baseline", test_run)
        self.grader = Grading(self.config)


    def run(self):
        gen_params = self.config.task.hyperparameters
        test_data = Dataset.from_pandas(self.get_dataset("test"))
        feed_instr = self.config.task.toDict()["instructions"]

        # Create the query inputs separately
        gen_ds = test_data.map(_create_zero_shot_prompt, 
                            fn_kwargs={"instructions": feed_instr})
        
        # Batch the creation of the responses
        gen_ds = gen_ds.map(self._generate_feedback, 
                            batched=True, batch_size=2, 
                            fn_kwargs={"gen_params": gen_params})
        
        # Extracting the repairs from the feedbacks
        grader = self.dataset_handler.grade_fn
        repairs = extract_repaired_code(gen_ds["feedback"], "", "```python", "```")
        gen_ds = gen_ds.add_column("repair", repairs)
        gen_ds = gen_ds.map(_add_repair_correctness, fn_kwargs={"grader": grader})
        gen_ds = gen_ds.remove_columns(["query_inputs"])
        # Run evaluation of NLE correctness 
        results = self._evaluate(gen_ds.to_pandas())
        
        self._save_results(results, gen_params, self.results_save_dir, "")

        return results 


    def _generate_feedback(self, samples, gen_params):
        """ Generate feedback """
        
        query_inputs = samples["query_inputs"]

        if self.agent.config.source == "openai":
            prompts = ["\n".join([f"\n{q['role']}\n: {q['content']}" for q in qi]) 
                       for qi in query_inputs]
        else:
            prompts = [self.agent.tokenizer.apply_chat_template(qi, 
                                                                tokenize=False, 
                                                                add_generation_prompt=True)
                        for qi in query_inputs]
            
        samples["full_prompt"] = prompts 

        # If the LLM cannot process messages (i.e. it's not a chat model)
        # then we can format the query as a 'normal' prompt to be completed
        if not self.agent.config.is_chat: query_inputs = prompts 

        responses = self.agent.query(query_inputs, **gen_params)
        beacon = self.config.task.extract
        feedbacks = [extract_feedback(r, beacon) for r in responses]
        n_generations = len(responses) // len(query_inputs)
        
        if n_generations != 1: raise NotImplementedError()

        samples["full_response"] = responses
        samples["feedback"] = feedbacks
        
        return samples


    def _evaluate(self, grading_df):
        """
        Use GPT-4 to evaluate the quality of the natural language explanations
        generated by the other languages models. 

        Returns the judged dataset as well as the score for each evaluation 
        criteria.

        """

        gen_param = self.config.task.evaluation.toDict()["hyperparameters"]
        grading_df = self.grader._generate(grading_df, gen_param)
        criteria = list(self.config.task.evaluation.criteria)
        
        # Obtaining the results of performance per concept
        details = []
        columns = grading_df.loc[:, 'input_str': 'tuple'].columns
        for concept in columns:
            sub_df = grading_df[grading_df[concept].astype(bool)]
            if not len(sub_df): continue
            details.append({
                "concept": concept, 
                "n_exercises": len(set(sub_df["problem_id"])),
                "n_problems": len(sub_df),
                "scores": {c: sub_df[c].mean() for c in criteria},
                "eval_ds": sub_df.to_dict()
            }) 

        results = {
            "eval_ds": grading_df.to_dict(),
            "scores": {c: grading_df[c].mean() for c in criteria},
            "concepts_scores": details
        }
            
        return results


def extract_feedback(full_chat, beacon):
    # ChatGPT does not output the original prompt 
    start_index,end_index = 0, len(full_chat)
    if beacon and beacon in full_chat:
        start_index = full_chat.rfind(beacon) + len(beacon)
        
    feedback = full_chat[start_index:end_index] 
    lines = feedback.splitlines()
    lines = [l for l in lines if not l.startswith("<|")]
    return "\n".join(lines)


def _add_repair_correctness(sample, grader):
    sample["generation"] = sample["repair"]
    sample["completion_id"] = 0
    eval_ds = Dataset.from_list([sample])
    _, eval_ds = evaluate_functional_correctness(eval_ds, grader, k=[1])
    return eval_ds[0]