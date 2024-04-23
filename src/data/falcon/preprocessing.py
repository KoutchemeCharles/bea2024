import re
import os 
import pandas as pd 
from src.utils.TableConverter import TableConverter
from src.utils.code import (
    does_compile, keep_unique_solutions, simple_clean
)

def preprocess_dataset(self):
    path = self.config.path 

    problems_df = pd.read_csv(os.path.join(path, "falconcode_v1_table_problems.csv"))
    samples_df = pd.read_csv(os.path.join(path, "falconcode_v1_code_samples.csv"))
    runs_df = pd.read_csv(os.path.join(path, "falconcode_v1_table_runs.csv"))

    # transform the html prompts into markdown for easier reading
    problems_df["prompt"] = problems_df["prompt"].apply(html_to_md)

    rejected_problems = find_rejected_problems(problems_df)

    dataframe = merge_in_one(problems_df, samples_df, runs_df)

    if self.config.processing.remove_zero_passing:
        dataframe = dataframe[dataframe.score > 0]
    
    # remove the assignments which we cannot process
    dataframe = dataframe[dataframe.type != "project"]
    dataframe = dataframe[~dataframe.problem_id.isin(rejected_problems)]
    # We only care about the code submitted for evaluation
    dataframe = dataframe[dataframe.score != -1]
    dataframe = dataframe.dropna()

    if self.config.processing.select_last:
        print("only selecting the last submitted codes")
        # Only selecting the last solutions from students 
        groups = dataframe.groupby(["problem_id", "student_id"], as_index=False)
        dataframe = groups.apply(filter_that)
    
    # Keep only the compiling codes
    dataframe = dataframe[[does_compile(code) 
                            for code in dataframe["source_code"]]]
    dataframe.source_code = dataframe.source_code.apply(simple_clean)
    
    # dataframe["source_code"] = dataframe["source_code"].apply(simple_clean)
    
    # So far, we do not care about the skills columns 
    #dataframe = dataframe.loc[:, :'max_score']
    dataframe = dataframe.reset_index(drop=True)
    # must be added here to not be removed later 
    dataframe["correct"] = (dataframe.score == dataframe.max_score.astype(int))
        
    course_frames = split_per_semester(dataframe)

    for cid, df in course_frames.items():
        #print("saving dataset for course", cid, "with df", df)
        df.to_pickle(os.path.join(self.config.path, f"course_{cid}.json"))

    # create the mapping 
    return course_frames


def find_rejected_problems(problems_df):
    """ Change the prompts format to markdown and
    obtain the prompts to filter out. """

    # transform the html prompts into markdown for easier reading
    
    rejected_sentences = ["template", "file", "dataset", ".csv", ".txt"]
    check = lambda p: bool(sum([rs in p for rs in rejected_sentences]))
    mask = [check(prompt) for prompt in problems_df["prompt"]]
    rejected_problems = list(problems_df[mask].id.unique())

    return rejected_problems

def split_per_semester(dataframe):
    """ 
    split the dataset per semester and filter out
    duplicate submissions in each semester.

    """ 

    groups = dataframe.groupby("course_id").groups
    course_frames = {course_id: dataframe.loc[index].drop_duplicates("code_hash", ignore_index=True, keep='last')
                    for course_id, index in groups.items()}
    course_frames = {cid: keep_unique_solutions(df, "source_code", "problem_id")
                    for cid, df in course_frames.items()}

    for cid, sdf in course_frames.items():
        print(f"Course id {cid} dataset details")
        print("Number of solutions", len(sdf.source_code))
        print("Number of correct and incorrect solutions", sdf["correct"].value_counts())
        print("Number of problems", len(sdf.problem_id.unique()))
        
        print("Number of problems per difficulty level", sdf.groupby("type").problem_id.nunique())
        print("Number of correct and incorrect per difficulty level", sdf.groupby(["type"]).correct.value_counts())
        print("Number of students", len(sdf.student_id.unique()))

    return course_frames


def merge_in_one(problems_df, samples_df, runs_df):
    df = runs_df.merge(samples_df, left_on="code_hash", right_on="hash", 
                       how="left", suffixes=(None, "_y"))
    df = df.merge(problems_df, left_on=["problem_id", "course_id"], 
                  right_on=["id", "course_id"], how="left", suffixes=(None, "_y"))
    return df


def html_to_md(html, **options):
    mkdwn = TableConverter(**options).convert(html)
    return re.sub(r'\n\s*\n', '\n', mkdwn).strip()

def filter_that(gp):
    gp = gp.sort_values("timestamp")
    return gp.iloc[-1]
