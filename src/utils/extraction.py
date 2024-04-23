import re 

def locate(start_beacon, end_beacon, string):
    regex = f"(?<={start_beacon})(.+?)(?={end_beacon})"
    matches = re.findall(regex, string, re.DOTALL)
    return [m for m in matches 
            if m and not (m.startswith(start_beacon) 
                          or m.endswith(end_beacon))]


def smart_sample(df, n=None, m=1):
    
    def take_highest_scoring(sub_df):
        return sub_df.sort_values(by="score").iloc[-m:]
    
    if n is None: n = len(df["problem_id"].unique()) 
    # Take easiest cases to repair from the hardest exercises
    df = df.groupby("problem_id", as_index=False).apply(take_highest_scoring)
    # df = df.sort_values(by="score", ascending=True).iloc[:]

    return df

