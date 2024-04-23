import os
import pandas as pd

class FalconCode():

    splits = ["train", "val", "test"]

    def __init__(self, config) -> None:
        self.config = config 
        self.functional_form = False

        if self.config.reprocess:
            self.preprocess_dataset()

    from .preprocessing import preprocess_dataset
    from .execution import grade_fn

    def get_split(self, split):
        cid = self.splits.index(split) + 2
        tr_path = os.path.join(self.config.path, f"course_{cid}.json")
        if not os.path.exists(tr_path):
            raise ValueError("Processed training dataset does not exists")
        
        df = pd.read_pickle(tr_path)

        if self.config.subset: df = df[df["type"] == self.config.subset]
            
        return df 
    