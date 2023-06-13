import numpy as np
import pandas as pd

from pydantic import BaseModel
from typing import Any, Optional, List
from shap import KernelExplainer

class CustomExplainer(BaseModel):

    model: Any
    data: pd.DataFrame
    groupby_cols: List[str] = ['user', 'age_level']
    ts_cols: List[str] = ['brand']
    

    class Config:
        arbitrary_types_allowed=True

    def _predict_proba(self, x: np.ndarray):

        # reconstruct data from numpy array
        x_transformed = pd.DataFrame(x, columns=self.groupby_cols + self.ts_cols).explode(self.ts_cols)

        # should not be x, should be whole untransformed feature set
        y_pred_proba = self.model.predict_proba(self.data)

        return np.random.normal(loc=0, scale=1, size=(1,3))

    def explain(self):

        x_transformed = self.data.groupby(['user', 'age_level'])[self.ts_cols].agg(list).reset_index().values
        
        explainer = KernelExplainer(
            self._predict_proba,
            x_transformed
        )

        print(explainer.shap_values(x_transformed))


class Custom_TFT(BaseModel):

    def fit(self):
        pass

    def predict_proba(self, x: np.ndarray):

        return np.random.normal(loc=0, scale=1, size=(1,3))

if __name__ == '__main__':

    df = pd.read_csv('df_10.csv')

    df.rename(columns={"clk": "click"},inplace=True)
    df['age_level'] = df['age_level'].astype(str)
    df['brand'] = df['brand'].astype(int).astype(str)
    df['click'] = df['click'].astype(str)
    df = df[df.user == 43]

    #print(df)

    tft = Custom_TFT()

    explainer = CustomExplainer(
        model=tft,
        data=df
    )

    explainer.explain()

    

    
