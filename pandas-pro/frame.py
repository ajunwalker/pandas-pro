import numpy as np
import pandas as pd
from scipy import stats

class Frame(pd.DataFrame):

    def __init__(self, filename: str):
        """
        Args:
            filename: Location of CSV file to be loaded.
        """
        df = pd.read_csv(filename)
        super().__init__(data=df)


    def shrink(self):
        """
        Automatically shrinks the integer and floating point column data types
        to a minimum.
        """
        for col in self.columns:
            if self[col].dtype == int:
                # Maximum value for reference
                maximum_value = self[col].max()

                if maximum_value > np.iinfo(np.uint32).max:
                    self[col] = self[col].astype(np.uint64)
                elif maximum_value > np.iinfo(np.uint16).max:
                    self[col] = self[col].astype(np.uint32)
                elif maximum_value > np.iinfo(np.uint8).max:
                    self[col] = self[col].astype(np.uint16)
                elif maximum_value < np.iinfo(np.uint8).max:
                    self[col] = self[col].astype(np.uint8)

            elif self[col].dtype == float:
                # Maximum value for reference
                maximum_value = self[col].max()

                if maximum_value > np.finfo(np.float32).max:
                    self[col] = self[col].astype(np.float64)
                elif maximum_value > np.finfo(np.float16).max:
                    self[col] = self[col].astype(np.float32)
                elif maximum_value < np.finfo(np.float32).max:
                    self[col] = self[col].astype(np.float16)



    def transform(self, column: str, transformation: float) -> None:
        """
        Applies box transformation to the specified column.
        Please refer to the scipy.stats.boxcox documentation for details.

        Args:
            column: Column name to be transformed.
            transformation: Type of transformation to be applied.
        """
        self.df[column] = stats.boxcox(self.df[column], transformation)
