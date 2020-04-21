import pandas as pd
from scipy import stats

class Frame(pd.DataFrame):

    def __init__(self, filename: str):
        """
        Args:
            filename: Location of CSV file to be loaded.
        """
        self.df = pd.read_csv(filename)


    def transform(self, column: str, transformation: float) -> None:
        """
        Applies box transformation to the specified column.
        Please refer to the scipy.stats.boxcox documentation for details.

        Args:
            column: Column name to be transformed.
            transformation: Type of transformation to be applied.
        """
        self.df[column] = stats.boxcox(self.df[column], transformation)
