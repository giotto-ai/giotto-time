import pandas as pd
import numpy as np

def compute_n_points(n_windows: int, sliding_stride: int, sliding_window_width: int, takens_stride: int,
                     takens_dimension: int, takens_time_delay: int) -> int:
    """Helper function to reshape TDA feature for use with giotto-time.

    Parameters
    ----------
    n_windows : int
        Number of windows
    sliding_stride : int 
        Sliding stride used in pipeline
    sliding_window_width : int 
        Width of the window used in pipeline
    takens_stride : int
        Stride used in the Takens embedding
    
    Returns
    -------
    n_used_points : int
        Parameter for reshaping the feature values to pandas dataframes.
    """
    embedder_length = (
        sliding_stride * (n_windows - 1) + sliding_window_width
    )

    n_used_points = (
        takens_stride * (embedder_length - 1)
        + takens_dimension * takens_time_delay
    )
    return n_used_points

def align_indices(X: pd.DataFrame, n_points: int, tda_feature_values: np.array) -> int:
    """Helper function to reshape TDA feature for use with giotto-time.

    Parameters
    ----------
    X : pd.DataFrame, required
        Original time series
    n_points : int, required
        Output of compute_n_points
    tda_feature_values : np.array, required
        Results of the TDA pipeline

    Returns
    -------
    output_X : pandas dataframe
        Reshaped dataframe with feature values.
    """
    output_X = X.copy()

    output_X.iloc[:-n_points] = np.nan

    splits = np.array_split(
        output_X.iloc[-n_points:].index.values, len(tda_feature_values)
    )

    for index, split in enumerate(splits):
        if isinstance(tda_feature_values[index], list) or isinstance(
            tda_feature_values[index], np.ndarray
        ):
            target_value = tda_feature_values[index][0]
        else:
            target_value = tda_feature_values[index]
        output_X.loc[split] = target_value

    return output_X