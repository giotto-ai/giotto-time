import numpy as np
import pandas as pd
import pandas.util.testing as testing
import pytest

from gtime.custom.crest_factor_detrending import CrestFactorDetrending

def get_input_data():
    input_data = pd.DataFrame.from_dict({"x_1": [0, 7, 2], "x_2": [2, 10, 4]})
    input_data.index = [
        pd.Timestamp(2000, 1, 1),
        pd.Timestamp(2000, 2, 1),
        pd.Timestamp(2000, 3, 1),
    ]
    return input_data

def get_output_causal():
    custom_feature = CrestFactorDetrending(window_size=2, is_causal = True)
    feature_name = custom_feature.__class__.__name__
    output_causal = pd.DataFrame.from_dict(
        {
            f"x_1__{feature_name}": [np.nan, 1.0, 0.07547169811320754],
            f"x_2__{feature_name}": [np.nan, 0.9615384615384616, 0.13793103448275862],
        }
    )
    output_causal.index = [
        pd.Timestamp(2000, 1, 1),
        pd.Timestamp(2000, 2, 1),
        pd.Timestamp(2000, 3, 1),
    ]
    return output_causal
    
def get_output_anticausal():
    custom_feature = CrestFactorDetrending(window_size=2, is_causal = False)
    feature_name = custom_feature.__class__.__name__
    output_anticausal = pd.DataFrame.from_dict(
        {
            f"x_1__{feature_name}": [1.0, 0.07547169811320754],
            f"x_2__{feature_name}": [0.9615384615384616, 0.13793103448275862],
        }
    )
    output_anticausal.index = [
        pd.Timestamp(2000, 2, 1),
        pd.Timestamp(2000, 3, 1),
    ]    
    return output_anticausal

input_data = get_input_data()
output_causal = get_output_causal()
output_anticausal = get_output_anticausal()

class TestCrestFactorDetrending:
    @pytest.mark.parametrize("test_input, expected", [(input_data, output_causal)])
    def test_crest_factor_detrending_causal(self, test_input, expected):
        feature = CrestFactorDetrending(window_size=2, is_causal = True)
        output = feature.fit_transform(test_input)
        testing.assert_frame_equal(output, expected)

    @pytest.mark.parametrize("test_input, expected", [(input_data, output_anticausal)])
    def test_crest_factor_detrending_anticausal(self, test_input, expected):
        feature = CrestFactorDetrending(window_size=2, is_causal = False)
        output = feature.fit_transform(test_input)
        testing.assert_frame_equal(output, expected)