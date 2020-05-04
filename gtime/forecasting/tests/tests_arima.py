from hypothesis import given, settings
from gtime.forecasting.tests.test_simple_models import forecast_input, SimplePipelineTest # TODO to utilities?
import hypothesis.strategies as st

from gtime.forecasting import ARIMAForecaster


class TestNaiveModel(SimplePipelineTest):

    @given(data=forecast_input(50, 10),
           order=st.tuples(st.integers(min_value=0, max_value=4),
                           st.integers(min_value=0, max_value=2),
                           st.integers(min_value=0, max_value=4))
           )
    @settings(deadline=None)
    def setup(self, data, order):
        super().setup(data, ARIMAForecaster(order=order))
