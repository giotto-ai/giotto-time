import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
from gtime.time_series_models import CVPipeline
from gtime.metrics import max_error, mae, rmse, log_mse
from gtime.time_series_models import AR, Naive, SeasonalNaive, TimeSeriesForecastingModel
from gtime.feature_extraction import MovingAverage, Shift
from gtime.forecasting import NaiveForecaster, DriftForecaster


@st.composite
def draw_unique_subset(draw, lst):
    return draw(st.lists(st.sampled_from(lst), min_size=1, max_size=len(lst)))


@st.composite
def naive_model(draw):
    horizon = draw(st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=4, unique=True))
    return (Naive, {'horizon': horizon})


@st.composite
def seasonal_naive_model(draw):
    horizon = draw(st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=4, unique=True))
    seasonal_length = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=4, unique=True))
    return (SeasonalNaive, {'horizon': horizon, 'seasonal_length': seasonal_length})


@st.composite
def ar_model(draw):
    horizon = draw(st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=4, unique=True))
    p = draw(st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=4, unique=True))
    explainer = draw(st.sampled_from([None, "lime", "shap"]))
    return (AR, {'horizon': horizon, 'p': p, 'explainer_type': [explainer]})


@st.composite
def models_grid(draw):
    model_list = [draw(ar_model()), draw(seasonal_naive_model()), draw(naive_model())]
    return dict(draw(draw_unique_subset(model_list)))


@st.composite
def metrics(draw):
    metric_list = [max_error, mae, rmse, log_mse]
    metrics = draw(draw_unique_subset(metric_list))
    metrics_dict = dict(zip([x.__name__ for x in metrics], metrics))
    return metrics_dict


class TestCVPipeline:
    @given(models=models_grid(),
           n_splits=st.integers(min_value=2, max_value=10),
           blocking=st.booleans(),
           metrics=metrics())
    def test_constructor(self, models, n_splits, blocking, metrics):
        cv_pipeline = CVPipeline(models_sets=models, n_splits=n_splits, blocking=blocking, metrics=metrics)
        list_len = np.sum([np.prod([len(y) for y in x.values()]) for x in models.values()])
        assert list_len == len(cv_pipeline.model_list)
        assert len(metrics) == len(cv_pipeline.metrics)

    @pytest.mark.parametrize('models', [{Naive: {'horizon': [3]},
                                        AR: {'horizon': [3],
                                             'p': [2, 3]}
                                        }])
    @pytest.mark.parametrize('metrics', [{'RMSE': rmse,
                                          'MAE': mae}])
    @pytest.mark.parametrize('n_splits', [3, 5])
    @pytest.mark.parametrize('blocking', [True, False])
    @pytest.mark.parametrize('seed', [5, 1000])
    def test_fit_predict(self, models, n_splits, blocking, metrics, seed):
        cv_pipeline = CVPipeline(models_sets=models, n_splits=n_splits, blocking=blocking, metrics=metrics)
        np.random.seed(seed)
        idx = pd.period_range(start='2011-01-01', end='2012-01-01')
        df = pd.DataFrame(np.random.standard_normal((len(idx), 1)), index=idx, columns=['1'])
        cv_pipeline.fit(df)
        assert cv_pipeline.cv_results_.shape == (len(cv_pipeline.model_list) * len(metrics), 4)
        y_pred = cv_pipeline.predict()
        horizon = cv_pipeline.best_model_.horizon
        assert y_pred.shape == (horizon, horizon)



    @pytest.mark.parametrize('models', [ {TimeSeriesForecastingModel: {'features': [[('s3', Shift(1), ['1'])]
        , [('ma10', MovingAverage(10), ['1'])]],
                                 'horizon': [4],
                                 'model': [NaiveForecaster(), DriftForecaster()]}}])
    @pytest.mark.parametrize('metrics', [{'RMSE': rmse,
                                          'MAE': mae}])
    @pytest.mark.parametrize('n_splits', [5])
    @pytest.mark.parametrize('fit_all', ['all', 'best'])
    def test_model_assembly(self, models, n_splits, metrics, fit_all):
        cv_pipeline = CVPipeline(models_sets=models, n_splits=n_splits, metrics=metrics)
        idx = pd.period_range(start='2011-01-01', end='2012-01-01')
        df = pd.DataFrame(np.random.standard_normal((len(idx), 1)), index=idx, columns=['1'])
        cv_pipeline.fit(df, refit=fit_all)
        assert cv_pipeline.cv_results_.shape == (len(cv_pipeline.model_list) * len(metrics), 4)
        y_pred = cv_pipeline.predict()
        horizon = cv_pipeline.best_model_.horizon
        assert y_pred.shape == (horizon, horizon)