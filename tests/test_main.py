"""
Unit tests for ml-volatility-forecasting
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.volatility_model import VolatilityForecaster, generate_sample_data


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for tests."""
    return generate_sample_data(300)


@pytest.fixture
def trained_forecaster(sample_data):
    """Return a trained VolatilityForecaster."""
    forecaster = VolatilityForecaster(model_type='xgboost')
    forecaster.train(sample_data)
    return forecaster


class TestGenerateSampleData:
    def test_returns_dataframe(self):
        df = generate_sample_data(100)
        assert isinstance(df, pd.DataFrame)

    def test_has_ohlcv_columns(self):
        df = generate_sample_data(100)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in df.columns

    def test_correct_length(self):
        df = generate_sample_data(200)
        assert len(df) == 200

    def test_datetime_index(self):
        df = generate_sample_data(50)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_positive_prices(self):
        df = generate_sample_data(100)
        assert (df['close'] > 0).all()


class TestVolatilityForecaster:
    def test_init_xgboost(self):
        f = VolatilityForecaster(model_type='xgboost')
        assert f.model is not None
        assert f.model_type == 'xgboost'

    def test_init_rf(self):
        f = VolatilityForecaster(model_type='rf')
        assert f.model is not None

    def test_init_gb(self):
        f = VolatilityForecaster(model_type='gb')
        assert f.model is not None

    def test_init_invalid_type(self):
        with pytest.raises(ValueError):
            VolatilityForecaster(model_type='invalid')

    def test_calculate_returns(self, sample_data):
        f = VolatilityForecaster()
        returns = f.calculate_returns(sample_data['close'])
        assert len(returns) == len(sample_data)
        assert pd.isna(returns.iloc[0])  # first return is NaN

    def test_calculate_realized_volatility(self, sample_data):
        f = VolatilityForecaster()
        returns = f.calculate_returns(sample_data['close'])
        vol = f.calculate_realized_volatility(returns, window=20)
        assert len(vol) == len(returns)
        # First 19 values should be NaN (window=20)
        assert pd.isna(vol.iloc[0])
        # Non-NaN values should be positive
        valid_vol = vol.dropna()
        assert (valid_vol >= 0).all()

    def test_create_features(self, sample_data):
        f = VolatilityForecaster()
        features = f.create_features(sample_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        assert 'vol_5d' in features.columns
        assert 'vol_20d' in features.columns
        assert 'return_lag_1' in features.columns
        assert 'hl_ratio' in features.columns

    def test_train_returns_metrics(self, sample_data):
        f = VolatilityForecaster(model_type='xgboost')
        metrics = f.train(sample_data)
        assert isinstance(metrics, dict)
        for key in ['train_mse', 'test_mse', 'train_r2', 'test_r2',
                     'train_mae', 'test_mae', 'train_rmse', 'test_rmse']:
            assert key in metrics
            assert isinstance(metrics[key], float)

    def test_predict_returns_array(self, trained_forecaster, sample_data):
        predictions = trained_forecaster.predict(sample_data.tail(30))
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 30

    def test_feature_importance(self, trained_forecaster):
        importance = trained_forecaster.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) > 0

    def test_save_load_model(self, trained_forecaster, sample_data, tmp_path):
        filepath = str(tmp_path / "model.joblib")
        trained_forecaster.save_model(filepath)

        new_forecaster = VolatilityForecaster()
        new_forecaster.load_model(filepath)
        assert new_forecaster.model_type == 'xgboost'
        assert len(new_forecaster.feature_names) > 0

        # Predictions should match
        pred_original = trained_forecaster.predict(sample_data.tail(10))
        pred_loaded = new_forecaster.predict(sample_data.tail(10))
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
