"""
Volatility Forecasting using Machine Learning
Author: Gabriel Demetrios Lafis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class VolatilityForecaster:
    """
    Machine Learning model for volatility forecasting in financial markets.
    Supports multiple algorithms: Random Forest, Gradient Boosting, XGBoost.
    """
    
    def __init__(self, model_type: str = 'xgboost', lookback_period: int = 20):
        """
        Initialize the volatility forecaster.
        
        Args:
            model_type: Type of model ('rf', 'gb', 'xgboost')
            lookback_period: Number of periods to look back for features
        """
        self.model_type = model_type
        self.lookback_period = lookback_period
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        # Initialize model based on type
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns"""
        return np.log(prices / prices.shift(1))
    
    def calculate_realized_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Calculate realized volatility (rolling standard deviation)"""
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for volatility prediction.
        
        Features include:
        - Historical volatility at different windows
        - Returns statistics
        - Price momentum
        - Volume features
        """
        features = pd.DataFrame(index=df.index)
        
        # Calculate returns
        returns = self.calculate_returns(df['close'])
        
        # Historical volatility at different windows
        for window in [5, 10, 20, 30]:
            features[f'vol_{window}d'] = self.calculate_realized_volatility(returns, window)
        
        # Returns features
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
        
        # Squared returns (proxy for volatility)
        features['squared_return'] = returns ** 2
        features['squared_return_ma5'] = features['squared_return'].rolling(5).mean()
        features['squared_return_ma20'] = features['squared_return'].rolling(20).mean()
        
        # Price momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # High-Low range (intraday volatility proxy)
        features['hl_ratio'] = (df['high'] - df['low']) / df['close']
        features['hl_ratio_ma5'] = features['hl_ratio'].rolling(5).mean()
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Day of week effect
        features['day_of_week'] = df.index.dayofweek
        
        return features
    
    def prepare_data(self, df: pd.DataFrame, target_horizon: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df: DataFrame with OHLCV data
            target_horizon: Periods ahead to forecast
            
        Returns:
            X: Features DataFrame
            y: Target Series (future realized volatility)
        """
        # Create features
        X = self.create_features(df)
        
        # Create target (future realized volatility)
        returns = self.calculate_returns(df['close'])
        y = self.calculate_realized_volatility(returns, window=target_horizon).shift(-target_horizon)
        
        # Remove NaN values
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train(self, df: pd.DataFrame, target_horizon: int = 1, test_size: float = 0.2) -> Dict:
        """
        Train the volatility forecasting model.
        
        Args:
            df: DataFrame with OHLCV data
            target_horizon: Periods ahead to forecast
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare data
        X, y = self.prepare_data(df, target_horizon)
        
        # Split data (time series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict volatility for new data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Array of volatility predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create features
        X = self.create_features(df)
        X = X[self.feature_names]  # Ensure same feature order
        
        # Handle NaN values
        X = X.fillna(method='ffill').fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            return importance.sort_values('importance', ascending=False)
        else:
            return None
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'lookback_period': self.lookback_period
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.lookback_period = model_data['lookback_period']

def generate_sample_data(n_days: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq='D')
    
    # Generate realistic price data with volatility clustering
    returns = np.random.normal(0.0005, 0.02, n_days)
    volatility = np.abs(np.random.normal(0.02, 0.01, n_days))
    
    # Add GARCH-like volatility clustering
    for i in range(1, n_days):
        volatility[i] = 0.05 + 0.9 * volatility[i-1] + 0.05 * (returns[i-1] ** 2)
        returns[i] = np.random.normal(0, volatility[i])
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.abs(np.random.uniform(0, 0.02, n_days))),
        'low': prices * (1 - np.abs(np.random.uniform(0, 0.02, n_days))),
        'close': prices,
        'volume': np.random.uniform(1e6, 5e6, n_days)
    }, index=dates)
    
    return df

if __name__ == "__main__":
    # Example usage
    print("Generating sample data...")
    df = generate_sample_data(1000)
    
    print("\nTraining volatility forecasting model...")
    forecaster = VolatilityForecaster(model_type='xgboost')
    metrics = forecaster.train(df, target_horizon=1)
    
    print("\nTraining Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nFeature Importance:")
    importance = forecaster.get_feature_importance()
    print(importance.head(10))
    
    print("\nMaking predictions...")
    predictions = forecaster.predict(df.tail(30))
    print(f"Last 5 predictions: {predictions[-5:]}")
    
    print("\nModel trained successfully!")
