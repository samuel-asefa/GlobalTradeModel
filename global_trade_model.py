"""
Global Trade Flow Prediction Model - Initial Version
Basic linear regression model for trade flow prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_basic_trade_data():
    """Create basic trade dataset"""
    np.random.seed(42)
    n_samples = 100
    
    # Basic features
    gdp_exporter = np.random.lognormal(8, 1, n_samples)
    gdp_importer = np.random.lognormal(8, 1, n_samples)
    distance = np.random.lognormal(2, 0.5, n_samples)
    
    # Simple trade flow calculation
    trade_flow = (gdp_exporter * gdp_importer) / (distance ** 2) * np.random.lognormal(0, 0.3, n_samples)
    
    data = pd.DataFrame({
        'Exporter_GDP': gdp_exporter,
        'Importer_GDP': gdp_importer,
        'Distance': distance,
        'Trade_Flow': trade_flow
    })
    
    return data

def build_basic_model(data):
    """Build basic linear regression model"""
    X = data[['Exporter_GDP', 'Importer_GDP', 'Distance']]
    y = np.log(data['Trade_Flow'])  # Log transform
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.3f}")
    print(f"MSE: {mse:.3f}")
    
    return model

def main():
    print("Basic Trade Flow Prediction Model")
    print("=" * 35)
    
    data = create_basic_trade_data()
    print(f"Dataset created with {len(data)} samples")
    
    model = build_basic_model(data)
    
    return data, model

if __name__ == "__main__":
    data, model = main()