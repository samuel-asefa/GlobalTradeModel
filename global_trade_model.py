"""
Global Trade Flow Prediction Model - Enhanced Dataset
Added more features and realistic country data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_trade_data():
    """Create enhanced trade dataset with more features"""
    np.random.seed(42)
    
    # Major trading countries
    countries = ['USA', 'China', 'Germany', 'Japan', 'UK', 'France', 'Italy', 
                'Canada', 'India', 'Brazil', 'Australia', 'Mexico', 'Spain', 
                'Russia', 'Netherlands', 'South Korea']
    
    # Create country pairs
    country_pairs = []
    for i, exporter in enumerate(countries):
        for j, importer in enumerate(countries):
            if i != j:
                country_pairs.append((exporter, importer))
    
    n_pairs = len(country_pairs)
    print(f"Creating data for {n_pairs} country pairs")
    
    # Economic indicators
    exporter_gdp = np.random.lognormal(10, 1.2, n_pairs)
    importer_gdp = np.random.lognormal(10, 1.2, n_pairs)
    exporter_pop = np.random.lognormal(3, 1, n_pairs)
    importer_pop = np.random.lognormal(3, 1, n_pairs)
    distance = np.random.lognormal(2, 0.8, n_pairs)
    
    # New features
    trade_agreement = np.random.binomial(1, 0.3, n_pairs)
    common_language = np.random.binomial(1, 0.2, n_pairs)
    
    # Enhanced trade flow calculation
    base_trade = (
        np.sqrt(exporter_gdp * importer_gdp) / distance *
        (1 + 0.5 * trade_agreement) *
        (1 + 0.3 * common_language)
    )
    
    trade_flow = base_trade * np.random.lognormal(0, 0.4, n_pairs)
    
    data = pd.DataFrame({
        'Exporter': [pair[0] for pair in country_pairs],
        'Importer': [pair[1] for pair in country_pairs],
        'Exporter_GDP': exporter_gdp,
        'Importer_GDP': importer_gdp,
        'Exporter_Population': exporter_pop,
        'Importer_Population': importer_pop,
        'Distance': distance,
        'Trade_Agreement': trade_agreement,
        'Common_Language': common_language,
        'Trade_Flow': trade_flow
    })
    
    return data

def basic_visualization(data):
    """Create basic visualizations"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trade flow distribution
    axes[0].hist(np.log(data['Trade_Flow']), bins=20, alpha=0.7)
    axes[0].set_title('Trade Flow Distribution (Log)')
    axes[0].set_xlabel('Log(Trade Flow)')
    
    # GDP vs Trade Flow
    axes[1].scatter(data['Exporter_GDP'], data['Trade_Flow'], alpha=0.6)
    axes[1].set_title('Exporter GDP vs Trade Flow')
    axes[1].set_xlabel('Exporter GDP')
    axes[1].set_ylabel('Trade Flow')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def build_enhanced_model(data):
    """Build model with more features"""
    feature_cols = ['Exporter_GDP', 'Importer_GDP', 'Exporter_Population', 
                   'Importer_Population', 'Distance', 'Trade_Agreement', 'Common_Language']
    
    X = data[feature_cols]
    y = np.log(data['Trade_Flow'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Enhanced Model Performance:")
    print(f"R² Score: {r2:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"Features: {len(feature_cols)}")
    
    return model

def main():
    print("Enhanced Trade Flow Prediction Model")
    print("=" * 40)
    
    data = create_enhanced_trade_data()
    print(f"Dataset created with {len(data)} country pairs")
    
    basic_visualization(data)
    model = build_enhanced_model(data)
    
    return data, model

if __name__ == "__main__":
    data, model = main()