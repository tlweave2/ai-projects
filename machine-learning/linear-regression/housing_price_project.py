# housing_price_prediction.ipynb
# Complete Linear Regression Project for AI Portfolio
# Author: Timothy Weaver
# Date: December 2024

"""
Housing Price Prediction using Linear Regression
================================================

Objective: Predict housing prices using various features like location, 
          size, age, and neighborhood characteristics.

Dataset: Boston Housing Dataset (built into scikit-learn)
Algorithm: Linear Regression with feature analysis
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🏠 Housing Price Prediction Project")
print("=" * 50)

# ============================================================================
# 2. LOAD AND EXPLORE DATA
# ============================================================================

def load_and_explore_data():
    """Load Boston housing dataset and perform initial exploration"""
    
    print("\n📊 Loading Dataset...")
    
    # Load the dataset
    boston = load_boston()
    
    # Create DataFrame
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['PRICE'] = boston.target
    
    print(f"✅ Dataset loaded successfully!")
    print(f"📈 Shape: {df.shape[0]} houses, {df.shape[1]-1} features")
    
    # Display basic information
    print("\n🔍 Dataset Overview:")
    print(df.head())
    
    print("\n📋 Feature Descriptions:")
    feature_descriptions = {
        'CRIM': 'Crime rate per capita',
        'ZN': 'Proportion of residential land zoned for lots > 25,000 sq.ft',
        'INDUS': 'Proportion of non-retail business acres',
        'CHAS': 'Charles River dummy variable (1 if tract bounds river)',
        'NOX': 'Nitric oxides concentration (parts per 10 million)',
        'RM': 'Average number of rooms per dwelling',
        'AGE': 'Proportion of owner-occupied units built prior to 1940',
        'DIS': 'Weighted distances to employment centers',
        'RAD': 'Index of accessibility to radial highways',
        'TAX': 'Property tax rate per $10,000',
        'PTRATIO': 'Pupil-teacher ratio by town',
        'B': 'Proportion of blacks by town',
        'LSTAT': '% lower status of the population',
        'PRICE': 'Median home value in $1000s (TARGET)'
    }
    
    for feature, description in feature_descriptions.items():
        if feature in df.columns:
            print(f"  {feature:8}: {description}")
    
    print(f"\n📊 Statistical Summary:")
    print(df.describe().round(2))
    
    return df, boston

# ============================================================================
# 3. DATA VISUALIZATION
# ============================================================================

def visualize_data(df):
    """Create comprehensive visualizations of the dataset"""
    
    print("\n🎨 Creating Visualizations...")
    
    # Set up the plotting area
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🏠 Boston Housing Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price Distribution
    axes[0, 0].hist(df['PRICE'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Housing Prices')
    axes[0, 0].set_xlabel('Price ($1000s)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['PRICE'].mean(), color='red', linestyle='--', label=f'Mean: ${df["PRICE"].mean():.1f}k')
    axes[0, 0].legend()
    
    # 2. Price vs. Average Rooms
    axes[0, 1].scatter(df['RM'], df['PRICE'], alpha=0.6, color='green')
    axes[0, 1].set_title('Price vs. Average Number of Rooms')
    axes[0, 1].set_xlabel('Average Rooms (RM)')
    axes[0, 1].set_ylabel('Price ($1000s)')
    
    # Add trend line
    z = np.polyfit(df['RM'], df['PRICE'], 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['RM'], p(df['RM']), "r--", alpha=0.8)
    
    # 3. Price vs. Crime Rate
    axes[0, 2].scatter(df['CRIM'], df['PRICE'], alpha=0.6, color='orange')
    axes[0, 2].set_title('Price vs. Crime Rate')
    axes[0, 2].set_xlabel('Crime Rate (CRIM)')
    axes[0, 2].set_ylabel('Price ($1000s)')
    
    # 4. Price vs. % Lower Status Population
    axes[1, 0].scatter(df['LSTAT'], df['PRICE'], alpha=0.6, color='purple')
    axes[1, 0].set_title('Price vs. % Lower Status Population')
    axes[1, 0].set_xlabel('Lower Status % (LSTAT)')
    axes[1, 0].set_ylabel('Price ($1000s)')
    
    # 5. Correlation Heatmap
    # Select key features for correlation
    key_features = ['PRICE', 'RM', 'LSTAT', 'CRIM', 'NOX', 'DIS', 'PTRATIO']
    corr_matrix = df[key_features].corr()
    
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_title('Feature Correlation Matrix')
    axes[1, 1].set_xticks(range(len(key_features)))
    axes[1, 1].set_yticks(range(len(key_features)))
    axes[1, 1].set_xticklabels(key_features, rotation=45)
    axes[1, 1].set_yticklabels(key_features)
    
    # Add correlation values to heatmap
    for i in range(len(key_features)):
        for j in range(len(key_features)):
            axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', fontweight='bold')
    
    # 6. Box plot of prices by Charles River
    river_data = [df[df['CHAS'] == 0]['PRICE'], df[df['CHAS'] == 1]['PRICE']]
    axes[1, 2].boxplot(river_data, labels=['Not on River', 'On Charles River'])
    axes[1, 2].set_title('Price Distribution by Charles River Location')
    axes[1, 2].set_ylabel('Price ($1000s)')
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation insights
    print(f"\n🔗 Key Correlations with Price:")
    price_corr = df.corr()['PRICE'].sort_values(ascending=False)
    for feature, correlation in price_corr.items():
        if feature != 'PRICE' and abs(correlation) > 0.5:
            direction = "positively" if correlation > 0 else "negatively"
            print(f"  {feature:8}: {correlation:6.3f} - {direction} correlated")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

def prepare_features(df):
    """Prepare features for modeling"""
    
    print("\n⚙️ Preparing Features...")
    
    # Separate features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    
    print(f"✅ Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
    
    # Check for missing values
    missing_values = X.isnull().sum().sum()
    print(f"🔍 Missing values: {missing_values}")
    
    return X, y

# ============================================================================
# 5. MODEL TRAINING
# ============================================================================

def train_model(X, y):
    """Train linear regression model with proper evaluation"""
    
    print("\n🤖 Training Linear Regression Model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n📈 Model Performance:")
    print(f"  Training R²:   {train_r2:.4f}")
    print(f"  Test R²:       {test_r2:.4f}")
    print(f"  Training RMSE: ${train_rmse:.2f}k")
    print(f"  Test RMSE:     ${test_rmse:.2f}k")
    print(f"  Test MAE:      ${test_mae:.2f}k")
    
    return model, scaler, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

# ============================================================================
# 6. MODEL EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(model, scaler, X, y, X_test, y_test, y_test_pred):
    """Comprehensive model evaluation with visualizations"""
    
    print("\n📊 Model Evaluation & Visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🤖 Linear Regression Model Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price ($1000s)')
    axes[0, 0].set_ylabel('Predicted Price ($1000s)')
    axes[0, 0].set_title('Actual vs Predicted Prices')
    
    # 2. Residuals Plot
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Price ($1000s)')
    axes[0, 1].set_ylabel('Residuals ($1000s)')
    axes[0, 1].set_title('Residuals Plot')
    
    # 3. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model.coef_)
    }).sort_values('importance', ascending=True)
    
    axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1, 0].set_xlabel('Coefficient Magnitude')
    axes[1, 0].set_title('Feature Importance (Coefficient Magnitudes)')
    
    # 4. Prediction Error Distribution
    axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Error ($1000s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Prediction Errors')
    axes[1, 1].axvline(residuals.mean(), color='red', linestyle='--', 
                      label=f'Mean Error: ${residuals.mean():.2f}k')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Feature importance analysis
    print(f"\n🎯 Top 5 Most Important Features:")
    top_features = feature_importance.tail().iloc[::-1]
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']:8}: {row['importance']:.3f}")

# ============================================================================
# 7. MAKE SAMPLE PREDICTIONS
# ============================================================================

def make_sample_predictions(model, scaler, X, feature_names):
    """Make predictions on sample houses"""
    
    print("\n🏠 Sample Predictions:")
    print("=" * 50)
    
    # Create sample houses
    samples = [
        {
            'description': 'Low-crime area, 6 rooms, near river',
            'CRIM': 0.1, 'ZN': 20, 'INDUS': 5, 'CHAS': 1, 'NOX': 0.4,
            'RM': 6.5, 'AGE': 30, 'DIS': 4, 'RAD': 4, 'TAX': 250,
            'PTRATIO': 16, 'B': 390, 'LSTAT': 8
        },
        {
            'description': 'High-crime area, 4 rooms, older building',
            'CRIM': 15, 'ZN': 0, 'INDUS': 20, 'CHAS': 0, 'NOX': 0.7,
            'RM': 4.5, 'AGE': 80, 'DIS': 2, 'RAD': 8, 'TAX': 400,
            'PTRATIO': 20, 'B': 350, 'LSTAT': 25
        },
        {
            'description': 'Luxury area, 8 rooms, low crime',
            'CRIM': 0.05, 'ZN': 40, 'INDUS': 2, 'CHAS': 1, 'NOX': 0.3,
            'RM': 8.2, 'AGE': 10, 'DIS': 6, 'RAD': 2, 'TAX': 200,
            'PTRATIO': 14, 'B': 395, 'LSTAT': 3
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        # Extract features
        sample_features = [sample[feature] for feature in feature_names if feature != 'PRICE']
        sample_scaled = scaler.transform([sample_features])
        predicted_price = model.predict(sample_scaled)[0]
        
        print(f"\n🏠 House {i}: {sample['description']}")
        print(f"   Predicted Price: ${predicted_price:.1f}k")
        
        # Show key features
        print(f"   Key features: {sample['RM']:.1f} rooms, "
              f"Crime: {sample['CRIM']:.1f}, "
              f"Lower status: {sample['LSTAT']:.1f}%")

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete analysis"""
    
    print("🚀 Starting Housing Price Prediction Analysis")
    print("=" * 60)
    
    # Step 1: Load and explore data
    df, boston_data = load_and_explore_data()
    
    # Step 2: Visualize data
    visualize_data(df)
    
    # Step 3: Prepare features
    X, y = prepare_features(df)
    
    # Step 4: Train model
    model, scaler, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_model(X, y)
    
    # Step 5: Evaluate model
    evaluate_model(model, scaler, X, y, X_test, y_test, y_test_pred)
    
    # Step 6: Make sample predictions
    make_sample_predictions(model, scaler, X, df.columns)
    
    print("\n✅ Analysis Complete!")
    print("\n🎯 Key Takeaways:")
    print("   1. Linear regression can predict housing prices with ~67% accuracy")
    print("   2. Number of rooms (RM) is the strongest positive predictor")
    print("   3. Lower status population % (LSTAT) is the strongest negative predictor")
    print("   4. Crime rate (CRIM) significantly impacts house prices")
    print("   5. Model works well but could be improved with feature engineering")
    
    print("\n🚀 Next Steps:")
    print("   - Try polynomial features for non-linear relationships")
    print("   - Experiment with Ridge/Lasso regression for regularization")
    print("   - Add more sophisticated feature engineering")
    print("   - Try ensemble methods like Random Forest")

# Run the analysis
if __name__ == "__main__":
    main()