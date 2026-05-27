"""
ADVANCED SALES FORECASTING ML PROJECT
Fixed Version

Features:
✅ JSON save fixed
✅ Graph save fixed
✅ 2026 Forecast
✅ Better visualization
✅ Auto close graph
✅ Error handling added
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import json
import warnings

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────

def generate_sales_data(n_months=72):

    np.random.seed(42)

    dates = pd.date_range(
        start='2021-01-01',
        periods=n_months,
        freq='MS'
    )

    # Growth trend
    trend = np.linspace(
        50000,
        180000,
        n_months
    )

    # Seasonality
    seasonality = 18000 * np.sin(
        2 * np.pi * (np.arange(n_months) - 3) / 12
    )

    # Festival effect
    festival_boost = np.zeros(n_months)

    for i, d in enumerate(dates):

        if d.month in [10, 11]:
            festival_boost[i] = 30000

        if d.month == 12:
            festival_boost[i] += 20000

    # Noise
    noise = np.random.normal(
        0,
        7000,
        n_months
    )

    sales = (
        trend
        + seasonality
        + festival_boost
        + noise
    )

    sales = np.maximum(sales, 15000)

    df = pd.DataFrame({

        'date': dates,

        'sales': sales.astype(int),

        'month': dates.month,

        'year': dates.year,

        'quarter': dates.quarter,

        'advertising_spend':
            np.random.randint(
                5000,
                30000,
                n_months
            ),

        'num_products':
            np.random.randint(
                80,
                250,
                n_months
            ),

        'discount_pct':
            np.random.uniform(
                0,
                35,
                n_months
            ).round(1),

        'is_festival':
            dates.month.isin(
                [10, 11, 12]
            ).astype(int)
    })

    # Lag features
    df['sales_lag1'] = df['sales'].shift(1)
    df['sales_lag2'] = df['sales'].shift(2)
    df['sales_lag3'] = df['sales'].shift(3)

    # Rolling averages
    df['rolling_avg_3'] = (
        df['sales'].rolling(3).mean()
    )

    df['rolling_avg_6'] = (
        df['sales'].rolling(6).mean()
    )

    return df.dropna()


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def prepare_features(df):

    features = [

        'month',
        'quarter',
        'year',

        'advertising_spend',

        'num_products',

        'discount_pct',

        'is_festival',

        'sales_lag1',
        'sales_lag2',
        'sales_lag3',

        'rolling_avg_3',
        'rolling_avg_6'
    ]

    X = df[features]

    y = df['sales']

    return X, y, features


# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────

def train_models(X_train, y_train):

    models = {

        'Linear Regression':
            LinearRegression(),

        'Random Forest':
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

        'Gradient Boosting':
            GradientBoostingRegressor(
                n_estimators=200,
                random_state=42
            )
    }

    trained = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        trained[name] = model

        print(f"✅ Trained: {name}")

    return trained


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_models(models, X_test, y_test):

    results = {}

    for name, model in models.items():

        predictions = model.predict(X_test)

        results[name] = {

            'MAE': round(
                mean_absolute_error(
                    y_test,
                    predictions
                ),
                2
            ),

            'RMSE': round(
                np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                ),
                2
            ),

            'R2': round(
                r2_score(
                    y_test,
                    predictions
                ),
                4
            )
        }

    return results


# ─────────────────────────────────────────────
# FUTURE FORECAST
# ─────────────────────────────────────────────

def forecast_future(best_model, df, n_future=12):

    forecasts = []

    current_sales = (
        df['sales']
        .values[-6:]
        .tolist()
    )

    last_row = df.iloc[-1]

    last_date = df['date'].iloc[-1]

    for i in range(1, n_future + 1):

        future_date = (
            last_date
            + pd.DateOffset(months=i)
        )

        lag1 = current_sales[-1]
        lag2 = current_sales[-2]
        lag3 = current_sales[-3]

        roll3 = np.mean(current_sales[-3:])
        roll6 = np.mean(current_sales[-6:])

        future_features = pd.DataFrame([{

            'month':
                future_date.month,

            'quarter':
                future_date.quarter,

            'year':
                future_date.year,

            'advertising_spend':
                int(
                    last_row['advertising_spend']
                    * 1.03
                ),

            'num_products':
                int(
                    last_row['num_products']
                    * 1.02
                ),

            'discount_pct':
                last_row['discount_pct'],

            'is_festival':
                1 if future_date.month
                in [10, 11, 12]
                else 0,

            'sales_lag1': lag1,
            'sales_lag2': lag2,
            'sales_lag3': lag3,

            'rolling_avg_3': roll3,
            'rolling_avg_6': roll6
        }])

        prediction = best_model.predict(
            future_features
        )[0]

        forecasts.append({

            'date':
                str(future_date.date()),

            'month':
                future_date.strftime('%b %Y'),

            'predicted_sales':
                int(prediction)
        })

        current_sales.append(prediction)

    return forecasts


# ─────────────────────────────────────────────
# GRAPH VISUALIZATION
# ─────────────────────────────────────────────

def plot_graph(df, forecasts):

    plt.figure(figsize=(15, 6))

    # Historical sales
    plt.plot(
        df['date'],
        df['sales'],
        linewidth=3,
        label='Historical Sales'
    )

    # Forecast sales
    future_dates = pd.to_datetime(
        [f['date'] for f in forecasts]
    )

    future_sales = [
        f['predicted_sales']
        for f in forecasts
    ]

    plt.plot(
        future_dates,
        future_sales,
        marker='o',
        linestyle='--',
        linewidth=3,
        label='Forecast Sales'
    )

    plt.title(
        'Sales Forecasting 2021 - 2026',
        fontsize=18
    )

    plt.xlabel('Date')
    plt.ylabel('Sales')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    # SAVE GRAPH
    plt.savefig(
        'sales_forecast_graph.png'
    )

    # IMPORTANT FIX
    plt.close()

    print("✅ Graph saved successfully")


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def main():

    try:

        print("\n🚀 ADVANCED SALES FORECASTING")
        print("=" * 60)

        # Generate data
        df = generate_sales_data(72)

        print(
            f"\n📊 Dataset Size: {len(df)}"
        )

        # Prepare features
        X, y, features = prepare_features(df)

        # Split data
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                shuffle=False
            )
        )

        print(
            f"🔀 Train: {len(X_train)}"
        )

        print(
            f"🧪 Test : {len(X_test)}"
        )

        # Train models
        print("\n📈 Training Models\n")

        models = train_models(
            X_train,
            y_train
        )

        # Evaluate
        results = evaluate_models(
            models,
            X_test,
            y_test
        )

        print("\n📊 MODEL PERFORMANCE")
        print("=" * 60)

        for name, metrics in results.items():

            print(
                f"\n{name}"
            )

            print(
                f"MAE  : ₹{metrics['MAE']:,.0f}"
            )

            print(
                f"RMSE : ₹{metrics['RMSE']:,.0f}"
            )

            print(
                f"R²   : {metrics['R2']}"
            )

        # Best model
        best_name = max(
            results,
            key=lambda x:
                results[x]['R2']
        )

        best_model = models[best_name]

        print(
            f"\n🏆 Best Model: {best_name}"
        )

        # Forecast
        forecasts = forecast_future(
            best_model,
            df,
            n_future=12
        )

        print(
            "\n🔮 2026 Forecast"
        )

        print("=" * 60)

        for f in forecasts:

            print(
                f"{f['month']} "
                f"₹{f['predicted_sales']:,.0f}"
            )

        # Save graph
        plot_graph(df, forecasts)

        # JSON Output
        output = {

            'best_model':
                best_name,

            'forecast':
                forecasts,

            'model_results':
                results
        }

        print(
            "\n💾 Saving results.json..."
        )

        # SAVE JSON
        with open(
            'results.json',
            'w'
        ) as file:

            json.dump(
                output,
                file,
                indent=4
            )

        print(
            "✅ results.json saved"
        )

        print(
            "✅ sales_forecast_graph.png saved"
        )

        print(
            "\n🔥 PROJECT COMPLETED"
        )

    except Exception as e:

        print(
            "\n❌ ERROR OCCURRED:"
        )

        print(e)


# ─────────────────────────────────────────────

if __name__ == '__main__':

    main()