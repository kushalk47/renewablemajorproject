from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

# === Load Models and Metadata ===
renewable_model = joblib.load('renewable_energy_model.joblib')
non_renewable_model = joblib.load('non_renewable_energy_model.joblib')
growth_model = joblib.load('growth_rate_model.joblib')
encoder = joblib.load('country_encoder.joblib')
renewable_features = joblib.load('renewable_feature_columns.joblib')
non_renewable_features = joblib.load('non_renewable_feature_columns.joblib')
growth_features = joblib.load('growth_feature_columns.joblib')

renewable_model.set_params(device="cpu")
non_renewable_model.set_params(device="cpu")
growth_model.set_params(device="cpu")

# === Load Countries ===
conn = sqlite3.connect("energy_filled.db")
query = "SELECT DISTINCT country FROM filled_energy_data ORDER BY country"
countries = pd.read_sql_query(query, conn)['country'].tolist()
conn.close()

# === Helper Function for Prediction ===
def create_prediction_input_df(target_year, last_year, last_value, target_entity, feature_columns, encoder, model_type, last_growth_rate=None, last_population=None):
    data = {col: 0.0 for col in feature_columns}
    data['year'] = target_year
    data['year_trend'] = (target_year - 1980) / (2050 - 1980)

    if model_type == 'renewable':
        data['lagged_renewables_consumption'] = last_value
    elif model_type == 'non_renewable':
        data['lagged_non_renewable_consumption'] = last_value
    elif model_type == 'growth':
        data['lagged_renewables_consumption'] = last_value
        data['lagged_smoothed_growth_rate'] = last_growth_rate if last_growth_rate is not None else 0.0

    if last_population is not None:
        years_diff = target_year - last_year
        population_growth_rate = 0.01
        data['population'] = last_population * (1 + population_growth_rate) ** years_diff

    entity_df = pd.DataFrame([[target_entity]], columns=['country'])
    entity_encoded = encoder.transform(entity_df)
    encoded_df = pd.DataFrame(entity_encoded, columns=encoder.get_feature_names_out(['country']))
    for col in encoded_df.columns:
        if col in data:
            data[col] = encoded_df[col].iloc[0]

    return pd.DataFrame([data], columns=feature_columns)

# === Route for Home Page ===
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    non_renewable_prediction = None
    predicted_growth_rate = None
    plot_url = None
    error = None

    if request.method == 'POST':
        try:
            country = request.form['country']
            growth_rate = request.form.get('growth_rate')
            growth_rate = float(growth_rate) / 100 if growth_rate else None

            # Load historical data
            conn = sqlite3.connect("energy_filled.db")
            query = """
                SELECT year, renewables_consumption, fossil_fuel_consumption, coal_consumption, oil_consumption, population
                FROM filled_energy_data
                WHERE country = ? AND renewables_consumption IS NOT NULL
                ORDER BY year
            """
            df = pd.read_sql_query(query, conn, params=(country,))
            conn.close()

            if df.empty:
                error = f"No data found for {country}."
                return render_template('index.html', countries=countries, prediction=prediction,
                                     non_renewable_prediction=non_renewable_prediction,
                                     predicted_growth_rate=predicted_growth_rate, plot_url=plot_url, error=error)

            df['non_renewable_consumption'] = df['fossil_fuel_consumption'] + df['coal_consumption'] + df['oil_consumption']
            if df[['renewables_consumption', 'non_renewable_consumption']].isna().any().any() or \
               np.isinf(df[['renewables_consumption', 'non_renewable_consumption']]).any().any():
                error = f"Invalid historical data for {country} (NaN or infinity detected)."
                return render_template('index.html', countries=countries, prediction=prediction,
                                     non_renewable_prediction=non_renewable_prediction,
                                     predicted_growth_rate=predicted_growth_rate, plot_url=plot_url, error=error)

            last_year = int(df.iloc[-1]['year'])
            last_renewable = df.iloc[-1]['renewables_consumption']
            last_non_renewable = df.iloc[-1]['non_renewable_consumption']
            last_population = df.iloc[-1]['population']
            start_year = max(2025, last_year + 1)
            future_years = list(range(start_year, 2051))

            # Compute historical growth rate (smoothed)
            historical_growth = df['renewables_consumption'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
            smoothed_growth = historical_growth.rolling(window=3, min_periods=1).mean()
            last_growth_rate = smoothed_growth.iloc[-1] if len(smoothed_growth) > 1 else 0.0

            # Predict growth rate if not provided
            if growth_rate is None:
                input_df = create_prediction_input_df(
                    target_year=start_year,
                    last_year=last_year,
                    last_value=last_renewable,
                    target_entity=country,
                    feature_columns=growth_features,
                    encoder=encoder,
                    model_type='growth',
                    last_growth_rate=last_growth_rate,
                    last_population=last_population
                )
                growth_rate = growth_model.predict(input_df)[0]
                growth_rate = np.clip(growth_rate, 0.01, 0.1)  # Ensure positive growth, cap at 10%
                predicted_growth_rate = round(growth_rate * 100, 2)

            # Predict renewable and non-renewable consumption
            renewable_consumption = df['renewables_consumption'].tolist()
            non_renewable_consumption = df['non_renewable_consumption'].tolist()
            historical_years = df['year'].tolist()
            current_renewable = last_renewable
            current_non_renewable = last_non_renewable

            for year in future_years:
                # Renewable prediction
                renew_input = create_prediction_input_df(
                    target_year=year,
                    last_year=last_year,
                    last_value=current_renewable,
                    target_entity=country,
                    feature_columns=renewable_features,
                    encoder=encoder,
                    model_type='renewable',
                    last_population=last_population
                )
                renew_pred = renewable_model.predict(renew_input)[0]
                renew_pred = max(renew_pred, current_renewable)  # Prevent unrealistic decline
                renewable_consumption.append(renew_pred)
                current_renewable = renew_pred * (1 + growth_rate) if growth_rate else renew_pred

                # Non-renewable prediction with decay
                non_renew_input = create_prediction_input_df(
                    target_year=year,
                    last_year=last_year,
                    last_value=current_non_renewable,
                    target_entity=country,
                    feature_columns=non_renewable_features,
                    encoder=encoder,
                    model_type='non_renewable',
                    last_population=last_population
                )
                non_renew_pred = non_renewable_model.predict(non_renew_input)[0]
                decay_factor = 0.99  # 1% annual reduction in non-renewable
                non_renew_pred *= (decay_factor ** (year - last_year))
                non_renewable_consumption.append(non_renew_pred)
                current_non_renewable = non_renew_pred

            # Prepare data for plotting
            historical_df = pd.DataFrame({
                'Year': historical_years * 2,
                'Consumption': renewable_consumption[:len(historical_years)] + non_renewable_consumption[:len(historical_years)],
                'Type': ['Historical Renewable'] * len(historical_years) + ['Historical Non-Renewable'] * len(historical_years)
            })
            predicted_df = pd.DataFrame({
                'Year': future_years * 2,
                'Consumption': renewable_consumption[-len(future_years):] + non_renewable_consumption[-len(future_years):],
                'Type': ['Predicted Renewable'] * len(future_years) + ['Predicted Non-Renewable'] * len(future_years)
            })
            plot_data = pd.concat([historical_df, predicted_df], ignore_index=True)

            # Plot and encode as base64
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=plot_data, x='Year', y='Consumption', hue='Type', style='Type',
                         palette={'Historical Renewable': 'blue', 'Predicted Renewable': 'red',
                                  'Historical Non-Renewable': 'green', 'Predicted Non-Renewable': 'orange'})
            plt.title(f'Energy Consumption for {country} (2025-2050)')
            plt.xlabel('Year')
            plt.ylabel('Consumption (TWh)')
            plt.grid(True)
            plt.xticks(rotation=45)

            # Save plot to a bytes buffer and encode as base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plot_url = f'data:image/png;base64,{plot_base64}'
            plt.close()

            prediction = round(renewable_consumption[-1], 2)
            non_renewable_prediction = round(non_renewable_consumption[-1], 2)

        except Exception as e:
            error = f"Error during prediction or plotting: {str(e)}"
            print(f"Error details: {str(e)}")

    return render_template('index.html', countries=countries, prediction=prediction,
                         non_renewable_prediction=non_renewable_prediction,
                         predicted_growth_rate=predicted_growth_rate, plot_url=plot_url, error=error)

if __name__ == '__main__':
    app.run(debug=True)