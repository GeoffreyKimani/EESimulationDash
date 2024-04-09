import json, os
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, dash_table, State
from ee_imageCol import *
# Functions and variables from each tab file
from constants import DATA_DIR
from step_three import load_features_for_crop, preprocess_features, scale_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ----------------------------------------------------- #
#                   TAB 3 CONTENT                       #
# ----------------------------------------------------- #
def tab_3_content():
    content_style = {
        "width": "60%",
        "margin": "30px auto",
        "padding": "2rem",
        "borderRadius": "8px",
        "border": "1px solid #ccc",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
    }

    dropdown_style = {
        "marginBottom": "20px",
        "border": "1px solid #ccc",
    }

    label_style = {
        "display": "block",
        "marginBottom": "5px",
        "marginTop": "20px",
        "fontSize": "1.2rem",
        "fontWeight": "500",
    }

    box_style = {
        "position": "relative",
        "height": "100%",
        "border": "1px solid #ddd",
        "borderRadius": "5px",
        "padding": "20px",
        "marginTop": "10px",
        "boxShadow": "2px 2px 10px #aaa",
    }

    return html.Div(
        [
            html.Div(
                [
                    html.Button(
                        "Preview Data",
                        id="load-csv-button",
                        className="button-predicted",
                    ),
                    html.Div(
                        id="csv-data-table-container",
                        style={"width": "100%", "overflowX": "auto"},
                    ),
                ],
                style=box_style,
            ),
            html.Div(
                [
                    html.Label("Select Features", style=label_style),
                    dcc.Dropdown(
                        id="features-dropdown",
                        multi=True,
                        placeholder="Select Features",
                        style=dropdown_style,
                    ),
                    html.Div(
                        id="filtered-features-table-container",
                        style={"width": "100%", "overflowX": "auto"},
                    ),
                    html.Button(
                        "Preprocess Data",
                        id="preprocess-button",
                        className="button-predicted",
                    ),
                    html.Div("", id="preprocess-info"),
                ],
                style=box_style,
            ),  # Container for the second data table
            html.Div(
                [
                    html.Label("Model Selection", style=label_style),
                    dcc.Dropdown(
                        id="model-selection-dropdown",
                        options=[
                            {"label": "Random Forest Regressor", "value": "RFR"},
                            {"label": "Linear Regression", "value": "LR"},
                            {"label": "Gradient Boosting Regressor", "value": "GBR"},
                            {"label": "Ridge Regression", "value": "Ridge"},
                            {"label": "Support Vector Regressor", "value": "SVR"},
                            {"label": "Lasso Regression", "value": "Lasso"},
                        ],
                        value="RFR",
                        style=dropdown_style,
                    ),
                    html.Button(
                        "Fit Model", id="fit-model-button", className="button-predicted"
                    ),
                    html.Div(
                        id="model-metrics-output",
                        children="Model fitting results will appear here",
                        style={"whiteSpace": "pre-line", "marginTop": "0px"},
                    ),
                ],
                style=box_style,
            ),
            html.Div(
                [
                    html.Button(
                        "Predict", id="predict-button", className="button-predicted"
                    ),
                    html.Div(
                        id="prediction-metrics-output",
                        children="Prediction results will appear here",
                        style={"whiteSpace": "pre-line", "marginTop": "20px"},
                    ),
                ],
                style=box_style,
            ),
        ],
        style=content_style,
    )


@callback(
    [
        Output("csv-data-table-container", "children"),
        Output("features-df-store", "data"),
    ],
    [Input("load-csv-button", "n_clicks")],
    [State("selected-crop", "data"), State("gdf-data", "data")],
    prevent_initial_call=True,
)
def load_and_display_csv(n_clicks, value, gdf_data):
    if n_clicks:
        df = load_features_for_crop(value)
        print(df.columns)

        # Remove the unnecessary features
        necessary_features = (
            ["district", "pesticide", "year", "season", "plot_hectares", "pest_disease"]
            + [col for col in df.columns if col.endswith("VI")]
            + ["yield_kg_ph"]
        )
        df = df[
            necessary_features
        ]  # keep yield_kg_ph in this or it will cause downstream error in scale_y
        df_display = df[necessary_features[:-2]]  # Remove the yield column for display

        if gdf_data:
            # Parse the GeoJSON to extract districts
            features = json.loads(gdf_data)["features"]
            districts = [
                feature["properties"]["district"]
                for feature in features
                if "district" in feature["properties"]
            ]
            years = [
                feature["properties"]["year"]
                for feature in features
                if "year" in feature["properties"]
            ]

            # Filter the DataFrame based on the extracted districts
            df = df[df["district"].isin(districts)]
            df = df[df["year"].isin(years)]

        # Store the filtered DataFrame in dcc.Store
        filtered_df_json = df.to_json(date_format="iso", orient="split")

        # Display the filtered DataFrame in a DataTable
        data_table = dash_table.DataTable(
            df_display.to_dict("records"),
            [{"name": i, "id": i} for i in df_display.columns],
            style_table={"overflowX": "auto"},
            style_cell={  # General style for each cell
                "minWidth": "150px",
                "width": "150px",
                "maxWidth": "150px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "textAlign": "center",
            },
            style_header={  # Style for header cells
                "backgroundColor": "white",
                "fontWeight": "bold",
                "textAlign": "center",
            },
            style_data={"textAlign": "center"},  # Style for data cells
            page_size=10,
        )

        return data_table, filtered_df_json
    return None, None


@callback(Output("features-dropdown", "options"), [Input("features-df-store", "data")])
def update_features_dropdown(filtered_df_json):
    if filtered_df_json:
        df = pd.read_json(filtered_df_json, orient="split")

        # Remove 'yield_kg_ph' from the list of columns
        del_columns = ["yield_kg_ph", "yield_kg_pa"]
        for col in del_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Ensure you're working with strings; apply .str.strip() to remove any leading/trailing whitespace
        # Then check for non-empty strings across all columns
        # This operation is safe as it converts all types to string before stripping
        non_empty_columns = [
            col for col in df.columns if df[col].astype(str).str.strip().any()
        ]
        print(
            f"All columns: {len(df.columns)} vs Non-empty columns: {len(non_empty_columns)}"
        )

        # Additionally, filter out entirely NaN columns if not already excluded
        non_null_columns = [col for col in non_empty_columns if df[col].notnull().any()]

        # Combine the filters: non-null and non-empty string columns, excluding 'yield_kg_ph'
        final_columns = non_null_columns
        print(f"Final cleaned columns: {len(final_columns)}")

        return [{"label": col, "value": col} for col in final_columns]

    return []


@callback(
    Output("preprocessing-df-store", "data"),
    [Input("features-dropdown", "value")],
    [State("features-df-store", "data")],
)
def update_filtered_features_store(selected_features, original_df_json):
    if selected_features and original_df_json:
        df = pd.read_json(original_df_json, orient="split")

        # Filter the DataFrame to keep only selected features
        filtered_df = df[selected_features]

        # Return the filtered DataFrame as JSON
        return filtered_df.to_json(date_format="iso", orient="split")

    # If no features are selected, return None or keep the original data
    return original_df_json


@callback(
    Output("filtered-features-table-container", "children"),
    [Input("preprocessing-df-store", "data")],
)
def display_filtered_features_table(filtered_df_json):
    if filtered_df_json:
        filtered_df = pd.read_json(filtered_df_json, orient="split")

        # Create and return a DataTable for the filtered DataFrame
        return dash_table.DataTable(
            data=filtered_df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in filtered_df.columns],
            style_table={"overflowX": "auto"},
            style_cell={  # General style for each cell
                "minWidth": "80px",
                "width": "80px",
                "maxWidth": "80px",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "textAlign": "center",
            },
            style_header={  # Style for header cells
                "backgroundColor": "white",
                "fontWeight": "bold",
                "textAlign": "center",
            },
            style_data={"textAlign": "center"},  # Style for data cells
            page_size=10,  # Adjust as needed
        )

    # If there's no data, return an empty div or a message
    return html.Div("No data has been selected.")


@callback(
    [
        Output("modeling-df-store", "data"),
        Output("preprocess-info", "children"),
    ],  # Assuming you have dcc.Store to hold preprocessed data
    [Input("preprocess-button", "n_clicks")],
    [
        State("preprocessing-df-store", "data"),
        State("features-df-store", "data"),
    ],  # Your stored DataFrame with selected features
)
def preprocess_data(n_clicks, features_df_json, all_features_df):
    if n_clicks and features_df_json:
        X_preprocessed = preprocess_features(features_df_json)
        print("X processing done ...")

        print("processing y")
        y_scaled = scale_y(all_features_df)

        # Properly serialize both X and y into JSON
        preprocessed_data_json = json.dumps(
            {"X": X_preprocessed.tolist(), "y": y_scaled.tolist()}
        )

        print("Saving preprocessed data to store")
        return preprocessed_data_json, "Data preprocessed and ready for modeling."
    return None, None


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
import json
import numpy as np
from joblib import dump, load


@callback(
    Output("model-metrics-output", "children"),  # Output to display model metrics
    [Input("fit-model-button", "n_clicks")],
    [
        State("model-selection-dropdown", "value"),
        State("modeling-df-store", "data"),
    ],  # Preprocessed data
)
def fit_model(n_clicks, selected_model, preprocessed_data_json):
    if n_clicks and preprocessed_data_json:
        print("modeling")
        data = json.loads(preprocessed_data_json)
        X = np.array(data["X"])
        y = np.array(data["y"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # remove nan values in y
        mask = ~np.isnan(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        # Select the model based on dropdown selection
        if selected_model == "LR":
            model = LinearRegression()
        elif selected_model == "RFR":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif selected_model == "GBR":
            model = GradientBoostingRegressor(random_state=42)
        elif selected_model == "Ridge":
            model = Ridge(random_state=42)
        elif selected_model == "SVR":
            model = SVR()
        elif selected_model == "Lasso":
            model = Lasso(random_state=42)

        # Fit the model
        model.fit(X_train, y_train)

        # Instead of returning the model, return a message indicating success
        model_message = f"Model {selected_model} fitted. Use 'Predict' to evaluate."

        # Save the fitted model to disk
        model_filename = f"model_{selected_model}.pkl"
        model_path = os.path.join(DATA_DIR, model_filename)
        print(f"model file: {model_path}")
        dump(model, model_path)

        print(model_message)
        return model_message
    return html.Div("No model has been fit yet.")


@callback(
    Output("test-data-store", "data"),  # dcc.Store to hold test data
    [Input("fit-model-button", "n_clicks")],
    [State("model-selection-dropdown", "value"), State("modeling-df-store", "data")],
)
def store_test_data(n_clicks, selected_model, preprocessed_data_json):
    if n_clicks and preprocessed_data_json:
        data = json.loads(preprocessed_data_json)
        X = np.array(data["X"])
        y = np.array(data["y"])

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Convert test data to list for JSON serialization
        test_data_json = json.dumps(
            {"X_test": X_test.tolist(), "y_test": y_test.tolist()}
        )

        print("Test data set")
        return test_data_json
    return None


@callback(
    Output(
        "prediction-metrics-output", "children"
    ),  # Output for displaying prediction metrics
    [Input("predict-button", "n_clicks")],
    [
        State("test-data-store", "data"),
        State("model-selection-dropdown", "value"),
        State("features-df-store", "data"),
    ],  # Assuming test data is stored here
)
def predict_and_evaluate(n_clicks, test_data_json, selected_model, all_features_df):
    if n_clicks:
        print("Evaluating")
        # Deserialize test data
        test_data = json.loads(test_data_json)
        X_test = np.array(test_data["X_test"])
        y_test = np.array(test_data["y_test"])

        print("X, y test data found")

        # Load the model
        model_filename = f"model_{selected_model}.pkl"
        model_path = os.path.join(DATA_DIR, model_filename)

        print(f"Path 2 model: {model_path}")

        model = load(model_path)
        print("Model loaded")

        # Perform predictions
        y_pred = model.predict(X_test)
        print("predictions made")

        # Convert y_pred to its default values
        y_pred_inv = scale_y(all_features_df, y_pred=y_pred)
        print("Actual y_pred", y_pred_inv)
        y_test_inv = scale_y(all_features_df, y_test)
        print(y_test_inv)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        rmse_norm = rmse / (np.max(y_test_inv) - np.min(y_test_inv))
        percent_error = 100 * (1 - rmse_norm)
        std_dev = np.std(y_pred_inv)
        mean = np.mean(y_pred_inv)

        metrics_acc = f"Model Accuracy: {round(percent_error)}%"
        metrics_others = f"Root Mean Squared Error: {round(rmse,2)}kg/ph\nStandard Deviation: {round(std_dev,2)}kg/ph \nMean Crop Yield: {round(mean,2)}kg/ph"

        metrics = html.Div(
            [
                html.Div(metrics_acc),
                dbc.Button(
                    "See Other Metrics",
                    id="collapse-button",
                    className="button-predicted",
                    n_clicks=0,
                ),
                dbc.Collapse(
                    dbc.Card(dbc.CardBody(metrics_others)),
                    id="collapse",
                    is_open=False,
                ),
            ]
        )

        metrics_message = html.Div(
            metrics,
            style={
                "position": "relative",
                "height": "100%",
                "border": "2px solid #ddd",
                "borderRadius": "15px",
                "padding": "20px",
                "boxShadow": "2px 2px 10px #aaa",
            },
        )
        print(metrics_message)
        return metrics_message
    return html.Div(
        "No predictions made yet.",
        style={
            "position": "relative",
            "height": "100%",
            "border": "2px solid #ddd",
            "borderRadius": "15px",
            "padding": "20px",
            "boxShadow": "2px 2px 10px #aaa",
        },
    )


@callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
