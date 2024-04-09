import json, dash
import pandas as pd
from dash import dcc, html, Input, Output, callback, dash_table, State
import geopandas as gpd
from shapely.geometry import Polygon
from components.load_extract_components import (
    layout_container2,
)
from src.utils.ee_imageCol import *

# Functions and variables from each tab file
from constants import district_shape_file
from src.utils.step_one import (
    load_data_for_crop,
    plot_plots_in_data,
    plot_districts_with_plotly,
    hectares_to_square_edges,
)

# ----------------------------------------------------- #
#                   DATA EXPLORATION CONTENT            #
# ----------------------------------------------------- #


# Separate functions for each tab's content
def data_exploration_tab():
    return html.Div(
        [
            html.Div(
                [
                    dcc.Store(id="aggregated-data-store1"),
                    layout_container2,
                    # table_and_button_container,
                ],
                style={
                    "position": "relative",
                    "height": "100%",
                    "border": "2px solid #ddd",
                    "borderRadius": "15px",
                    "padding": "20px",
                    "boxShadow": "2px 2px 10px #aaa",
                },
            ),
        ]
    )


@callback(
    [
        Output("dynamic-input-container", "children"),
        Output("csv-data-table", "children"),
        Output("selected-crop", "data"),
    ],
    [
        Input("crop-selection-dropdown", "value"),
        Input("district-selection-dropdown", "value"),
        Input("year-selection-dropdown", "value"),
    ],
    [
        State("crop-selection-dropdown", "value")
    ],  # State allows us to pass in additional values without triggering the callback
)
def update_inputs_and_display_csv(crop, selected_districts, selected_years, crop_state):
    df_filtered = load_data_for_crop(
        crop_state
    )  # Using crop_state here to avoid confusion with the callback trigger
    districts = df_filtered["district"].unique()
    years = df_filtered["year"].unique()

    # Filtering logic
    if selected_districts and "all" not in selected_districts:
        df_filtered = df_filtered[df_filtered["district"].isin(selected_districts)]
    if selected_years:
        df_filtered = df_filtered[df_filtered["year"].isin(selected_years)]

    # Generate district and year dropdowns
    district_dropdown = html.Div(
        [
            html.Label(
                "Select District:", style={"fontSize": 20, "marginBottom": "10px"}
            ),
            dcc.Dropdown(
                id="district-selection-dropdown",
                options=[{"label": "All", "value": "all"}]
                + [{"label": district, "value": district} for district in districts],
                value=selected_districts or ["all"],
                multi=True,
                style={"width": "100%", "marginBottom": "20px"},
            ),
        ]
    )

    year_dropdown = html.Div(
        [
            html.Label("Select Year:", style={"fontSize": 20, "marginBottom": "10px"}),
            dcc.Dropdown(
                id="year-selection-dropdown",
                options=[{"label": year, "value": year} for year in years],
                value=selected_years if selected_years else [],
                multi=True,
                style={"width": "100%", "marginBottom": "20px"},
            ),
        ]
    )

    # Data table for displaying filtered data
    data_table = dash_table.DataTable(
        data=df_filtered.to_dict("records"),
        columns=[{"name": i, "id": i} for i in df_filtered.columns],
        style_table={"overflowX": "auto", "width": "100%"},
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
        page_size=10,  # Adjust as per your requirement
    )

    return [district_dropdown, year_dropdown], data_table, crop_state


@callback(
    Output("stored-data", "data"),  # Store data in dcc.Store
    [
        Input(
            "crop-selection-dropdown", "value"
        ),  # Plus other inputs that affect the DataFrame
        Input("district-selection-dropdown", "value"),
        Input("year-selection-dropdown", "value"),
    ],
)
def filter_data_and_store(crop, selected_districts, selected_years):
    # Load the full dataset for the selected crop
    full_data = load_data_for_crop(crop)

    # If selected_districts or selected_years is None (e.g., nothing is selected), set them to empty lists
    selected_districts = selected_districts or []
    selected_years = selected_years or []

    # Apply district filtering if 'all' is not selected
    if selected_districts and "all" not in selected_districts:
        df_filtered = full_data[full_data["district"].isin(selected_districts)]
    else:
        df_filtered = full_data

    # Apply year filtering
    if selected_years:
        df_filtered = df_filtered[df_filtered["year"].isin(selected_years)]

    # Convert the filtered DataFrame to JSON and return
    return df_filtered.to_json(date_format="iso", orient="split")


@callback(
    Output("districts-map-container", "children"),
    [Input("stored-data", "data")],  # Triggered when stored-data updates
)
def show_districts_on_data_load(stored_data):
    if stored_data:
        df = pd.read_json(stored_data, orient="split")
        selected_districts = df["district"].dropna().unique().tolist()
        # Load your district geometries GeoDataFrame
        gdf = gpd.read_file(district_shape_file)
        fig = plot_districts_with_plotly(gdf, selected_districts)
        return dcc.Graph(figure=fig)
    return html.Div()  # Return empty if no data


@callback(
    Output("plots-map-container", "children"),
    [Input("stored-data", "data")],  # Triggered when stored-data updates
)
def show_plots_on_data_load(stored_data):
    if stored_data:
        df = pd.read_json(stored_data, orient="split")
        encoded_image = plot_plots_in_data(df)
        image_style = {
            "max-width": "100%",
            "max-height": "100%",
            "width": "100%",
            "height": "100%",  # Maintain aspect ratio
        }
        return html.Img(src=f"data:image/png;base64,{encoded_image}", style=image_style)
    return html.Div()  # Return empty if no data


@callback(
    Output("gdf-data", "data"),
    [Input("btn-plots-box", "n_clicks")],
    [State("stored-data", "data")],
)
def show_plots_box(n_clicks, stored_data):
    if n_clicks > 0:
        df = pd.read_json(stored_data, orient="split")

        # Apply the function to each row in the dataframe to create a polygon for each plot
        df["geometry"] = df.apply(
            lambda row: Polygon(
                hectares_to_square_edges(
                    row["field_longitude"], row["field_latitude"], row["plot_hectares"]
                )
            ),
            axis=1,
        )
        # Convert the dataframe to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        # Instead of directly returning an image, save the modified DataFrame to 'modified-data'
        return gdf.to_json()  # Convert GeoDataFrame to JSON for storage

    return dash.no_update  # Use dash.no_update when there's no update to the store


@callback(
    Output("plots-map-box", "children"),
    [Input("gdf-data", "data"), Input("btn-plots-box", "n_clicks")],
)
def update_image(gdf_data, n_clicks):
    if gdf_data and n_clicks > 0:
        # Deserialize the JSON string into a dictionary
        gdf_dict = json.loads(gdf_data)

        # Create a GeoDataFrame from the 'features' key of the GeoJSON dictionary
        gdf = gpd.GeoDataFrame.from_features(gdf_dict["features"], crs="EPSG:4326")

        encoded_image = plot_plots_in_data(
            gdf
        )  # Assuming this function now accepts a GeoDataFrame
        return html.Img(src=f"data:image/png;base64,{encoded_image}")
    return html.Div()
