from ee_init import initialize_ee

initialize_ee()

import json, dash, os
import folium
import tempfile
import pandas as pd
from datetime import date
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
import geopandas as gpd
from shapely.geometry import Polygon

# from dash.dependencies import Input, Output
import tempfile
from ee_imageCol import *
import plotly.graph_objs as go


# Functions and variables from each tab file
from utils import defineROI
from constants import DATA_DIR, district_shape_file
from step_one import (
    load_data_for_crop,
    plot_plots_in_data,
    plot_districts_with_plotly,
    hectares_to_square_edges,
)
from step_two import (
    satellite_dict,
    add_ee_layer,
    integrate_indices_to_dataframe,
    calculate_satellite_dates,
)
from step_three import load_features_for_crop, preprocess_features, scale_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Functions for Yield Gap Tab
from yield_tab import create_rwanda_map, aggregate_data, color_map

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
    ],
)

app.layout = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            value="tab-1",
            children=[
                dcc.Tab(
                    label="Step 1: Load and Extract Fields Data",
                    value="tab-1",
                    id="tab-1",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                ),
                dcc.Tab(
                    label="Step 2: Satellite Data Extraction",
                    value="tab-2",
                    id="tab-2",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                ),
                dcc.Tab(
                    label="Step 3: Prediction",
                    value="tab-3",
                    id="tab-3",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                ),
                dcc.Tab(
                    label="Yield Gap", value="tab-yield-gap"
                ),  # New tab for Yield Gap
            ],
            className="custom-tabs",
        ),
        html.Div(id="tabs-content"),
        # define the data storage globally
        dcc.Store(id="stored-data"),  # To store the filtered DataFrame
        dcc.Store(id="gdf-data"),  # To store the GDF DataFrame
        dcc.Store(id="features-df-store"),  # Stores the features for modeling
        dcc.Store(
            id="preprocessing-df-store"
        ),  # Stores the selected features for processing
        dcc.Store(id="modeling-df-store"),  # Stores the data after preprocessing
        dcc.Store(id="test-data-store"),  # Stores the data for model evaluation
    ],
    style={"textAlign": "center"},
)


# ----------------------------------------------------- #
#                   TAB 1 CONTENT                       #
# ----------------------------------------------------- #
# Separate functions for each tab's content
def tab_1_content():
    return html.Div(
        [
            html.Label("Select Crop:", style={"fontSize": 20, "marginBottom": "10px"}),
            dcc.Dropdown(
                id="crop-selection-dropdown",
                options=[
                    {"label": "Maize", "value": "maize"},
                    {"label": "Potato", "value": "potato"},
                ],
                value="maize",  # Default value
                clearable=False,
                style={"width": "100%", "marginBottom": "20px"},
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="district-selection-dropdown", style={"display": "none"}
                    ),
                    dcc.Dropdown(
                        id="year-selection-dropdown", style={"display": "none"}
                    ),
                ],
                id="dynamic-input-container",
            ),  # Container for dynamic objects
            html.Div(id="csv-data-table"),  # Container for displaying CSV data
            html.Button("Show Districts", id="btn-show-districts", n_clicks=0),
            html.Button("Show Plots", id="btn-show-plots", n_clicks=0),
            html.Button("Create Plots Box", id="btn-plots-box", n_clicks=0),
            html.Div(id="districts-map-container"),  # To display the district map
            html.Div(id="plots-map-container"),  # To display the plots map
            html.Div(id="plots-map-box"),  # To display the plots map
        ],
        style={
            "width": "50%",
            "margin": "0 auto",
            "border": "2px solid #ddd",
            "borderRadius": "15px",
            "padding": "20px",
            "boxShadow": "2px 2px 10px #aaa",
        },
    )


@callback(
    [
        Output("dynamic-input-container", "children"),
        Output("csv-data-table", "children"),
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
        page_size=10,  # Adjust as per your requirement
    )

    return [district_dropdown, year_dropdown], data_table


@app.callback(
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


@app.callback(
    Output("districts-map-container", "children"),
    Input("btn-show-districts", "n_clicks"),
    State("stored-data", "data"),
)
def show_districts(n_clicks, stored_data):
    if n_clicks > 0:
        df = pd.read_json(stored_data, orient="split")
        selected_districts = df["district"].dropna().unique().tolist()
        print(selected_districts)
        # Load your district geometries GeoDataFrame
        gdf = gpd.read_file(district_shape_file)
        fig = plot_districts_with_plotly(gdf, selected_districts)

        return dcc.Graph(figure=fig)
    return html.Div()


@app.callback(
    Output("plots-map-container", "children"),
    [Input("btn-show-plots", "n_clicks")],
    [State("stored-data", "data")],
)
def show_plots(n_clicks, stored_data):
    if n_clicks > 0:
        df = pd.read_json(stored_data, orient="split")

        # Generate the base64-encoded image
        encoded_image = plot_plots_in_data(df)

        # Use the `html.Img` component to display the image directly from the base64 string
        return html.Img(src=f"data:image/png;base64,{encoded_image}")

    # If no image is to be displayed, return an empty `div`
    return html.Div()


@app.callback(
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


@app.callback(
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


# ----------------------------------------------------- #
#                   TAB 2 CONTENT                       #
# ----------------------------------------------------- #
# Define the sidebar layout with controls
def create_sidebar_controls():
    return html.Div(
        [
            # Container for the Satellite name with label
            html.Div(
                [
                    html.Label(
                        "Satellite name:",
                        style={"display": "inline-block", "margin-right": "10px"},
                    ),
                    dcc.Dropdown(
                        id="dropdown-Satellite",
                        options=[
                            {"label": name, "value": value}
                            for name, value in satellite_dict.items()
                        ],
                        value=None,
                        searchable=True,
                        placeholder="Select Satellite...",
                        # labelStyle={"display": "block", "text-align": "justify"},
                        style={"width": "100%", "marginBottom": "20px"},
                    ),
                ],
                style={"margin-bottom": "10px", "padding-left": "20px"},
            ),
            # Container for the District name with label
            html.Div(
                [
                    html.Label(
                        "District name:",
                        style={"display": "inline-block", "margin-right": "10px"},
                    ),
                    dcc.Dropdown(
                        id="dropdown-district",
                        options=[
                            {"label": name, "value": name} for name in district_names
                        ],
                        value=None,
                        searchable=True,
                        placeholder="Select or type your district...",
                        style={"width": "100%", "marginBottom": "20px"},
                    ),
                ],
                style={"margin-bottom": "10px", "padding-left": "20px"},
            ),
            # Container for the Time of Interest with label
            html.Div(
                [
                    html.Label(
                        "Time of Interest:",
                        style={"display": "block", "margin-bottom": "5px"},
                    ),
                    dcc.DatePickerRange(
                        id="time-of-interest",
                        min_date_allowed=date(2000, 1, 1),
                        max_date_allowed=date.today(),
                        initial_visible_month=date.today(),
                        end_date=date.today(),
                        style={"width": "100%"},
                    ),
                ],
                style={"margin-bottom": "10px", "padding-left": "20px"},
            ),
            # Container for the cloud-mask-checkbox
            html.Div(
                [
                    dcc.Checklist(
                        id="cloud-mask-checkbox",
                        options=[{"label": "Mask Clouds", "value": "cloud_mask"}],
                        value=["cloud_mask"],
                        labelStyle={"display": "block", "text-align": "justify"},
                        style={
                            "border": "1px solid #ddd",
                            "padding": "10px",
                            "border-radius": "5px",
                            "padding-left": "20px",
                        },  # Adjust the padding-left as needed
                    ),
                ],
                style={"margin-bottom": "10px", "padding-left": "20px"},
            ),
            # Container for the filter-checkbox
            html.Div(
                [
                    dcc.Checklist(
                        id="filter-checkbox",
                        options=[{"label": "Use MA Filter", "value": "filters"}],
                        value=[],
                        labelStyle={"display": "block", "text-align": "justify"},
                        style={
                            "border": "1px solid #ddd",
                            "padding": "10px",
                            "border-radius": "5px",
                            "padding-left": "20px",
                        },  # Adjust the padding-left as needed
                    ),
                ],
                style={"margin-bottom": "10px", "padding-left": "20px"},
            ),
            # Container for the Spectral indices with label
            html.Div(
                [
                    html.Label(
                        "Spectral indices",
                        style={
                            "display": "inline-block",
                        },
                    ),
                    dcc.Checklist(
                        id="index-checkboxes",
                        options=[
                            {"label": "NDVI", "value": "NDVI"},
                            {"label": "EVI", "value": "EVI"},
                            {"label": "SAVI", "value": "SAVI"},
                            # Add more indices as needed
                        ],
                        value=[],
                        labelStyle={"display": "block", "text-align": "justify"},
                        style={
                            "border": "1px solid #ddd",
                            "padding": "10px",
                            "border-radius": "5px",
                            "padding-left": "20px",
                        },  # Adjust the padding-left as needed
                    ),
                ],
                style={"margin-bottom": "10px", "padding-left": "20px"},
            ),
            # Container for the year-input with label
            html.Div(
                [
                    dcc.Input(
                        id="year-input",
                        type="number",
                        placeholder="Enter year",
                        min=date(2000, 1, 1).year,
                        max=date.today().year,
                        style={
                            "width": "100%",
                            "text-align": "center",
                        },
                    ),
                ],
                style={"margin-bottom": "10px", "padding-left": "20px"},
            ),
            # Container for the buttons aligned with other components
            html.Div(
                [
                    html.Button(
                        "View Indices",
                        id="toggle-indices-btn",
                        n_clicks=0,
                        className="btn btn-outline-primary btn-lg",
                        style={
                            "width": "100%",
                            "marginBottom": "10px",
                        },
                    ),
                    html.Button(
                        "Submit",
                        id="submit-val",
                        n_clicks=0,
                        className="btn btn-outline-success btn-lg",
                        style={
                            "width": "100%",
                            "marginTop": "10px",
                        },  # Added marginTop for spacing
                    ),
                ],
                style={"margin-bottom": "10px", "padding-left": "20px"},
            ),
        ],
        style={
            "padding": "20px",
            "z-index": "1000",  # Apply z-index here for the entire sidebar
            "position": "relative",  # Required for z-index to work
        },
    )


# The main layout function of your app tab content
def tab_2_content():
    sidebar_controls = create_sidebar_controls()

    # Now use these controls in your sidebar layout
    sidebar = html.Div(
        id="sidebar",
        style=SIDEBAR_EXPANDED_STYLE,
        children=[
            html.Button(
                html.I(className="fa fa-chevron-left"),
                # children=[],  # No content needed, we'll use CSS for the arrow icon
                id="toggle-sidebar-button",
                className="toggle-button",
                n_clicks=0,
                style={
                    "position": "absolute",
                    "top": 0,
                    "right": "0",  # Adjusted from -30 to 0
                    "width": "30px",
                    "height": "30px",  # Adjust size as needed
                    "backgroundColor": "transparent",  # Make background transparent
                    "border": "none",  # Remove border
                },
            ),
            html.Div(
                id="sidebar-header",
                children="Map Settings",
                style=SIDEBAR_HEADER_EXPANDED_STYLE,
            ),
            html.Div(
                id="controls-container",
                children=[sidebar_controls],
                style={
                    "display": "block"
                },  # Use 'none' to hide the controls when sidebar is collapsed
            ),
        ],
    )

    satellite_data__tab_accordion = html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    title="Map",
                    children=[
                        html.Div(  # Container for the map
                            id="map-container",
                            children=[
                                html.Iframe(
                                    id="map-iframe",
                                    width="100%",
                                    height="400px",  # Adjust height as needed
                                    style={
                                        "border": "2px solid lightgrey",
                                        "border-radius": "8px",
                                        "zIndex": 0,
                                        "height": "680px",
                                    },
                                ),
                            ],
                            style={
                                "width": "100%",
                                "height": "400px",
                                "position": "relative",
                                "flex": "1",
                            },
                        ),
                    ],
                    item_id="item-1",
                    style={"height": "100%", "width": "100%"},
                ),
            ],
            always_open=True,
            id="accordion-always-open",
            active_item="item-1",  # Set the ID of the item you want to be open by default
            style={"height": "100%", "width": "100%"},
        ),
    )

    layout = html.Div(
        style={
            "display": "flex",
            "flexDirection": "row",
            "height": "100%",  # Ensure it fills the vertical space
        },
        children=[
            sidebar,  # Assuming sidebar is a defined component
            html.Div(
                satellite_data__tab_accordion,
                style={
                    "flexGrow": 1,
                    "flexShrink": 1,
                    "flexBasis": "auto",
                    "height": "100%",
                },  # Allow accordion to grow and fill the space
            ),
        ],
    )

    return html.Div(
        [
            layout,
        ],
        style={
            "position": "relative",
            "height": "100%",
            "border": "2px solid #ddd",
            "borderRadius": "15px",
            "padding": "10px",
            "boxShadow": "2px 2px 10px #aaa",
        },
    )


@app.callback(
    Output("map-iframe", "srcDoc"),
    [Input("submit-val", "n_clicks")],
    [
        dash.dependencies.State("dropdown-Satellite", "value"),
        dash.dependencies.State("dropdown-district", "value"),
        dash.dependencies.State("time-of-interest", "start_date"),
        dash.dependencies.State("time-of-interest", "end_date"),
        dash.dependencies.State("cloud-mask-checkbox", "value"),
        dash.dependencies.State("filter-checkbox", "value"),
        dash.dependencies.State("index-checkboxes", "value"),
    ],
)
def update_map(
    n_clicks,
    satellite,
    district,
    start_date,
    end_date,
    mask_clouds,
    use_filters,
    indices,
):

    # Generate and save the map
    country_lon = 29.8739
    country_lat = -1.9403

    base_map = folium.Map(location=[country_lat, country_lon], zoom_start=10)

    if base_map:
        # Add a layer control panel to the map.
        baseMap = base_map._repr_html_()

        if n_clicks > 0:
            roi = defineAOI(district)
            img_collection = importLandsat(start_date, end_date, roi, data=satellite)
            center = roi.centroid(10).coordinates().reverse().getInfo()
            base_map = folium.Map(location=center, zoom_start=10)

            if mask_clouds:
                # Mask clouds
                img_collection = (
                    img_collection.map(maskCloudLandsat)
                    if satellite.startswith("LANDSAT")
                    else img_collection.map(maskCloudSentinel)
                )

            img_collection = (
                img_collection.map(landsat_scale_factors)
                if satellite.startswith("LANDSAT")
                else img_collection.map(Sentinel_scale_factors)
            )

            if use_filters:
                # Apply filters
                img_collection = img_collection.map(applyMovingAvg)

            cloud_free_composite = img_collection.map(
                lambda img: img.clip(roi)
            ).median()

            min_, max_, _ = (
                imageStats(cloud_free_composite, roi, satellite)
                if satellite.startswith("LANDSAT")
                else (0.0, 0.3, None)
            )
            bands = (
                ["SR_B4", "SR_B3", "SR_B2"]
                if satellite.startswith("LANDSAT")
                else ["B4", "B3", "B2"]
            )

            vis_params = {
                "bands": bands,
                "min": min_,
                "max": max_,
                # 'gamma': 1
            }

            base_map.add_ee_layer(cloud_free_composite, vis_params, "EE-Image", True)

            if "NDVI" in indices:
                cloud_free_composite = (
                    add_NDVI(cloud_free_composite)
                    if satellite.startswith("LANDSAT")
                    else add_NDVIsentinel(cloud_free_composite)
                )
                base_map.add_ee_layer(
                    cloud_free_composite, ndvi_params, "NDVI", True, 0.9
                )

            if "EVI" in indices:
                cloud_free_composite = (
                    add_EVI(cloud_free_composite)
                    if satellite.startswith("LANDSAT")
                    else add_EVIsentinel(cloud_free_composite)
                )
                base_map.add_ee_layer(
                    cloud_free_composite, evi_params, "EVI", True, 0.9
                )

            if "SAVI" in indices:
                cloud_free_composite = (
                    add_SAVI(cloud_free_composite)
                    if satellite.startswith("LANDSAT")
                    else add_SAVIsentinel(cloud_free_composite)
                )
                base_map.add_ee_layer(
                    cloud_free_composite, savi_params, "SAVI", True, 0.9
                )

            baseMap = base_map.add_child(folium.LayerControl())._repr_html_()

        # Save the map to a temporary HTML file
        tmp_html = tempfile.NamedTemporaryFile(suffix=".html").name
        with open(tmp_html, "w") as f:
            f.write(baseMap)

        return open(tmp_html).read()

    else:
        return None


# Add Indices time series plot
@app.callback(
    Output("indices-content", "children"),
    [Input("toggle-indices-btn", "n_clicks")],
    [
        dash.dependencies.State("index-checkboxes", "value"),
        dash.dependencies.State("year-input", "value"),
        dash.dependencies.State("dropdown-Satellite", "value"),
        dash.dependencies.State("dropdown-district", "value"),
    ],
)
def toggle_indices_content(n_clicks, indices, year, satellite, district):
    if n_clicks % 2 == 1 and indices and year:
        roi = defineAOI(district)
        img_collection = importLandsat(
            f"{year}-01-01", f"{year}-12-31", roi, data=satellite
        )
        img_collection = (
            img_collection.map(maskCloudLandsat)
            if satellite.startswith("LANDSAT")
            else img_collection.map(maskCloudSentinel)
        )
        img_collection = (
            img_collection.map(lambda img: img.clip(roi)).map(landsat_scale_factors)
            if satellite.startswith("LANDSAT")
            else img_collection.map(Sentinel_scale_factors)
        )
        indices_dict = {}
        if "NDVI" in indices:
            img_collection = (
                img_collection.map(add_NDVI)
                if satellite.startswith("LANDSAT")
                else img_collection.map(add_NDVIsentinel)
            )
            monthly_ndvi = aggregate_monthly(img_collection, "NDVI")
            monthly_ndvi_dict = extract_values(
                ee.ImageCollection(monthly_ndvi), "NDVI", roi
            ).getInfo()
            indices_dict["NDVI"] = [
                x["properties"]["value"] for x in monthly_ndvi_dict["features"]
            ]

        if "EVI" in indices:
            img_collection = (
                img_collection.map(add_EVI)
                if satellite.startswith("LANDSAT")
                else img_collection.map(add_EVIsentinel)
            )
            monthly_evi = aggregate_monthly(img_collection, "EVI")
            monthly_evi_dict = extract_values(
                ee.ImageCollection(monthly_evi), "EVI", roi
            ).getInfo()
            indices_dict["EVI"] = [
                x["properties"]["value"] for x in monthly_evi_dict["features"]
            ]

        if "SAVI" in indices:
            img_collection = (
                img_collection.map(add_SAVI)
                if satellite.startswith("LANDSAT")
                else img_collection.map(add_SAVIsentinel)
            )
            monthly_savi = aggregate_monthly(img_collection, "SAVI")
            monthly_savi_dict = extract_values(
                ee.ImageCollection(monthly_savi), "SAVI", roi
            ).getInfo()
            indices_dict["SAVI"] = [
                x["properties"]["value"] for x in monthly_savi_dict["features"]
            ]

        try:
            plots = []

            for index in indices:
                # Generate random data for the plot
                x = list(range(1, 13))
                y = indices_dict[index]

                # Create a line plot
                trace = go.Scatter(x=x, y=y, mode="lines", name=f"{index}")

                plots.append(trace)

            return dcc.Graph(
                id="indices-plot",
                figure={
                    "data": plots,
                    "layout": go.Layout(
                        title=f"Indices Time Series for {district} district",
                        xaxis=dict(title="Months"),
                        yaxis=dict(title="Index Value"),
                        hovermode="closest",
                    ),
                },
            )

        except:
            return ["Select Index, satellites, roi, and year"]

    else:
        return []


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
                    html.Label("Selected Crop", style=label_style),
                    dcc.Dropdown(
                        id="selected-crop-dropdown",
                        placeholder="Select Crop",
                        disabled=True,
                        style=dropdown_style,
                    ),
                    html.Button(
                        "Load CSV", id="load-csv-button", className="button-predicted"
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


@app.callback(
    [
        Output("selected-crop-dropdown", "options"),
        Output("selected-crop-dropdown", "value"),
    ],
    [Input("gdf-data", "data")],
)
def update_crop_selection(gdf_data):
    if gdf_data:
        # Load GeoJSON data
        features = json.loads(gdf_data)["features"]

        # Extract properties to a list of dictionaries
        properties_list = [feature["properties"] for feature in features]

        # Convert properties to DataFrame
        properties_df = pd.DataFrame(properties_list)

        # Now, you can proceed as before
        crops = properties_df["crop"].unique().tolist()
        crop_value = crops[0] if crops else None
        options = [{"label": crop, "value": crop} for crop in crops]

        return options, crop_value

    # Return empty options and None value if there's no data
    return [], None


@app.callback(
    [
        Output("csv-data-table-container", "children"),
        Output("features-df-store", "data"),
    ],
    [Input("load-csv-button", "n_clicks")],
    [State("selected-crop-dropdown", "value"), State("gdf-data", "data")],
    prevent_initial_call=True,
)
def load_and_display_csv(n_clicks, value, gdf_data):
    if n_clicks:
        df = load_features_for_crop(value)

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
            df.to_dict("records"),
            [{"name": i, "id": i} for i in df.columns],
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


@app.callback(
    Output("features-dropdown", "options"), [Input("features-df-store", "data")]
)
def update_features_dropdown(filtered_df_json):
    if filtered_df_json:
        df = pd.read_json(filtered_df_json, orient="split")

        # Remove 'yield_kg_ph' from the list of columns
        if "yield_kg_ph" and "yield_kg_pa" in df.columns:
            df = df.drop(columns=["yield_kg_ph", "yield_kg_pa"])

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


@app.callback(
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


@app.callback(
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


@app.callback(
    Output(
        "modeling-df-store", "data"
    ),  # Assuming you have dcc.Store to hold preprocessed data
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
        return preprocessed_data_json
    return None


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
import json
import numpy as np
from joblib import dump, load


@app.callback(
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


@app.callback(
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


@app.callback(
    Output(
        "prediction-metrics-output", "children"
    ),  # Output for displaying prediction metrics
    [Input("predict-button", "n_clicks")],
    [
        State("test-data-store", "data"),
        State("model-selection-dropdown", "value"),
    ],  # Assuming test data is stored here
)
def predict_and_evaluate(n_clicks, test_data_json, selected_model):
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
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        std_dev = np.std(y_test)
        mean = np.mean(y_test)

        metrics_message = f"Root Mean Squared Error: {rmse} \nStandard deviation: {std_dev} \nMean: {mean}"
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


# ----------------------------------------------------- #
#                   YIELD GAP TAB                       #
# ----------------------------------------------------- #

from components.yield_tab_components import (
    layout_container,
    table_and_link_container,
    SIDEBAR_EXPANDED_STYLE,
    SIDEBAR_COLLAPSED_STYLE,
    SIDEBAR_HEADER_COLLAPSED_STYLE,
    SIDEBAR_HEADER_EXPANDED_STYLE,
)


def tab_yield_content():
    # Return the layout that includes the map iframe
    return html.Div(
        [
            dcc.Store(id="aggregated-data-store"),
            layout_container,
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
    )


@app.callback(
    [
        Output("sidebar", "style"),
        Output("sidebar-header", "style"),
        Output(
            "toggle-sidebar-button", "children"
        ),  # Update the icon of the toggle button
        Output("controls-container", "style"),
    ],
    [Input("toggle-sidebar-button", "n_clicks")],
    [State("sidebar", "style")],
)
def toggle_sidebar(n_clicks, sidebar_style):
    if n_clicks and n_clicks % 2 == 1:
        # Sidebar is collapsed
        return (
            SIDEBAR_COLLAPSED_STYLE,
            SIDEBAR_HEADER_COLLAPSED_STYLE,
            html.I(className="fa fa-chevron-right"),  # Change icon to right arrow
            {"display": "none"},
        )
    else:
        # Sidebar is expanded
        return (
            SIDEBAR_EXPANDED_STYLE,
            SIDEBAR_HEADER_EXPANDED_STYLE,
            html.I(className="fa fa-chevron-left"),  # Change icon to left arrow
            {"display": "block"},
        )


# @app.callback(
#     Output('sidebar', 'style'),
#     Output('map-container', 'style'),
#     Input('toggle-sidebar-button', 'n_clicks'),
#     State('sidebar', 'style'),
# )
# def toggle_sidebar(n_clicks, sidebar_style):
#     if n_clicks % 2 == 1:  # If sidebar is to be collapsed
#         return SIDEBAR_COLLAPSED_STYLE, MAP_CONTAINER_STYLE
#     else
#         : # If sidebar is to be expanded
#         return SIDEBAR_EXPANDED_STYLE, MAP_CONTAINER_STYLE


# Callback to toggle the visibility of the table and link container
@app.callback(
    Output("table-and-link-container", "style"),
    [Input("toggle-table-button", "n_clicks")],
    [State("table-and-link-container", "style")],
)
def toggle_table(n, style):
    if n % 2 == 0:
        style["display"] = "block"  # Show the container
    else:
        style["display"] = "none"  # Hide the container
    return style


# Adjust the callback to modify the flexGrow property of the map container
@app.callback(
    Output("map-container", "style"),
    Input("toggle-sidebar-button", "n_clicks"),
)

# Map flex grows as the sidebar expands or not.
def adjust_map_width(n_clicks):
    if n_clicks and n_clicks % 2 == 1:  # If sidebar is collapsed
        return {"flexGrow": 1, "transition": "flex-grow 0.5s ease"}
    else:  # If sidebar is expanded
        return {"flexGrow": 1, "transition": "flex-grow 0.5s ease"}


# @app.callback(
#     [Output("table-container", "style"),
#      Output("toggle-table-button", "children")],
#     [Input("toggle-table-button", "n_clicks")],
#     [State("table-container", "style")],
# )
# def toggle_table(n_clicks, table_style):
#     if n_clicks % 2 == 0:
#         # Table is visible, shrink to a small strip
#         table_style["height"] = "5%"
#         table_style["minHeight"] = "20px"  # Ensure it does not disappear completely
#         button_children = [html.I(className="fas fa-arrow-up")]  # Change icon to arrow-up
#     else:
#         # Table is shrunk, expand it
#         table_style["height"] = "100%"
#         button_children = [html.I(className="fas fa-arrow-down")] # Change icon to arrow-down
#     return table_style, button_children

# @app.callback(
#         Output('yield-iframe', 'srcDoc'),
#         [
#             Input('crop-checklist', 'value'),
#             State('aggregated-data-table', 'data'),
#         ]
# )
# def update_base_map(crop, aggregated_data):
#     baseMap = create_rwanda_map()

#     # Save the map to a temporary HTML file
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
#         baseMap.save(tmp.name)
#         tmp.close()
#         # Read the content of this temporary file into a string
#         with open(tmp.name, 'r') as f:
#             html_string = f.read()

#     # Delete the temporary file now that we've read it into a string
#     os.unlink(tmp.name)

#     # Color the map
#     print("DF in map?")
#     print(aggregated_data)
#     color_map(aggregated_data)

#     return html_string

# @app.callback(
#     Output('aggregated-data-table', 'data'),
#     Output('aggregated-data-table', 'columns'),
#     [   Input('crop-checklist', 'value'),
#         Input('year-dropdown', 'value'),
#         Input('season-checklist', 'value'),
#         Input('variable-dropdown', 'value'),
#         Input('map-indicators-radioitems', 'value')]
# )
# def update_table(crop, years_filter, season_filter, aggregation_type, map_idx):
#     # Call your aggregate_data function
#     df_aggregated = aggregate_data(crop, years_filter, season_filter, aggregation_type, map_idx)

#     # Convert the DataFrame into a format suitable for the DataTable
#     data = df_aggregated.to_dict('records')
#     columns = [{'name': i, 'id': i} for i in df_aggregated.columns]

#     return data, columns


@app.callback(
    Output("aggregated-data-store", "data"),
    [
        Input("crop-checklist", "value"),
        Input("year-dropdown", "value"),
        Input("season-checklist", "value"),
        Input("variable-dropdown", "value"),
        Input("map-indicators-radioitems", "value"),
    ],
)
def store_aggregated_data(crop, years_filter, season_filter, aggregation_type, map_idx):
    df_aggregated = aggregate_data(
        crop, years_filter, season_filter, aggregation_type, map_idx
    )
    return df_aggregated.to_json(date_format="split", orient="split")


# @app.callback(
#     Output('yield-iframe', 'srcDoc'),
#     [Input('aggregated-data-store', 'data')]
# )
# def update_base_map(aggregated_data_json):
#     if aggregated_data_json:
#         # Deserialize the JSON string to DataFrame
#         df_aggregated = pd.read_json(aggregated_data_json, orient='split')

#         # Create the base map with coloring based on the aggregated data
#         baseMap = create_rwanda_map()
#         map_ = color_map(df_aggregated) # coloring logic using df_aggregated

#         # Save the map to a temporary HTML file and read the content
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
#             baseMap.save(tmp.name)
#             tmp.close()
#             with open(tmp.name, 'r') as f:
#                 html_string = f.read()
#         os.unlink(tmp.name)
#         return html_string
#     return "Please select filters to display the map."


@app.callback(
    Output("yield-iframe", "srcDoc"), [Input("aggregated-data-store", "data")]
)
def update_base_map(aggregated_data_json):
    if aggregated_data_json:
        # Deserialize the JSON string to DataFrame
        df_aggregated = pd.read_json(aggregated_data_json, orient="split")

        # Create the colored map
        rwanda_map = color_map(df_aggregated)

        # Save the map to a temporary HTML file and read the content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            rwanda_map.save(tmp.name)
            tmp.close()
            with open(tmp.name, "r") as f:
                html_string = f.read()
        os.unlink(tmp.name)
        return html_string
    return "Please select filters to display the map."


@app.callback(
    [
        Output("aggregated-data-table", "data"),
        Output("aggregated-data-table", "columns"),
    ],
    [Input("aggregated-data-store", "data")],
)
def update_table(aggregated_data_json):
    if aggregated_data_json:
        df_aggregated = pd.read_json(aggregated_data_json, orient="split")
        data = df_aggregated.to_dict("records")
        columns = [{"name": i, "id": i} for i in df_aggregated.columns]
        return data, columns
    return [], []


# ----------------------------------------------------- #
#                   TABS JOINER                        #
# ----------------------------------------------------- #
@callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "tab-1":
        return tab_1_content()
    elif tab == "tab-2":
        return tab_2_content()
    elif tab == "tab-3":
        return tab_3_content()
    elif tab == "tab-yield-gap":
        return tab_yield_content()


if __name__ == "__main__":
    app.run_server(debug=True)
