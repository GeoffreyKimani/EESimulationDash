import dash
import folium
from datetime import date
from dash import dcc, html, Input, Output, callback
import tempfile
from src.utils.ee_imageCol import *
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from components.yield_tab_components import (
    SIDEBAR_EXPANDED_STYLE,
    SIDEBAR_HEADER_EXPANDED_STYLE,
)
from components.load_extract_components import (
    SIDEBAR_EXPANDED_STYLE,
    SIDEBAR_HEADER_EXPANDED_STYLE,
)

# Functions and variables from each tab file
from src.utils.step_two import (
    satellite_dict,
)
from src.utils.step_three import load_features_for_crop, preprocess_features, scale_y
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ------------------------------------------------------------------#
#                   SATELLITE DATA EXTRACTION                       #
# ------------------------------------------------------------------#
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
def satellite_data_extraction_tab():
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


@callback(
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
@callback(
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
