import os
import tempfile
import pandas as pd
from dash import dcc, html, Input, Output, callback, State
from src.components.yield_gap_layout import (
    layout_container,
    SIDEBAR_EXPANDED_STYLE,
    SIDEBAR_COLLAPSED_STYLE,
    SIDEBAR_HEADER_COLLAPSED_STYLE,
    SIDEBAR_HEADER_EXPANDED_STYLE,
)
from src.components.load_extract_components import (
    SIDEBAR_EXPANDED_STYLE,
    SIDEBAR_COLLAPSED_STYLE,
    SIDEBAR_HEADER_COLLAPSED_STYLE,
    SIDEBAR_HEADER_EXPANDED_STYLE,
)

# from dash.dependencies import Input, Output
import tempfile
from src.utils.ee_imageCol import *

# Functions for Yield Gap Tab
from src.utils.yield_tab import aggregate_data, color_map


# ----------------------------------------------------- #
#                   YIELD GAP TAB                       #
# ----------------------------------------------------- #
def tab_yield_content():
    # Return the layout that includes the map iframe and table
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


@callback(
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


@callback(
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
@callback(
    Output("map-container", "style"),
    Input("toggle-sidebar-button", "n_clicks"),
)

# Map flex grows as the sidebar expands or not.
def adjust_map_width(n_clicks):
    if n_clicks and n_clicks % 2 == 1:  # If sidebar is collapsed
        return {"flexGrow": 1, "transition": "flex-grow 0.5s ease"}
    else:  # If sidebar is expanded
        return {"flexGrow": 1, "transition": "flex-grow 0.5s ease"}


@callback(
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


@callback(Output("yield-iframe", "srcDoc"), [Input("aggregated-data-store", "data")])
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


@callback(
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
