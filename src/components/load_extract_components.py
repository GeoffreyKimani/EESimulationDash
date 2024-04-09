# Layout imports
from dash import dcc, html, dash_table
from src.utils.yield_tab import create_legend
import dash_bootstrap_components as dbc


# ----------------------------------------------------- #
#                   LAYOUT CONTENT                      #
# ----------------------------------------------------- #


# Step 1: Create the map container

district_container = html.Div(
    id="districts-map-container",
    children=[],
    style={"width": "100%", "height": "100%", "position": "relative", "flex": "1"},
)

plots_container = html.Div(
    id="plots-map-container",
    children=[],
    style={"width": "100%", "height": "100%", "position": "relative", "flex": "1"},
)
# ----------------------------------------------------- #
#                    SIDE BAR CONTROLS                  #
# ----------------------------------------------------- #

# CSS for the sidebar header when the sidebar is expanded
SIDEBAR_HEADER_EXPANDED_STYLE = {
    "textAlign": "center",
    "color": "#052C65",
    "padding": "1rem 1rem",
    "backgroundColor": "#CFE2FF",
    "height": "3%",
}

# CSS for the sidebar header when the sidebar is collapsed
SIDEBAR_HEADER_COLLAPSED_STYLE = {
    "textAlign": "center",
    "color": "white",
    "backgroundColor": "#6DCFF2",
    "width": "40px",
    "height": "100vh",  # Full height of the viewport
    "position": "relative",
    "top": 0,
    "left": 0,
    "writingMode": "vertical-lr",
    "textOrientation": " sideways-right",
    "zIndex": "1",
    "padding": "15px 11px 15px 6px",
    "font-size": "x-large",
}

# CSS for the entire sidebar when collapsed
SIDEBAR_COLLAPSED_STYLE = {
    "width": "40px",
    "maxWidth": "40px",
    "minWidth": "40px",
    # "height": "100%",  # Adjust this value to account for your navbar height
    "position": "relative",  # Use 'fixed' to position the sidebar relative to the viewport
    "top": "0",  # Set this value to the height of your navbar
    "left": 0,
    "backgroundColor": "#6DCFF2",
    "zIndex": "1",
    "overflowX": "hidden",
    "overflowY": "hidden",
    "transition": "all 0.5s",
}

# CSS for the expanded sidebar
SIDEBAR_EXPANDED_STYLE = {
    "flex": "0 0 20%",  # 20% width, do not grow or shrink
    "maxWidth": "20%",
    "minWidth": "250px",
    "height": "100%",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
    "overflowY": "auto",
    # 'position': 'fixed',
    "position": "relative",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20%",
    "zIndex": "1",
}

# Container for the map and the input controls side by side
controls_container = html.Div(
    [
        # Container for the input controls
        html.Div(
            [
                # Container for the crop selection with label
                html.Div(
                    [
                        html.Label(
                            "Select Crop:",
                            style={"fontSize": 20, "marginBottom": "10px"},
                        ),
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
                                    id="district-selection-dropdown",
                                    style={"display": "none"},
                                ),
                                dcc.Dropdown(
                                    id="year-selection-dropdown",
                                    style={"display": "none"},
                                ),
                            ],
                            id="dynamic-input-container",
                        ),  # Container for dynamic objects
                    ],
                    style={"margin-bottom": "10px", "padding-left": "20px"},
                ),  # Adjust the padding-left as needed
                # Container for year selection
            ],
            style={
                "width": "100%",
                "display": "inline-block",
                "padding": "10px",
                "verticalAlign": "top",
                "boxSizing": "border-box",
            },
        ),
    ]
)  # , style={'display': 'flex', 'flexWrap': 'wrap', 'width': '100%', 'margin-top': '20px'})

# Step 2: Create the collapsible sidebar with vertical text when collapsed
sidebar = html.Div(
    id="sidebar",
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
            children="Filter Settings",
            style=SIDEBAR_HEADER_EXPANDED_STYLE,
        ),
        html.Div(
            id="controls-container",
            children=[controls_container],
            style={
                "display": "block"
            },  # Use 'none' to hide the controls when sidebar is collapsed
        ),
    ],
    style=SIDEBAR_EXPANDED_STYLE,
)

# ----------------------------------------------------- #
#                    TABLE INFORMATION                  #
# ----------------------------------------------------- #
toggle_table_button = html.Button("Toggle Table", id="toggle-table-button", n_clicks=0)

# Container for the table and the legend
table_container1 = html.Div(
    [html.Div(id="csv-data-table")],  # This will be where the DataTable is inserted
    style={
        "width": "100%",  # Adjusted to take full width
        "display": "inline-block",
        "verticalAlign": "top",
        "height": "100%",
    },
)

table_and_link_container1 = html.Div(
    id="table-and-link-container1",
    children=[
        table_container1,
    ],
    style={
        "display": "block",
        "position": "relative",
        "width": "100%",  # Adjusted to take full width of its parent
        "height": "100%",
    },
)

table_row = dbc.Row(
    [
        dbc.Col(  # This column will take up 8 out of 12 columns in the grid
            html.Div(
                id="table-and-link-container1",
                children=[
                    html.Div(id="csv-data-table")  # Placeholder for the DataTable
                ],
                style={
                    "width": "100%",  # Use 100% of the column width
                    "height": "100%",  # Adjust the height as necessary
                },
            ),
            width=8,  # Specifies that this column should take 8 out of 12 columns in the grid
        ),
        # Add more dbc.Col here if you need other columns beside the table
    ]
)

load_tab_accordion = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                title="Plots",
                                children=[
                                    plots_container
                                ],  # Ensure map_container is properly defined elsewhere
                                item_id="item-1",
                            ),
                        ],
                        start_collapsed=False,
                    ),
                    width=6,  # Use half the width for the first accordion
                    style={"padding": "0 5px", "height": "100%"},
                ),
                dbc.Col(
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                title="Districts",
                                children=[
                                    district_container
                                ],  # This will be the same map_container or another one as per your requirement
                                item_id="item-2",
                            ),
                        ],
                        start_collapsed=False,
                    ),
                    width=6,  # Use the other half for the second accordion
                    style={"padding": "0 0", "height": "100%"},
                ),
            ],
        ),
        dbc.Row(
            dbc.Col(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            title="Filtered Data",
                            children=[
                                table_and_link_container1
                            ],  # Ensure this container is properly defined elsewhere
                            item_id="item-3",
                        ),
                    ],
                    start_collapsed=False,
                ),
                width=12,  # Use full width for the third accordion
                style={"padding": "0 0", "height": "100%"},
            ),
        ),
    ],
    style={"height": "100%"},
)


layout_container2 = html.Div(
    style={
        "display": "flex",
        "flexDirection": "row",
        "height": "100%",  # Ensure it fills the vertical space
    },
    children=[
        sidebar,  # assuming sidebar1 has a fixed width or a flex-basis set
        html.Div(
            load_tab_accordion,
            style={
                "flexGrow": 1,
                "flexShrink": 1,
                "flexBasis": "auto",
                "height": "100%",
            },  # Allow accordion to grow and fill the space
        ),
    ],
)
