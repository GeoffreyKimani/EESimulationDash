# Layout imports
from dash import dcc, html, dash_table
from yield_tab import create_legend
import dash_bootstrap_components as dbc

# ----------------------------------------------------- #
#                   LAYOUT CONTENT                      #
# ----------------------------------------------------- #

map_iframe = html.Iframe(
    id="yield-iframe",
    width="100%",
    height="480px",  # Adjust height as needed
    style={"border": "2px solid lightgrey", "border-radius": "8px", "zIndex": 0},
)

# # Step: Create accordion container
# map_container = html.Div(
#     dbc.AccordionItem([])
#     # id="accordion-container",
#     # children=[map_iframe],
#     # style={"width": "100%", "height": "80%", "position": "relative", "flex": "1"},
# )


# Step 1: Create the map container
map_container = html.Div(
    id="map-container",
    children=[map_iframe],
    style={"width": "100%", "height": "480px", "position": "relative", "flex": "1"},
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
                            "Crops:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        dcc.RadioItems(
                            id="crop-checklist",
                            options=[
                                {"label": "Maize", "value": "maize"},
                                {"label": "Potatoes", "value": "potato"},
                                {"label": "Other", "value": "other"},
                            ],
                            value="maize",
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
                ),  # Adjust the padding-left as needed
                # Container for year selection
                html.Div(
                    [
                        html.Label(
                            "Year Selection:",
                            style={"display": "inline-block", "margin-right": "10px"},
                        ),
                        # Map indicators
                        dcc.Dropdown(
                            id="year-dropdown",
                            options=[
                                # {'label': 'All', 'value': 'all'},
                                {"label": "2016", "value": 2016},
                                {"label": "2019", "value": 2019},
                                {"label": "2020", "value": 2020},
                            ],
                            value=2016,
                            multi=False,
                            style={"width": "100%", "marginBottom": "20px"},
                        ),
                    ],
                    style={"margin-bottom": "10px", "padding-left": "20px"},
                ),
                # Container for season selection
                html.Div(
                    [
                        html.Label(
                            "Season Selection:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        # Map indicators
                        dcc.RadioItems(
                            id="season-checklist",
                            options=[
                                {"label": "Season A", "value": "a season"},
                                {"label": "Season B", "value": "b season"},
                                # {'label': 'Aggregate Seasons', 'value': 'all'}
                            ],
                            value="a season",
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
                # Container for the map indicators with label
                html.Div(
                    [
                        html.Label(
                            "Map Indicators:",
                            style={
                                "display": "inline-block",
                            },
                        ),
                        # Map indicators
                        dcc.RadioItems(
                            id="map-indicators-radioitems",
                            options=[
                                {"label": "Actual Yield", "value": "actual_yield"},
                                {
                                    "label": "Potential Yield",
                                    "value": "potential_yield",
                                },
                                {
                                    "label": "Predicted Yield",
                                    "value": "predicted_yield",
                                },
                                {"label": "Yield Gap", "value": "yield_gap"},
                            ],
                            value="actual_yield",
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
                # Container for the new Variable selection with label
                html.Div(
                    [
                        html.Label(
                            "Aggregation:",
                            style={"display": "inline-block", "margin-right": "10px"},
                        ),
                        dcc.Dropdown(
                            id="variable-dropdown",
                            options=[
                                {"label": "Mean Value", "value": "mean_value"},
                                {"label": "Sum Total", "value": "sum_total"},
                            ],
                            placeholder="Select Variable",
                            value="mean_value",
                            style={"width": "100%", "marginBottom": "20px"},
                        ),
                    ],
                    style={"margin-bottom": "10px", "padding-left": "20px"},
                ),
                # Crop mask
                html.Div(
                    [
                        html.Label(
                            "Apply crop mask?",
                            style={"display": "inline-block", "margin-right": "10px"},
                        ),
                        dcc.RadioItems(
                            id="crop-mask-radioitems",
                            options=[
                                {"label": "No", "value": "no"},
                                {"label": "Yes", "value": "yes"},
                            ],
                            value="no",  # Default selected value
                            labelStyle={
                                "display": "inline-block",
                                "margin-right": "15px",
                            },
                            style={"display": "inline-block"},
                        ),
                    ],
                    style={"margin-bottom": "10px"},
                ),
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
            children="Map Settings",
            style=SIDEBAR_HEADER_EXPANDED_STYLE,
        ),
        html.Div(
            id="controls-container",
            children=[controls_container, create_legend()],
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
table_container = html.Div(
    [
        dash_table.DataTable(
            id="aggregated-data-table",
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
            page_size=10,  # Adjust as per your requirement
        )
    ],
    style={
        "width": "85%",
        "display": "inline-block",
        "verticalAlign": "top",
        "height": "60%",
    },
)

# Step 3: Create the bottom table and dataset link
table_and_link_container = html.Div(
    id="table-and-link-container",
    children=[
        table_container,
    ],
    style={
        "display": "block",
        "position": "relative",
        "width": "100%",
        "height": "60%",
    },
)


# Container that includes both the toggle button and the table and link container
# table_and_button_container = html.Div(
#     [toggle_table_button, table_and_link_container],
#     style={
#         "position": "relative",
#         "width": "80%",
#         "margin": "auto",
#         "height":"100px",
#     },  # Center the container
# )

yield_tab_accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                title="Map",
                children=[map_container],
                style={"height": "100%", "width": "100%"},
                item_id="item-1",
            ),
            dbc.AccordionItem(
                title="Filtered Data",
                children=[table_and_link_container],
                style={"height": "60%", "width": "100%"},
                item_id="item-2",
            ),
        ],
        always_open=True,
        id="accordion-always-open",
        active_item=[
            "item-1",
            "item-2",
        ],  # Set the IDs of the item you want to be open by default
        # Adjust the style here to ensure the accordion fills its container
        style={"height": "100%", "width": "100%"},
    ),
    # Adjust the style here to make sure the div containing the accordion also fills its container
    style={"height": "100%", "width": "100%"},
)


layout_container = html.Div(
    style={
        "display": "flex",
        "flexDirection": "row",
        "height": "100%",  # Ensure it fills the vertical space
    },
    children=[
        sidebar,  # assuming sidebar has a fixed width or a flex-basis set
        html.Div(
            yield_tab_accordion,
            style={
                "flexGrow": 1,
                "flexShrink": 1,
                "flexBasis": "auto",
                "height": "100%",
            },  # Allow accordion to grow and fill the space
        ),
    ],
)
