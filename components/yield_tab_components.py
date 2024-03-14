# Layout imports
from dash import dcc, html, dash_table
from yield_tab import create_legend

# ----------------------------------------------------- #
#                   LAYOUT CONTENT                      #
# ----------------------------------------------------- #

map_iframe = html.Iframe(
            id='yield-iframe',
            width='100%',
            height='100%',  # Adjust height as needed
            style={"border": '2px solid lightgrey', 'border-radius': '8px', "zIndex": 0}
        )

# Step 1: Create the map container
map_container = html.Div(
    id='map-container',
    children=[map_iframe],
    style={'width': '100%', 'height': '800px', 'position': 'relative', "flex": "1"}
)

# ----------------------------------------------------- #
#                    SIDE BAR CONTROLS                  #
# ----------------------------------------------------- #

# CSS for the sidebar header when the sidebar is expanded
SIDEBAR_HEADER_EXPANDED_STYLE = {
    'textAlign': 'center',
    'color': 'white',
    'padding': '2rem 1rem',
    'backgroundColor': '#6DCFF2',
    'height': '7%'
}

# CSS for the sidebar header when the sidebar is collapsed
SIDEBAR_HEADER_COLLAPSED_STYLE = {
    'textAlign': 'center',
    'color': 'white',
    'backgroundColor': '#6DCFF2',
    'width': '40px',
    'height': '100vh',  # Full height of the viewport
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'writingMode': 'vertical-rl',
    'textOrientation': 'upright',
    'zIndex': '1'
}

# CSS for the entire sidebar when collapsed
SIDEBAR_COLLAPSED_STYLE = {
    "flex": "0 0 30px",  # 30px width, do not grow or shrink
    "maxWidth": "30px",
    "minWidth": "30px",
    "height": "100%",
    
    'width': '30px',
    'backgroundColor': '#6DCFF2',
    # 'position': 'fixed',
    'position': 'relative',
    'top': 0,
    'left': 0,
    'zIndex': '1',
    'overflowX': 'visible',
    'height': '100vh'
}

# CSS for the expanded sidebar
SIDEBAR_EXPANDED_STYLE = {
    "flex": "0 0 20%",  # 20% width, do not grow or shrink
    "maxWidth": "20%",
    "minWidth": "250px", 
    "height": "100%",

    'padding': '2rem 1rem',
    'backgroundColor': '#f8f9fa',
    'overflowY': 'auto',
    # 'position': 'fixed',
    'position': 'relative',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'zIndex': '1',
}

# Container for the map and the input controls side by side
controls_container = html.Div([
    # Container for the input controls
    html.Div([
        # Container for the crop selection with label
        html.Div([
            html.Label('Crops:', style={'display': 'inline-block', 'margin-right': '10px'}),
            dcc.RadioItems(
                id='crop-checklist',
                options=[
                    {'label': 'Maize', 'value': 'maize'},
                    {'label': 'Potatoes', 'value': 'potato'},
                    {'label': 'Other', 'value': 'other'}
                ],
                value='maize',
                labelStyle={'display': 'block'},
                style={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
            ),
        ], style={'margin-bottom': '10px'}),

        # Container for year selection
        html.Div([
            html.Label('Year Selection:', style={'display': 'inline-block', 'margin-right': '10px'}),
            
            # Map indicators
            dcc.Dropdown(
                id='year-dropdown',
                options=[
                    # {'label': 'All', 'value': 'all'},
                    {'label': '2016', 'value': 2016},
                    {'label': '2019', 'value': 2019},
                    {'label': '2020', 'value': 2020},
                    
                    ],
                value=2016,
                multi=False,
                style={"width": "100%", "marginBottom": '20px'}
                    ),
        ], style={'margin-bottom': '10px'}),

        # Container for season selection
        html.Div([
            html.Label('Season Selection:', style={'display': 'inline-block', 'margin-right': '10px'}),
            
            # Map indicators
            dcc.RadioItems(
                id='season-checklist',
                options=[
                    {'label': 'Season A', 'value': 'a season'},
                    {'label': 'Season B', 'value': 'b season'},
                    # {'label': 'Aggregate Seasons', 'value': 'all'}
                ],
                value='a season',
                style={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
            ),
        ], style={'margin-bottom': '10px'}),

        # Container for the map indicators with label
        html.Div([
            html.Label('Map Indicators:', style={'display': 'inline-block', 'margin-right': '10px'}),
            
            # Map indicators
            dcc.RadioItems(
                id='map-indicators-radioitems',
                options=[
                    {'label': 'Actual Yield', 'value': 'actual_yield'},
                    {'label': 'Potential Yield', 'value': 'potential_yield'},
                    {'label': 'Predicted Yield', 'value': 'predicted_yield'},
                    {'label': 'Yield Gap', 'value': 'yield_gap'},
                ],
                value='actual_yield',
                style={'border': '1px solid #ddd', 'padding': '10px', 'border-radius': '5px'}
            ),
        ], style={'margin-bottom': '10px'}),

        # Container for the new Variable selection with label
        html.Div([
            html.Label('Aggregation:', style={'display': 'inline-block', 'margin-right': '10px'}),
            dcc.Dropdown(
                id='variable-dropdown',
                options=[
                    {'label': 'Mean Value', 'value': 'mean_value'},
                    {'label': 'Sum Total', 'value': 'sum_total'}
                ],
                placeholder="Select Variable", value='mean_value',
                style={'display': 'inline-block', 'width': 'calc(100% - 100px)'}  # Adjust width as necessary
            ),
        ], style={'margin-bottom': '10px'}),

        # Crop mask
        html.Div([
            html.Label('Apply crop mask?', style={'display': 'inline-block', 'margin-right': '10px'}),
            dcc.RadioItems(
                id='crop-mask-radioitems',
                options=[
                    {'label': 'No', 'value': 'no'},
                    {'label': 'Yes', 'value': 'yes'}
                ],
                value='no',  # Default selected value
                labelStyle={'display': 'inline-block', 'margin-right': '15px'},
                style={'display': 'inline-block'}
            ),
        ], style={'margin-bottom': '10px'})
    ], style={'width': '100%', 'display': 'inline-block', 'padding': '10px', 'verticalAlign': 'top', 'boxSizing': 'border-box'}),
]) #, style={'display': 'flex', 'flexWrap': 'wrap', 'width': '100%', 'margin-top': '20px'})

# Step 2: Create the collapsible sidebar with vertical text when collapsed
sidebar = html.Div(
    id='sidebar',
    children=[
        html.Button(
            children=[],  # No content needed, we'll use CSS for the arrow icon
            id="toggle-sidebar-button",
            className="toggle-button",
            n_clicks=0,
            style={
                'position': 'absolute', 
                'top': 0, 
                'right': '0',  # Adjusted from -30 to 0
                'width': '30px', 
                'height': '30px',  # Adjust size as needed
                'backgroundColor': 'transparent',  # Make background transparent
                'border': 'none'  # Remove border
            }
        ),

        html.Div(
        id="sidebar-header",
        children="Map Settings",
        style=SIDEBAR_HEADER_EXPANDED_STYLE),

        html.Div(
        id='controls-container',
        children=[controls_container, create_legend()],
        style={'display': 'block'} # Use 'none' to hide the controls when sidebar is collapsed
        )
],style=SIDEBAR_EXPANDED_STYLE)

# ----------------------------------------------------- #
#                    TABLE INFORMATION                  #
# ----------------------------------------------------- #
toggle_table_button = html.Button(
    "Toggle Table",
    id="toggle-table-button",
    n_clicks=0
)

# Container for the table and the legend
table_container = html.Div([
    # html.Label('Filtered Data:', style={'margin-bottom': '5px'}),
    dash_table.DataTable(
        id='aggregated-data-table',
        style_table={'overflowX': 'auto'},
        page_size=10  # Adjust as per your requirement
    )
], style={'width': 'calc(100% - 260px)', 'display': 'inline-block', 'verticalAlign': 'top'})

# Step 3: Create the bottom table and dataset link
table_and_link_container = html.Div(
    id='table-and-link-container',
    children=[
        html.Div([
            html.H2("Filtered Data", style={'textAlign': 'center', 'color': 'white'}),
        ], style={'backgroundColor': '#009DD9'}),
        table_container,
    ],
    style={'display': 'none', 'position': 'relative', 'width': '100%'}
)

# Container that includes both the toggle button and the table and link container
table_and_button_container = html.Div(
    [
        toggle_table_button,
        table_and_link_container
    ],
    style={'position': 'relative', 'width': '80%', 'margin': 'auto'}  # Center the container
)

layout_container = html.Div(
    style={
        'display': 'flex',
        'flexDirection': 'row',
        # 'height': '100%'  # Ensure it fills the vertical space
    }, children=[sidebar, map_container])
