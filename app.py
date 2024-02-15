import os
import pandas as pd
from constants import land_cover_dir, district_shape_file
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
from step_one import load_data_for_crop, plot_plots_in_data, plot_districts_with_plotly, hectares_to_square_edges
import geopandas as gpd
from shapely.geometry import Polygon

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Tabs(
        id="tabs", 
        value='tab-1', 
        children=[
            dcc.Tab(label='Step 1: Load and Extract Fields Data', value='tab-1', className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Step 2: Preprocessing', value='tab-2', className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Step 3: Prediction', value='tab-3', className='custom-tab', selected_className='custom-tab--selected'),
        ],
        className='custom-tabs'
    ),
    html.Div(id='tabs-content'),

    # ----------------------------------------------------- #
    #                   TAB 1 CONTENT                       #
    # ----------------------------------------------------- #
    html.Div([
        html.Label('Select Crop:', style={'fontSize': 20, 'marginBottom': '10px'}),
        dcc.Dropdown(
            id='crop-selection-dropdown',
            options=[
                {'label': 'Maize', 'value': 'maize'},
                {'label': 'Potato', 'value': 'potato'}
            ],
            value='maize',  # Default value
            clearable=False,
            style={"width": "100%", "marginBottom": '20px'}
        ),
    html.Div([
        dcc.Dropdown(id='district-selection-dropdown', style={'display': 'none'}),
        dcc.Dropdown(id='year-selection-dropdown', style={'display': 'none'})
    ], id='dynamic-input-container'), # Container for dynamic objects
    html.Div(id='csv-data-table'),  # Container for displaying CSV data
    
    dcc.Store(id='stored-data'),  # To store the filtered DataFrame
    html.Button('Show Districts', id='btn-show-districts', n_clicks=0),
    html.Button('Show Plots', id='btn-show-plots', n_clicks=0),
    html.Button('Create Plots Box', id='btn-plots-box', n_clicks=0),
    html.Div(id='districts-map-container'),  # To display the district map
    html.Div(id='plots-map-container'),  # To display the plots map
    html.Div(id='plots-map-box'),  # To display the plots map
    ], style={
        'width': '50%', 
        'margin': '0 auto', 
        'border': '2px solid #ddd', 
        'borderRadius': '15px', 
        'padding': '20px',
        'boxShadow': '2px 2px 10px #aaa'
    })
], style={'textAlign': 'center'})

# ----------------------------------------------------- #
#                   TAB 1 CONTENT                       #
# ----------------------------------------------------- #
# Separate functions for each tab's content
def tab_1_content():
    return html.Div([
        html.H3('Step 1 Content Here'),
        # Add more content here
    ])

@callback(
    [Output('dynamic-input-container', 'children'),
     Output('csv-data-table', 'children')],
    [Input('crop-selection-dropdown', 'value'),
     Input('district-selection-dropdown', 'value'),
     Input('year-selection-dropdown', 'value')],
    [State('crop-selection-dropdown', 'value')]  # State allows us to pass in additional values without triggering the callback
)
def update_inputs_and_display_csv(crop, selected_districts, selected_years, crop_state):
    df_filtered = load_data_for_crop(crop_state)  # Using crop_state here to avoid confusion with the callback trigger
    districts = df_filtered['district'].unique()
    years = df_filtered['year'].unique()

    # Filtering logic
    if selected_districts and 'all' not in selected_districts:
        df_filtered = df_filtered[df_filtered['district'].isin(selected_districts)]
    if selected_years:
        df_filtered = df_filtered[df_filtered['year'].isin(selected_years)]

    # Generate district and year dropdowns
    district_dropdown = html.Div([
        html.Label('Select District:', style={'fontSize': 20, 'marginBottom': '10px'}),
        dcc.Dropdown(
            id='district-selection-dropdown',
            options=[{'label': 'All', 'value': 'all'}] + [{'label': district, 'value': district} for district in districts],
            value=selected_districts or ['all'],
            multi=True,
            style={"width": "100%", "marginBottom": '20px'}
        )
    ])

    year_dropdown = html.Div([
        html.Label('Select Year:', style={'fontSize': 20, 'marginBottom': '10px'}),
        dcc.Dropdown(
            id='year-selection-dropdown',
            options=[{'label': year, 'value': year} for year in years],
            value=selected_years if selected_years else [],
            multi=True,
            style={"width": "100%", "marginBottom": '20px'}
        )
    ])

    # Data table for displaying filtered data
    data_table = dash_table.DataTable(
        data=df_filtered.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df_filtered.columns],
        style_table={'overflowX': 'auto'},
        page_size=10  # Adjust as per your requirement
    )

    return [district_dropdown, year_dropdown], data_table

@app.callback(
    Output('stored-data', 'data'),  # Store data in dcc.Store
    [Input('crop-selection-dropdown', 'value'),  # Plus other inputs that affect the DataFrame
     Input('district-selection-dropdown', 'value'),
     Input('year-selection-dropdown', 'value')]
)
def filter_data_and_store(crop, selected_districts, selected_years):
    # Load the full dataset for the selected crop
    full_data = load_data_for_crop(crop)

    # If selected_districts or selected_years is None (e.g., nothing is selected), set them to empty lists
    selected_districts = selected_districts or []
    selected_years = selected_years or []

    # Apply district filtering if 'all' is not selected
    if selected_districts and 'all' not in selected_districts:
        df_filtered = full_data[full_data['district'].isin(selected_districts)]
    else:
        df_filtered = full_data

    # Apply year filtering
    if selected_years:
        df_filtered = df_filtered[df_filtered['year'].isin(selected_years)]

    # Convert the filtered DataFrame to JSON and return
    return df_filtered.to_json(date_format='iso', orient='split')

@app.callback(
    Output('districts-map-container', 'children'),
    Input('btn-show-districts', 'n_clicks'),
    State('stored-data', 'data')
)
def show_districts(n_clicks, stored_data):
    if n_clicks > 0:
        df = pd.read_json(stored_data, orient='split')
        selected_districts = df['district'].dropna().unique().tolist()
        print(selected_districts)
        # Load your district geometries GeoDataFrame
        gdf = gpd.read_file(district_shape_file)
        fig = plot_districts_with_plotly(gdf, selected_districts)
        
        return dcc.Graph(figure=fig)
    return html.Div()

@app.callback(
    Output('plots-map-container', 'children'),
    [Input('btn-show-plots', 'n_clicks')],
    [State('stored-data', 'data')]
)
def show_plots(n_clicks, stored_data):
    if n_clicks > 0:
        df = pd.read_json(stored_data, orient='split')
        
        # Generate the base64-encoded image
        encoded_image = plot_plots_in_data(df)
        
        # Use the `html.Img` component to display the image directly from the base64 string
        return html.Img(src=f"data:image/png;base64,{encoded_image}")

    # If no image is to be displayed, return an empty `div`
    return html.Div()

@app.callback(
    Output('plots-map-box', 'children'),
    [Input('btn-plots-box', 'n_clicks')],
    [State('stored-data', 'data')]
)
def show_plots_box(n_clicks, stored_data):
    if n_clicks > 0:
        df = pd.read_json(stored_data, orient='split')
        
        # Apply the function to each row in the dataframe to create a polygon for each plot
        df['geometry'] = df.apply(
            lambda row: Polygon(hectares_to_square_edges(row['field_longitude'], row['field_latitude'], row['plot_hectares'])),
            axis=1
        )

        # Convert the dataframe to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

         # Generate the base64-encoded image
        encoded_image = plot_plots_in_data(df)
        
        # Use the `html.Img` component to display the image directly from the base64 string
        return html.Img(src=f"data:image/png;base64,{encoded_image}")

    # If no image is to be displayed, return an empty `div`
    return html.Div()


# ----------------------------------------------------- #
#                   TAB 2 CONTENT                       #
# ----------------------------------------------------- #

def tab_2_content():
    return html.Div([
        html.H3('Step 2 Content Here'),
        # Add more content here
    ])

def tab_3_content():
    return html.Div([
        html.H3('Step 3 Content Here'),
        # Add more content here
    ])

@callback(Output('tabs-content', 'children'),
          Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return tab_1_content()
    elif tab == 'tab-2':
        return tab_2_content()
    elif tab == 'tab-3':
        return tab_3_content()

if __name__ == '__main__':
    app.run_server(debug=True)
