from ee_init import initialize_ee
initialize_ee()

import json, dash
import folium
import tempfile
import pandas as pd
from datetime import date
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
import geopandas as gpd
from shapely.geometry import Polygon

# Functions and variables from each tab file
from utils import defineROI
from constants import land_cover_dir, district_shape_file
from step_one import load_data_for_crop, plot_plots_in_data, plot_districts_with_plotly, hectares_to_square_edges
from step_two import satellite_dict, add_ee_layer, integrate_indices_to_dataframe, calculate_satellite_dates

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Tabs(
        id="tabs", 
        value='tab-1', 
        children=[
            dcc.Tab(label='Step 1: Load and Extract Fields Data', value='tab-1', id='tab-1', className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Step 2: Satellite Data Extraction', value='tab-2', id='tab-2', className='custom-tab', selected_className='custom-tab--selected'),
            dcc.Tab(label='Step 3: Prediction', value='tab-3', id='tab-3', className='custom-tab', selected_className='custom-tab--selected'),
        ],
        className='custom-tabs'
    ),
    html.Div(id='tabs-content'),
    # define the data storage globally
    dcc.Store(id='stored-data'),  # To store the filtered DataFrame
    dcc.Store(id='gdf-data'), # To store the GDF DataFrame
], style={'textAlign': 'center'})

# ----------------------------------------------------- #
#                   TAB 1 CONTENT                       #
# ----------------------------------------------------- #
# Separate functions for each tab's content
def tab_1_content():
    return     html.Div([
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
    Output('gdf-data', 'data'),
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

        # Instead of directly returning an image, save the modified DataFrame to 'modified-data'
        return gdf.to_json()  # Convert GeoDataFrame to JSON for storage
    
    return dash.no_update  # Use dash.no_update when there's no update to the store

@app.callback(
    Output('plots-map-box', 'children'),
    [Input('gdf-data', 'data'), Input('btn-plots-box', 'n_clicks')]
)
def update_image(gdf_data, n_clicks):
    if gdf_data and n_clicks > 0:
        # Deserialize the JSON string into a dictionary
        gdf_dict = json.loads(gdf_data)
        
        # Create a GeoDataFrame from the 'features' key of the GeoJSON dictionary
        gdf = gpd.GeoDataFrame.from_features(gdf_dict['features'], crs="EPSG:4326")

        encoded_image = plot_plots_in_data(gdf)  # Assuming this function now accepts a GeoDataFrame
        return html.Img(src=f"data:image/png;base64,{encoded_image}")
    return html.Div()

# ----------------------------------------------------- #
#                   TAB 2 CONTENT                       #
# ----------------------------------------------------- #

def tab_2_content():
    return html.Div([
    dcc.Store(id='stored-data'),  # To store the filtered DataFrame

    html.Div([
        html.Label("Satellite name:"),
        dcc.Dropdown(
            id='dropdown-Satellite',
            options=[{'label': sat_name, 'value': sat} for (sat_name, sat) in satellite_dict.items()],
            value='',
            searchable=True,
            placeholder="Select Satellite...",
            style={'min-width': '200px'})
    ], style={'display': 'inline-block', 'margin-left': '60px', 'margin-right': '20px'}),

    html.Div([
        html.Label("Time of Interest: "),
        dcc.DatePickerRange(
            id='time-of-interest',
            min_date_allowed=date(2016, 1, 1),
            max_date_allowed=date.today(),
            initial_visible_month=date(2020, 1, 1),
            end_date=date.today()
        )
    ], style={'display': 'inline-block', 'margin-right': '20px'}),

    html.Div([
        dcc.Checklist(id='cloud-mask-checkbox',
                    options=[{'label': 'Mask Clouds', 'value': 'cloud_mask'}],
                    value=['cloud_mask']
                    ),

        dcc.Checklist(
                    id='filter-checkbox',
                    options=[{'label': 'Use MA Filter', 'value': 'filters'}],
                    value=[]
    )
    ], style={'display': 'inline-block', 'margin-right': '20px'}),

    html.Div([
        html.Button('Submit', id='submit-val', n_clicks=0),
    ], style={'display': 'inline-block', 'margin-right': '20px'}),

    html.Div([
        html.Label("Spectral indices"),
        dcc.Checklist(
            id='index-checkboxes',
            options=[
                {'label': 'NDVI', 'value': 'NDVI'},
                {'label': 'EVI', 'value': 'EVI'},
                {'label': 'SAVI', 'value': 'SAVI'}
                # Add more indices as needed
            ],
            value=[]
        )
    ]),
    
    html.Div([
        html.Iframe(
            id='map-iframe',
            srcDoc='',
            style={'width': '100%', 'height': '600px'}
        )
    ]),

    html.Div(id='output-info')
])

@app.callback(
    Output('map-iframe', 'srcDoc'),
    [Input('submit-val', 'n_clicks'),
     Input('gdf-data', 'data')
     ],
    [State('dropdown-Satellite', 'value'),
     State('time-of-interest', 'start_date'),
     State('time-of-interest', 'end_date'),
     State('cloud-mask-checkbox', 'value'),
     State('filter-checkbox', 'value'),
     State('index-checkboxes', 'value')
    ]
)
def update_map(n_clicks, gdf_data, satellite, start_date, end_date, mask_clouds, use_filters,indices):
    
    # Generate and save the map
    country_lon = 29.8739
    country_lat = -1.9403
    
    # Add the Earth Engine layer method to folium.
    folium.Map.add_ee_layer = add_ee_layer
    
    base_map = folium.Map(location=[country_lat, country_lon], zoom_start=9)
    
    if base_map:
        # Add a layer control panel to the map.
        baseMap = base_map._repr_html_()
        
        if n_clicks > 0:
            # Deserialize the JSON string into a dictionary
            gdf_dict = json.loads(gdf_data)
            
            # Create a GeoDataFrame from the 'features' key of the GeoJSON dictionary
            df = gpd.GeoDataFrame.from_features(gdf_dict['features'], crs="EPSG:4326")
            print(df.columns)

            # Before Getting Satellite Images
            df['start_date'], df['end_date'] = zip(*df.apply(lambda row: calculate_satellite_dates(row['year'], row['season'], 90, 120), axis=1))
            band_image = None

            if satellite.startswith('LANDSAT'):
                print("Landsat selected")
                df['start_date_cswi'], df['end_date_cswi'] = zip(*df.apply(lambda row: calculate_satellite_dates(row['year'], row['season'], 45, 85), axis=1))
                band_image, df = integrate_indices_to_dataframe(df, 'LANDSAT')
            else:
                print("Sentinel selected")
                band_image, df = integrate_indices_to_dataframe(df, 'Sentinel')

            print('*'*20,df.columns)
            # img_collection = importLandsat(start_date, end_date, roi, data=satellite)
            base_map = folium.Map(location=[country_lat, country_lon], zoom_start=9)
           
        #     if mask_clouds:
        #         # Mask clouds
        #         img_collection = img_collection.map(maskCloudLandsat) if satellite.startswith('LANDSAT') else img_collection.map(maskCloudSentinel)

        #     img_collection = img_collection.map(apply_scale_factors) if satellite.startswith('LANDSAT') else img_collection.map(SentinelOptical)

        #     if use_filters:
        #         # Apply filters
        #         img_collection = img_collection.map(applyMovingAvg) 

        #     cloud_free_composite = img_collection.map(lambda img: img.clip(roi)).median()

        #     min_, max_, _ = imageStats(cloud_free_composite,roi, satellite) 
        #     bands = ['SR_B4', 'SR_B3', 'SR_B2'] if satellite.startswith('LANDSAT') else ['B4', 'B3', 'B2']

        #     vis_params = {
        #         'bands': bands,
        #         'min': min_,
        #         'max': max_,
        #         'gamma': 1
        #     }  

            
        #     base_map.add_ee_layer(cloud_free_composite,
        #         vis_params,
        #         'EE-Image', True)

        #     if 'NDVI' in indices:
        #         cloud_free_composite = add_NDVI(cloud_free_composite)#.map(add_NDVI)
        #         base_map.add_ee_layer(cloud_free_composite,
        #         ndvi_params,
        #         'NDVI', True,0.9)
                
        #     if 'EVI' in indices:
        #         cloud_free_composite = add_EVI(cloud_free_composite)#.map(add_EVI)
        #         base_map.add_ee_layer(cloud_free_composite,
        #         evi_params,
        #         'EVI', True,0.9)

        #     if 'SAVI' in indices:
        #         cloud_free_composite = add_SAVI(cloud_free_composite)#.map(add_SAVI)
        #         base_map.add_ee_layer(cloud_free_composite,
        #         savi_params,
        #         'SAVI', True,0.9)

            
            base_map.add_ee_layer(band_image)
            baseMap = base_map.add_child(folium.LayerControl())._repr_html_()          
          
        # Save the map to a temporary HTML file
        tmp_html = tempfile.NamedTemporaryFile(suffix='.html').name
        with open(tmp_html, 'w') as f:
            f.write(baseMap)

        return open(tmp_html).read()
    
    else:
        return None

# ----------------------------------------------------- #
#                   TAB 2 CONTENT                       #
# ----------------------------------------------------- #
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
