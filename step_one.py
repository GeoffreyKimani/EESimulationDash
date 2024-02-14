import base64
from io import BytesIO

import os, rasterio, matplotlib
import rasterio.plot
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape
from constants import DATA_DIR, district_shape_file, ASSETS_DIR
from contextlib import contextmanager

matplotlib.use('Agg')


def load_data_for_crop(crop):
    df = None
    if crop == 'maize':
        maize_file = os.path.join(DATA_DIR, 'Rwanda_locations_maize.csv')
        df = pd.read_csv(maize_file)
    else:
        potato_file = os.path.join(DATA_DIR, 'Rwanda_locations_potato.csv')
        df = pd.read_csv(potato_file)
    return df

@contextmanager
def plot_context():
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        yield fig, ax
    finally:
        plt.close(fig)

def plot_plots_in_data(locs_df, output_image_path=f"{ASSETS_DIR}/plots_image.png"):
    # Step 1: Load the base map
    with rasterio.open(DATA_DIR + f"/GlobalLandcover/Clipped_Rwanda_L1_LCC_19.tif") as src:
        raster_crs = src.crs
        print(f"Raster CRS: {raster_crs}")

        # Read the raster data
        base_map_array = src.read(1)

        # Get the transform (assuming it's correct in the metadata)
        transform = src.transform

    # Step 2: Create a GeoDataFrame from DataFrame with plots of interest
    gdf = gpd.GeoDataFrame(
        locs_df,
        geometry=gpd.points_from_xy(
            locs_df.field_longitude, locs_df.field_latitude
        )
    )

    # Set the CRS for the GeoDataFrame to match the raster CRS
    gdf.crs = raster_crs

    # If necessary, convert the GeoDataFrame to the same CRS as the raster
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    with plot_context() as (fig, ax):
        # Your existing plotting code...
        rasterio.plot.show(base_map_array, ax=ax, transform=transform, cmap='terrain', extent=src.bounds)
        gdf.plot(ax=ax, color='red', markersize=5)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()  # Make sure to close the plot
        # Encode the image in base64 and return
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # # Return the path to the saved image
    # return output_image_path

def defineROI(district_name, gdf):
    area = None
    for _, row in gdf.iterrows():
        if row['NAME_2'] == district_name:
            area = shape(row['geometry'])
            return area  # Return the ROI immediately after it's found
    return area

import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import shape
import json

def plot_districts_with_plotly(gdf, selected_districts):
    """
    Plots the boundaries of selected districts using Plotly on a Mapbox map.

    Args:
    gdf (GeoDataFrame): GeoDataFrame containing all district geometries.
    selected_districts (list): List of district names to be visualized.

    Returns:
    fig (plotly.graph_objects.Figure): Plotly figure object for displaying in Dash.
    """
    # Ensure the geometry is in latitude and longitude
    gdf = gdf.to_crs(epsg=4326)
    
    # Create a map using Plotly
    fig = px.choropleth_mapbox(gdf, geojson=gdf.geometry.__geo_interface__, 
                               locations=gdf.index, 
                               center={"lat": gdf.geometry.centroid.y.mean(), 
                                       "lon": gdf.geometry.centroid.x.mean()},
                               mapbox_style="carto-positron", zoom=7)
    
    # Loop over selected districts and draw their boundaries
    for district in selected_districts:
        roi = defineROI(district, gdf)
        if roi is not None:
            # Convert the ROI to GeoJSON format
            geojson = json.loads(json.dumps(shape(roi).__geo_interface__))
            
            # Draw the boundary based on the GeoJSON 'type'
            if geojson['type'] == 'Polygon':
                x, y = zip(*geojson['coordinates'][0])  # Polygon
                fig.add_trace(go.Scattermapbox(
                    mode='lines+text',
                    lon=x,
                    lat=y,
                    marker={'size': 1},
                    line={'width': 2, 'color': 'red'},
                    name=district,  # Set the trace name to the district's name
                    text=[district],  # Set the label text to the district's name
                    textposition="top center"  # Position the text above the centroid of the district
                ))
            elif geojson['type'] == 'MultiPolygon':
                # Handle MultiPolygon geometries
                for poly in geojson['coordinates']:
                    x, y = zip(*poly[0])  # Assumes outer boundary is first
                    fig.add_trace(go.Scattermapbox(
                        mode='lines+text',
                        lon=x,
                        lat=y,
                        marker={'size': 1},
                        line={'width': 2, 'color': 'red'},
                        name=district,  # Set the trace name to the district's name
                        text=[district],  # Set the label text to the district's name
                        textposition="top center"  # Position the text above the centroid of the district
                    ))

    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
    return fig
