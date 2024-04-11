import base64, json
from io import BytesIO

import os, rasterio, matplotlib
import rasterio.plot
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape
from constants import DATA_DIR, CROP_DATA_DIR
from contextlib import contextmanager

import plotly.express as px
import plotly.graph_objects as go

from src.utils.utils import defineROI

matplotlib.use("Agg")


def load_data_for_crop(crop):
    df = None
    if crop == "maize":
        maize_file = os.path.join(CROP_DATA_DIR, "RwandaMaizeLocations.csv")
        df = pd.read_csv(maize_file)
    else:
        potato_file = os.path.join(CROP_DATA_DIR, "RwandaPotatoLocations.csv")
        df = pd.read_csv(potato_file)
    return df


# ----------------------------------------------------- #
#                   TAB 1 PLOT FUNCTIONS                #
# ----------------------------------------------------- #
@contextmanager
def plot_context():
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        yield fig, ax
    finally:
        plt.close(fig)


def plot_plots_in_data(locs_df):
    # Step 1: Load the base map
    with rasterio.open(
        DATA_DIR + f"/GlobalLandcover/Clipped_Rwanda_L1_LCC_19.tif"
    ) as src:
        raster_crs = src.crs
        print(f"Raster CRS: {raster_crs}")

        # Read the raster data
        base_map_array = src.read(1)

        # Get the transform (assuming it's correct in the metadata)
        transform = src.transform

    # Step 2: Create a GeoDataFrame from DataFrame with plots of interest
    gdf = gpd.GeoDataFrame(
        locs_df,
        geometry=gpd.points_from_xy(locs_df.field_longitude, locs_df.field_latitude),
    )

    # Set the CRS for the GeoDataFrame to match the raster CRS
    gdf.crs = raster_crs

    # If necessary, convert the GeoDataFrame to the same CRS as the raster
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    # Image will not be saved to assets to avoid reloading the application
    with plot_context() as (fig, ax):
        rasterio.plot.show(
            base_map_array,
            ax=ax,
            transform=transform,
            cmap="terrain",
            extent=src.bounds,
        )
        gdf.plot(ax=ax, color="red", markersize=5)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()  # Make sure to close the plot
        # Encode the image in base64 and return
        return base64.b64encode(buf.getvalue()).decode("utf-8")


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
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry.__geo_interface__,
        locations=gdf.index,
        center={
            "lat": gdf.geometry.centroid.y.mean(),
            "lon": gdf.geometry.centroid.x.mean(),
        },
        mapbox_style="carto-positron",
        zoom=7,
    )

    # Loop over selected districts and draw their boundaries
    for district in selected_districts:
        roi = defineROI(district, gdf)
        if roi is not None:
            # Convert the ROI to GeoJSON format
            geojson = json.loads(json.dumps(shape(roi).__geo_interface__))

            # Draw the boundary based on the GeoJSON 'type'
            if geojson["type"] == "Polygon":
                x, y = zip(*geojson["coordinates"][0])  # Polygon
                fig.add_trace(
                    go.Scattermapbox(
                        mode="lines+text",
                        lon=x,
                        lat=y,
                        marker={"size": 1},
                        line={"width": 2, "color": "red"},
                        name=district,  # Set the trace name to the district's name
                        text=[district],  # Set the label text to the district's name
                        textposition="top center",  # Position the text above the centroid of the district
                    )
                )
            elif geojson["type"] == "MultiPolygon":
                # Handle MultiPolygon geometries
                for poly in geojson["coordinates"]:
                    x, y = zip(*poly[0])  # Assumes outer boundary is first
                    fig.add_trace(
                        go.Scattermapbox(
                            mode="lines+text",
                            lon=x,
                            lat=y,
                            marker={"size": 1},
                            line={"width": 2, "color": "red"},
                            name=district,  # Set the trace name to the district's name
                            text=[
                                district
                            ],  # Set the label text to the district's name
                            textposition="top center",  # Position the text above the centroid of the district
                        )
                    )

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


# ----------------------------------------------------- #
#                   TAB 1 PLOT EXTRACTION FUNCTIONS     #
# ----------------------------------------------------- #
def hectares_to_square_edges(lon, lat, hectares):
    """
    This function takes the center point (lon, lat) of a plot and the size in hectares,
    and returns the coordinates of the square edges.
    """
    # Convert hectares to square meters
    area_sqm = hectares * 10000
    # Assume square plot to simplify, calculate the side length in meters
    side_length_m = np.sqrt(area_sqm)
    # Convert side length from meters to degrees (approximation)
    side_length_deg = side_length_m / (
        111.32 * 1000
    )  # rough estimate: 111.32 km per degree

    # Calculate the coordinates of the four corners of the square
    top_left = (lon - side_length_deg, lat + side_length_deg)
    top_right = (lon + side_length_deg, lat + side_length_deg)
    bottom_left = (lon - side_length_deg, lat - side_length_deg)
    bottom_right = (lon + side_length_deg, lat - side_length_deg)

    return [top_left, top_right, bottom_right, bottom_left, top_left]  # Closed loop
