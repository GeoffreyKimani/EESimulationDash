import re, ee
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import shape

def extract_name_from_path(img_path, _type="name"):
    """Given a filename, extract the name of the image"""

    if _type == "name":
        pattern = r'.*/(.*)\.tif'
        match = re.search(pattern, img_path)

        if match:
            return match.group(1)
    else:
        pattern = r'(\d+)\.tif'
        match = re.search(pattern, img_path)

        if match:
            return match.group(1)

def display_gee_image_on_map(image_id, vis_params, region):
    """
    Displays a Google Earth Engine image on an interactive map.

    Args:
        image_id (str): The GEE image ID to display.
        vis_params (dict): Visualization parameters (e.g., bands and min/max values).
        region (ee.Geometry): The region of interest to center the map on.

    Returns:
        geemap.Map: An interactive map displaying the specified image.
    """
    # Create a GEE Image object.
    image = ee.Image(image_id)
    
    # Create an interactive map.
    Map = geemap.Map()
    
    # Add the image to the map using the visualization parameters.
    Map.addLayer(image, vis_params, 'GEE Image')
    
    # Set the map center to the region of interest.
    Map.centerObject(region, zoom=10)
    
    return Map

def defineROI(district_name, gdf):
    area = None
    for _, row in gdf.iterrows():
        if row['NAME_2'] == district_name:
            area = shape(row['geometry'])
    return area
