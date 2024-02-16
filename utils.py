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
            return area  # Return the ROI immediately after it's found
    return area

def calculate_lst(image):
    # Extract the thermal band (Band 10 for Landsat 8)
    thermal_band = image.select('B10').multiply(0.1)  # To convert to Kelvin; the scaling factor might need adjustment

    # Constants for conversion from digital number to temperature
    ML = ee.Number(image.get('RADIANCE_MULT_BAND_10'))
    AL = ee.Number(image.get('RADIANCE_ADD_BAND_10'))
    K1 = ee.Number(image.get('K1_CONSTANT_BAND_10'))
    K2 = ee.Number(image.get('K2_CONSTANT_BAND_10'))

    # Convert digital numbers to radiance
    radiance = thermal_band.multiply(ML).add(AL)

    # Convert radiance to temperature (using constants specific to Landsat 8 Band 10)
    kelvin = radiance.expression(
        'K2 / log((K1 / radiance) + 1)', {
            'K2': K2,
            'K1': K1,
            'radiance': radiance
        })

    # Convert to Celsius
    celsius = kelvin.subtract(273.15)

    return celsius.rename('LST')

def calculate_cwsi(image, tw, ts):
    if image is None:
        raise ValueError("Image must be provided for CWSI calculation.")
    else:
        print("Image found")

    # Ensure that 'tw' and 'ts' are not None and are numbers
    if tw is None or ts is None:
        raise ValueError("Baseline temperatures 'tw' and 'ts' must be provided.")
    else:
        print(f"Tw: {tw} and Ts: {ts} found")

    # Ensure that 'tw' and 'ts' are Earth Engine numbers
    tw_ee = ee.Number(tw) if not isinstance(tw, ee.Number) else tw
    ts_ee = ee.Number(ts) if not isinstance(ts, ee.Number) else ts

    # Calculate LST
    lst = calculate_lst(image)
    print("LST Image:", type(lst))  # Debugging information

    # Apply the CWSI formula
    cwsi = lst.subtract(tw_ee).divide(ts_ee.subtract(tw_ee)).rename('CWSI')

    # Retrieve the mean CWSI value
    mean_cwsi = cwsi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=image.geometry(),
        scale=30
    )

    # Debugging: Check the mean_cwsi object
    print("mean_cwsi object:", mean_cwsi)

    # Check if the mean CWSI value is valid
    mean_cwsi_value = mean_cwsi.get('CWSI')
    if mean_cwsi_value is None:
        print("No valid CWSI value found.")
        return None
    else:
        # Attempt to get the information
        try:
            cwsi_value = mean_cwsi_value.getInfo()
            return cwsi_value
        except Exception as e:
            print("Error in retrieving CWSI value:", e)
            return None
