import ee
import folium
import pandas as pd
from constants import ts, tw
from utils import defineROI, calculate_cwsi
from shapely.geometry import Polygon

satellite_dict = {
    'Landsat 8 Tier 1 Level-2':'LANDSAT/LC08/C02/T1_L2',
    # 'Landsat 8 Tier 2 Level-2':'LANDSAT/LC08/C02/T2_L2',
    'Sentinel-2 Level 2A':'COPERNICUS/S2_SR',
    # 'Sentinel-2 Level 1C':'COPERNICUS/S2'
}

ndvi_params = {'bands':['NDVI'],
                           'min': -1.0,
                           'max': 1.0,
                           'palette': ['orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
            }

evi_params = {'bands':['EVI'],
                           'min': -1.0,
                           'max': 1.0,
                           'palette': ['orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
            }

savi_params = {'bands':['SAVI'],
                           'min': -1.0,
                           'max': 1.0,
                           'palette':['orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
            }

# Define a method for displaying Earth Engine image tiles to a folium map.
def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1, min_zoom=0):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        show=show,
        opacity=opacity,
        min_zoom=min_zoom,
        overlay=True,
        control=True
        ).add_to(self)


# Todo: could be moved up
from datetime import datetime, timedelta

# Function to calculate start and end dates for satellite image collection
def calculate_satellite_dates(year, season, period_start, period_end):
    if season == 'a season':
        season_start = datetime(year, 9, 1)  # Start of A season is Early September
    elif season == 'b season':
        season_start = datetime(year, 2, 15)  # Start of B season is Mid-Late February
    else:
        raise ValueError("Invalid season. Season should be 'a season' or 'b season'.")

    # Satellite image collection starting and ending at the specified period
    # Indices will have different collection time periods
    start_date = season_start + timedelta(days=period_start)
    end_date = season_start + timedelta(days=period_end)

    return start_date, end_date

def get_sentinel_images(aoi, start_date, end_date, cloud_probability_threshold):
    """
    Fetch Sentinel-2 images from Google Earth Engine for the given AOI and time range,
    with cloud masking based on a cloud probability threshold.

    Parameters:
    - aoi (ee.Geometry.Polygon): Area of interest as a Polygon.
    - start_date (str): Start date in the format 'YYYY-MM-DD'.
    - end_date (str): End date in the format 'YYYY-MM-DD'.
    - cloud_probability_threshold (float): Cloud probability threshold for masking.

    Returns:
    - Dictionary: Mean values of indices within the AOI.
    """
    # Load the Sentinel-2 ImageCollection and the cloud probability dataset.
    s2_collection = ee.ImageCollection('COPERNICUS/S2')
    cloud_probability_collection = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')

    # Filter the collections based on the AOI and the date range.
    filtered_s2_collection = s2_collection.filterDate(start_date, end_date).filterBounds(aoi)
    filtered_cloud_probability_collection = cloud_probability_collection.filterDate(start_date, end_date).filterBounds(aoi)

    def mask_clouds(s2_image):
        # Get the corresponding cloud probability image.
        cloud_probability_image = ee.Image(filtered_cloud_probability_collection.filterMetadata('system:index', 'equals', s2_image.get('system:index')).first())

        # Create a cloud mask where the cloud probability is less than the threshold.
        cloud_mask = cloud_probability_image.select('probability').lt(cloud_probability_threshold)

        # Apply the mask to the Sentinel-2 image.
        return s2_image.updateMask(cloud_mask)

    # Apply cloud masking to each image in the Sentinel-2 collection.
    masked_s2_collection = filtered_s2_collection.map(mask_clouds)

    # Define the add_indices function
    def add_indices(image):
        # NDVI
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

        # EVI
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')

        # SAVI
        savi = image.expression(
            '(1 + L) * (NIR - RED) / (NIR + RED + L)',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'L': 0.5
            }
        ).rename('SAVI')

        # GNDVI
        gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI')

        # Add the indices as new bands
        return image.addBands([ndvi, evi, savi, gndvi])

    # Map the add_indices function over the masked collection.
    indexed_collection = masked_s2_collection.map(add_indices)

    # Create a median composite from the indexed collection.
    median_composite = indexed_collection.median()

    # Calculate the mean values of the indices over the AOI using the median composite.
    mean_values = median_composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,  # Set the scale for Sentinel-2
        maxPixels=1e9
    ).getInfo()  # Call getInfo() here to retrieve the computed values

    # Initialize an empty dictionary to store index values
    indices_dict = {}

    # List of expected index names
    expected_indices = ['NDVI', 'EVI', 'SAVI', 'GNDVI']

    # Retrieve the mean value for each index and store it in the dictionary
    for index in expected_indices:
        if index in mean_values:
            indices_dict[index] = mean_values[index]
        else:
            # If the index is not found, print an error message
            print(f"Index {index} not found in the image. Available keys: {mean_values.keys()}")

    return median_composite, indices_dict

def get_landsat_images(aoi, start_date, end_date, initial_max_cloud_cover=10, increment_step=5, max_attempts=5):
    """
    Fetch Landsat 8 Collection 2 images from Google Earth Engine for the given AOI and time range, adjusting cloud cover threshold if necessary and applying cloud masking.

    Parameters:
    - aoi (ee.Geometry.Polygon): Area of interest as a Polygon.
    - start_date (str): Start date in the format 'YYYY-MM-DD'.
    - end_date (str): End date in the format 'YYYY-MM-DD'.
    - initial_max_cloud_cover (int): Initial maximum cloud cover percentage.
    - increment_step (int): The increment step for cloud cover percentage.
    - max_attempts (int): Maximum number of attempts to increase cloud cover threshold.

    Returns:
    - ee.Image: Aggregated image or None if no suitable images found.
    """
    max_cloud_cover = initial_max_cloud_cover
    attempt = 0

    def mask_clouds(image):
        # Get the pixel QA band from Collection 2.
        qa = image.select('QA_PIXEL')
        # Clouds are represented as bit 1.
        cloud = qa.bitwiseAnd(1 << 1)
        # Mask pixels with clouds.
        return image.updateMask(cloud.Not())

    while attempt < max_attempts:
        # Create an ImageCollection for Landsat 8 Collection 2
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')\
            .filterDate(start_date, end_date)\
            .filterBounds(aoi)\
            .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))

        print("Landsat collected")

        # Apply the cloud masking function
        masked_collection = collection.map(mask_clouds)
        print(f"Landsat masked: {masked_collection.size().getInfo()}")

        # Check if the collection has any images
        if masked_collection.size().getInfo() > 0:
            print(f"Landsat collection found {masked_collection.size().getInfo()} ...")
            medianComposite = masked_collection.median()
            cwsi = None #calculate_cwsi(medianComposite, tw, ts) # compute the CWSI index

            # Calculate NDVI.
            ndvi = medianComposite.normalizedDifference(['B5', 'B4']).rename('NDVI')

            # Calculate EVI.
            evi = medianComposite.expression(
                '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
                    'NIR': medianComposite.select('B5'),
                    'RED': medianComposite.select('B4'),
                    'BLUE': medianComposite.select('B2')
                }).rename('EVI')

            # Calculate SAVI. Assuming L = 0.5
            savi = medianComposite.expression(
                '(1 + L) * (NIR - RED) / (NIR + RED + L)', {
                    'NIR': medianComposite.select('B5'),
                    'RED': medianComposite.select('B4'),
                    'L': 0.5
                }).rename('SAVI')

            # Calculate GNDVI.
            gndvi = medianComposite.normalizedDifference(['B5', 'B3']).rename('GNDVI')

            # Add the indices to the original image.
            compositeWithIndices = medianComposite.addBands([ndvi, evi, savi, gndvi])
            
            return compositeWithIndices, {"CWSI":cwsi, "EVI":evi, "NDVI":ndvi, "SAVI":savi, "GNDVI":gndvi}

        # Increment the cloud cover threshold for the next attempt
        max_cloud_cover += increment_step
        attempt += 1

    print(f"No suitable images found within {max_attempts} attempts. Returning an empty collection.")
    # Return None if no suitable images found
    return None, {{"CWSI":None, "EVI":None, "NDVI":None, "SAVI":None, "GNDVI":None}}

def get_indices_with_fallback(aoi_ee, start_date, end_date, initial_threshold=20, increment=5, max_threshold=50):
    threshold = initial_threshold
    indices_dict = {}
    image = None

    while threshold <= max_threshold:
        image, indices_dict = get_sentinel_images(aoi_ee, start_date, end_date, cloud_probability_threshold=threshold)
        if all(value is not None for value in indices_dict.values()):
            break
        threshold += increment

    return image, indices_dict

def integrate_indices_to_dataframe(df, satellite="LANDSAT"):
    for index, row in df.iterrows():
        # Extract the coordinates from the Shapely Polygon object
        if isinstance(row['geometry'], Polygon):
            aoi_coordinates = list(row['geometry'].exterior.coords)
            aoi_ee = ee.Geometry.Polygon(aoi_coordinates)
        else:
            print(f"Row {index} does not contain a valid Polygon object.")
            continue

        # Get the indices dictionary with a fallback loop for NaN values
        indices_dict = {}
        band_img = None
        if satellite=="LANDSAT":
            print("Processing landsat ...")
            band_img, indices_dict = get_landsat_images(aoi_ee, row['start_date'], row['end_date'])
        else:
            band_img, indices_dict = get_indices_with_fallback(aoi_ee, row['start_date'], row['end_date'])
        

        # Update the dataframe with the new columns for each index
        for key, value in indices_dict.items():
            # Create column if it does not exist
            if key not in df.columns:
                df[key] = None

            # Update value
            if pd.isna(df.at[index, key]) and value is not None:
                df.at[index, key] = value

    return band_img, df