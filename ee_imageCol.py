import ee
import folium
import random
import geopandas as gpd


# Read the shapefile; source-https://geodata.lib.utexas.edu/catalog/stanford-jg878ms7026
gdf = gpd.read_file('data/RwandaDistricts.zip')

# Convert to GeoJSON dictionary
geojson_dict = gdf.__geo_interface__
district_names = gdf['NAME_2']

CLOUD_FILTER = 70
CLD_PRB_THRESH = 60

satellite_dict = {
    'Landsat 8 Tier 1 Level-2':'LANDSAT/LC08/C02/T1_L2',
    # 'Landsat 8 Tier 2 Level-2':'LANDSAT/LC08/C02/T2_L2',
    'Sentinel-2 Level 2A':'COPERNICUS/S2_SR',
    # 'Sentinel-2 Level 1C':'COPERNICUS/S2'
}

ndvi_params = {'bands':['NDVI'],
                           'min': -1.0,
                           'max': 1.0,
                           'palette': ['blue', 'orange', 'brown', 'lightgreen', 'green', 'darkgreen']
            }

evi_params = {'bands':['EVI'],
                           'min': -1.0,
                           'max': 1.0,
                           'palette': ['blue', 'orange', 'brown', 'lightgreen', 'green', 'darkgreen']
            }

savi_params = {'bands':['SAVI'],
                           'min': -1.0,
                           'max': 1.0,
                           'palette':['blue', 'orange', 'brown', 'lightgreen', 'green', 'darkgreen']
            }

def defineAOI(district_name):
    area = None
    for feature in geojson_dict['features']:
        if feature['properties']['NAME_2'] == district_name:
            area = ee.Geometry(feature['geometry'])
    return area


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

# Add the Earth Engine layer method to folium.
folium.Map.add_ee_layer = add_ee_layer


"""
    Preprocessing functions
        - Cloud cover removal
        - Temperal filtering function
            - Reduces the noise in the time series data
"""

def maskCloudLandsat(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = 1 << 3
    cloudsBitMask = 1 << 5

    # Get the pixel QA band of Landsat 8 SR data.
    qa = image.select('QA_PIXEL')

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
        .And(qa.bitwiseAnd(cloudsBitMask).eq(0))

    # Return the masked image, scaled to reflectance, without the QA bands.
    return image.updateMask(mask)\
        .copyProperties(image, ['system:time_start'])

def maskCloudSentinel(image):
    # # Bits 10 and 11 are cloud shadow and cloud, respectively.
    # cloud_bit_mask = 1 << 10
    # cirrus_bit_mask = 1 << 11

    # # Get the QA60 band of Sentinel-2 data.
    # qa = image.select('QA60')

    # # Both flags should be set to zero, indicating clear conditions.
    # mask = qa.bitwiseAnd(cloud_bit_mask).eq(0) \
    #     .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    # # Return the masked image, scaled to reflectance.
    # return image.updateMask(mask)
    
    cld_prb = ee.Image(image.get('s2cloudless')).select('probability')

    # Condition s2cloudless by the probability threshold value.
    is_cloud = cld_prb.gt(CLD_PRB_THRESH).rename('clouds').Not()

    # Add the cloud probability layer and cloud mask as image bands.
    return image.updateMask(is_cloud)



def applyMovingAvg(image, kernel_size=3):
    kernel = ee.Kernel.square(kernel_size, 'pixels')
    smoothed = image.reduceNeighborhood(ee.Reducer.mean(), kernel).rename(image.bandNames())
    return smoothed.copyProperties(image, ["system:time_start"])


def getSampleImage(image_collection, type_='random'):
    """Function to fetch a sample image from an image collection based on type criteria"""
    image = None

    if type_ == 'first':
        # Sort the collection by time
        sorted_collection = image_collection.sort('system:time_start')
        image = sorted_collection.first()

    elif type_ == 'random':
        # Generate a random index based on the size of the image collection
        random_index = random.randint(0, image_collection.size().getInfo() - 1)

        # Fetch the image at the random index
        image = ee.Image(image_collection.toList(image_collection.size()).get(random_index))

    elif type_ == 'last':
        image = image_collection.sort('system:time_start', False).first()

    return image

def getImageDates(image_collection):
    # Sort the collection by time
    sorted_collection = image_collection.sort('system:time_start')

    # Get the earliest and latest images
    earliest_image = sorted_collection.first()
    latest_image = sorted_collection.sort('system:time_start', False).first()

    # Get the time information
    earliest_time = earliest_image.get('system:time_start').getInfo()
    latest_time = latest_image.get('system:time_start').getInfo()

    # Convert to human-readable date
    from datetime import datetime
    earliest_date = datetime.utcfromtimestamp(earliest_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
    latest_date = datetime.utcfromtimestamp(latest_time / 1000).strftime('%Y-%m-%d %H:%M:%S')

    # Print the dates
    print(f'Earliest image date: {earliest_date}')
    print(f'Latest image date: {latest_date}')

def debugLostDate(image_collection):
    # Fetch a few images from the original Landsat collection
    original_few_images = image_collection.toList(5)

    # Loop through the list and print the 'system:time_start' property
    for i in range(0, 5):
        image = ee.Image(original_few_images.get(i))
        date = image.get('system:time_start').getInfo()
        print(f"system:time_start for original image {i+1}: {date}")


def imageStats(image,roi, sat):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=30  if sat.startswith('LANDSAT') else 10,
        bestEffort=True
    )

    bands_dict = stats.getInfo()
    bands = ['SR_B4', 'SR_B3', 'SR_B2'] if sat.startswith('LANDSAT') else  ['B4', 'B3', 'B2'] # bands we want to visualize
    band_values = [bands_dict[band] for band in bands]

    return min(band_values), max(band_values), bands_dict


def landsat_scale_factors(image):
  return image.select('SR_B.').multiply(0.00001).select(['SR_B[2-7]*']) \
        .copyProperties(image, ['system:time_start'])

def Sentinel_scale_factors(image):
      return image.divide(10000).select(['B[2-9]*']).copyProperties(image, ['system:time_start'])

def importLandsat(start_date, end_date, roi, data):
    if data.startswith('LANDSAT'):
        collection = (
            ee.ImageCollection(data)\
            .filterDate(start_date, end_date) \
            .filterBounds(roi))
        return collection
    
    else:
        s2_sr_col = (ee.ImageCollection(data)
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

        # Import and filter s2cloudless.
        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
            .filterBounds(roi)
            .filterDate(start_date, end_date))

        # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_sr_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        })) 
          

"""
    SR_B4 for Red
    SR_B3 for Green
    SR_B2 for Blue
    SR_B5 for NIR (Near-Infrared)
    SR_B6 for SWIR-1 (Shortwave Infrared)
"""

# Add NDVI
def add_NDVI(image):
    bands = ['SR_B5', 'SR_B4']
    ndvi = image.normalizedDifference(bands).rename('NDVI')
    return image.addBands([ndvi])

def add_NDVIsentinel(image):
    bands = ['B8', 'B4']
    ndvi = image.normalizedDifference(bands).rename('NDVI')
    return image.addBands([ndvi])

# Add EVI
def add_EVI(image):
    bands = {'NIR': image.select('SR_B5'),
            'Red': image.select('SR_B4'),
            'Blue': image.select('SR_B2')}
    evi = image.expression(
    '2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))',
    bands
    ).rename('EVI')
    return image.addBands([evi])
        
def add_EVIsentinel(image):
    bands ={'NIR': image.select('B8'),
            'Red': image.select('B4'),
            'Blue': image.select('B2')}
    
    evi = image.expression(
        '2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))',
        bands
    ).rename('EVI')
    return image.addBands([evi])

# Add SAVI
def add_SAVI(image):
    L = 0.5  # Soil brightness correction factor
    bands = {'NIR': image.select('SR_B5'),
            'Red': image.select('SR_B4'),
            'L': L}
    savi = image.expression(
    '((NIR - Red) / (NIR + Red + L)) * (1 + L)',
    bands
    ).rename('SAVI')
    return image.addBands([savi])
        
def add_SAVIsentinel(image):
    L = 0.5  # Soil brightness correction factor
    bands ={'NIR': image.select('B8'),
            'Red': image.select('B4'),
            'L': L}
    savi = image.expression(
        '((NIR - Red) / (NIR + Red + L)) * (1 + L)',
        bands
    ).rename('SAVI')
    return image.addBands([savi])


def aggregate_monthly(image_collection, index_name):
    months = ee.List.sequence(1, 12)

    def by_month(m):
        m = ee.Number(m)
        filtered_collection = image_collection.filter(ee.Filter.calendarRange(m, m, 'month'))

        # Check if there are any images for the month
        count = filtered_collection.size()
        return ee.Algorithms.If(count.gt(0),
                                filtered_collection.mean().set({'month': m, 'index_name': index_name})
                                .select([index_name]), None)

    # Create a list to store the images
    image_list = months.map(by_month)

    # Filter out None values
    image_list = image_list.removeAll([None])

    # Convert the list to an ImageCollection
    return ee.ImageCollection.fromImages(image_list).map(lambda img: img.set('month', img.get('month')))


### Temporal Aggregation
def extract_values(image_collection, index_name, roi):
    def get_value(img):
        mean_value = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=roi, bestEffort=True).get(index_name)
        # Check for null and set a default value if null
        mean_value = ee.Algorithms.If(mean_value, mean_value, 0)
        return ee.Feature(None, {'year': img.get('year'), 'value': mean_value})
    return image_collection.map(get_value)

