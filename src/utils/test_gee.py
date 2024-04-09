import ee
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Initialize the Earth Engine module.
ee.Initialize()

# Test fetching a dataset.
image = ee.Image('CGIAR/SRTM90_V4')
print(image.getInfo())