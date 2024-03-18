import os, ee

dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(dir_path, 'data')

CROP_DATA_DIR = os.path.join(DATA_DIR, 'CropData')

ASSETS_DIR = os.path.join(dir_path, "assets")

DATA_content = os.listdir(DATA_DIR)
# print(f"Data content: {DATA_content}")

land_cover_dir = os.path.join(DATA_DIR, "GlobalLandcover")
# print(f"Land cover dir: {land_cover_dir}")

district_shape_file = os.path.join(DATA_DIR, 'RwandaDistricts.zip')
# print(f"District path: {district_shape_file}")

# todo: Define 'tw' and 'ts' based on empirical data or literature
tw = ee.Number(10)  # Wet baseline temperature in Celsius
ts = ee.Number(45)  # Dry baseline temperature in Celsius

# Placeholders awaiting implementation
potential_yield_factor = 1.2
model_yield_factor = 0.8

rsa_colors = {
    '26738A': '#26738A',
    '6DCFF2': '#6DCFF2',
    

}