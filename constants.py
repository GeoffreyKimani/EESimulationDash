import os

dir_path = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(dir_path, 'data')
# print(f"Data root: {DATA_DIR}")

ASSETS_DIR = os.path.join(dir_path, "assets")

DATA_content = os.listdir(DATA_DIR)
# print(f"Data content: {DATA_content}")

land_cover_dir = os.path.join(DATA_DIR, "GlobalLandcover")
# print(f"Land cover dir: {land_cover_dir}")

district_shape_file = os.path.join(DATA_DIR, 'RwandaDistricts.zip')
# print(f"District path: {district_shape_file}")