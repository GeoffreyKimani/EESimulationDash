import ee

def initialize_ee():
    try:
        ee.Initialize(project='ee-waoformation')
        print("Earth Engine initialized successfully.")
    except ee.EEException:
        print("Failed to initialize Earth Engine.")
        # Optionally, attempt to authenticate if needed
        ee.Authenticate()
        ee.Initialize()
