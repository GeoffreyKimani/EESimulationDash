import folium, os
import geopandas as gpd
from constants import DATA_DIR, potential_yield_factor, model_yield_factor
from src.utils.step_one import load_data_for_crop


def create_simple_rwanda_map():
    # Coordinates for the center of Rwanda
    rwanda_coords = [-1.9403, 29.8739]

    # Create a Folium map centered on Rwanda
    return folium.Map(location=rwanda_coords, zoom_start=8)


def create_rwanda_map():
    # Coordinates for the center of Rwanda and zoom level
    rwanda_coords = [-1.9403, 29.8739]
    map_zoom_start = 8.5

    # Load GeoJSON for Rwanda (replace 'rwanda_geojson_path' with the path to your GeoJSON file)
    rwanda_geojson_path = os.path.join(DATA_DIR, "RwandaDistricts.zip")
    rwanda_gdf = gpd.read_file(rwanda_geojson_path)

    # Create the base map
    m = folium.Map(location=rwanda_coords, zoom_start=map_zoom_start)

    # Create an inverse mask using the GeoJSON
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    inverse_mask = gpd.overlay(
        world, rwanda_gdf, how="difference", keep_geom_type=False
    )

    # Add the inverse mask to the map as a semi-transparent overlay
    # This will effectively "hide" areas outside Rwanda by covering them with the mask
    folium.GeoJson(
        inverse_mask,
        style_function=lambda feature: {
            "fillColor": "black",
            "color": "black",
            "fillOpacity": 0.8,
            "weight": 0.1,
        },
    ).add_to(m)

    # Add Rwanda boundary on top of the mask
    folium.GeoJson(
        rwanda_gdf,
        style_function=lambda feature: {
            "fillColor": "none",
            "color": "blue",
            "weight": 1,
        },
    ).add_to(m)

    return m


def aggregate_data(crop, years_filter, season_filter, aggregation_type, map_idx):
    """
    Aggregate data based on the district, season, and year, for a given yield column.

    Parameters:
    - crop: Crop type to filter the data.
    - years_filter: List of years to include in the aggregation.
    - season_filter: List of seasons to include in the aggregation.
    - aggregation_type: 'mean' or 'sum' to specify the type of aggregation.
    - map_idx: Indicator to specify which yield to compute ('actual_yield', 'potential_yield', 'yield_gap', 'predicted_yield').

    Returns:
    - Aggregated DataFrame.
    """
    # Load the crop dataframe
    df = load_data_for_crop(crop)

    # Filter the DataFrame by season and year if specified
    if season_filter:
        df = df[df["season"].isin([season_filter])]

    if years_filter:
        df = df[df["year"].isin([years_filter])]

    # Determine grouping columns based on filters applied
    filter_list = ["district"]
    if season_filter:
        filter_list.append("season")
    if years_filter:
        filter_list.append("year")

    # Perform the grouping
    grouped = df.groupby(filter_list)
    aggregated_df = None

    # Perform the aggregation based on the specified type
    if map_idx == "actual_yield":
        if aggregation_type == "mean_value":
            aggregated_df = grouped["yield_kg_ph"].mean().reset_index()
        elif aggregation_type == "sum_total":
            aggregated_df = grouped["yield_kg_ph"].sum().reset_index()

    elif map_idx == "potential_yield":
        if aggregation_type == "mean_value":
            aggregated_df = grouped["yield_kg_ph"].mean().reset_index()
            aggregated_df["yield_kg_ph"] = (
                aggregated_df["yield_kg_ph"] * potential_yield_factor
            )
        elif aggregation_type == "sum_total":
            aggregated_df = grouped["yield_kg_ph"].sum().reset_index()
            aggregated_df["yield_kg_ph"] = (
                aggregated_df["yield_kg_ph"] * potential_yield_factor
            )

    elif map_idx == "predicted_yield":
        if aggregation_type == "mean_value":
            aggregated_df = grouped["yield_kg_ph"].mean().reset_index()
            aggregated_df["yield_kg_ph"] = (
                aggregated_df["yield_kg_ph"] * model_yield_factor
            )
        elif aggregation_type == "sum_total":
            aggregated_df = grouped["yield_kg_ph"].sum().reset_index()
            aggregated_df["yield_kg_ph"] = (
                aggregated_df["yield_kg_ph"] * model_yield_factor
            )

    else:
        # Yield Gap (Potential - Actual)
        if aggregation_type == "mean_value":
            aggregated_df = grouped["yield_kg_ph"].mean().reset_index()
            aggregated_df["yield_kg_ph"] = (
                aggregated_df["yield_kg_ph"] * potential_yield_factor
            ) - aggregated_df["yield_kg_ph"]
        elif aggregation_type == "sum_total":
            aggregated_df = grouped["yield_kg_ph"].sum().reset_index()
            aggregated_df["yield_kg_ph"] = (
                aggregated_df["yield_kg_ph"] * potential_yield_factor
            ) - aggregated_df["yield_kg_ph"]

    # Add a column to convert yield to tonnes for both aggregation types
    aggregated_df["yield_tonne_ph"] = round((aggregated_df["yield_kg_ph"] / 1000), 2)
    # print(aggregated_df)

    return aggregated_df


from dash import html

yield_ranges = [
    "up to 1",
    "1 - 2",
    "2 - 3",
    "3 - 4",
    "4 - 5",
    "5 - 6",
    "6 - 7",
    "7 - 8",
    "8 - 9",
    "9 - 10",
    "10 - 11",
    "11 - 12",
    "12 - 13",
    "13 - 14",
    "14 - 15",
    "more than 15",
]

colors = [
    # Browns - less healthy vegetation
    "#8B4513",  # Saddle Brown
    "#A0522D",  # Sienna
    # Reds - unhealthy vegetation
    "#CD5C5C",  # Indian Red
    "#F08080",  # Light Coral
    # Yellows - dry vegetation
    "#FFD700",  # Gold
    "#FFFF00",  # Yellow
    # Oranges - somewhat dry vegetation
    "#FFA500",  # Orange
    "#FF8C00",  # Dark Orange
    # Orange-Green - transition to healthy vegetation
    "#9ACD32",  # Yellow Green
    "#32CD32",  # Lime Green
    # Avocado Greens - average health vegetation
    "#556B2F",  # Dark Olive Green
    "#6B8E23",  # Olive Drab
    # Blue Greens - healthy vegetation
    "#3CB371",  # Medium Sea Green
    "#2E8B57",  # Sea Green
    # Greens - healthy vegetation
    "#008000",  # Green
    "#006400",  # Dark Green
]


def create_legend():
    # Determine the number of ranges that should go in the first column
    half_length = len(yield_ranges) // 2

    # Create two columns of legend items
    left_column = [
        create_legend_item(yield_range, color)
        for yield_range, color in zip(yield_ranges[:half_length], colors[:half_length])
    ]
    right_column = [
        create_legend_item(yield_range, color)
        for yield_range, color in zip(yield_ranges[half_length:], colors[half_length:])
    ]

    # Combine columns into the legend layout
    legend_layout = html.Div(
        [
            html.Div(
                "Map Legend (tonnes per hectare)",
                style={"font-weight": "bold", "margin-bottom": "5px"},
            ),  # Header
            html.Div(
                left_column, style={"display": "inline-block", "verticalAlign": "top"}
            ),
            html.Div(
                right_column, style={"display": "inline-block", "verticalAlign": "top"}
            ),
        ],
        style={
            "border": "2px solid lightgrey",
            "padding": "10px",
            "border-radius": "8px",
            "background-color": "white",
            "box-shadow": "0 2px 4px rgba(0,0,0,0.2)",
            "max-width": "250px",  # Set a max-width if necessary
            "display": "inline-block",  # Align with the table
            "vertical-align": "top",  # Align with the bottom of the table
            "margin-right": "10px",  # Space between the table and legend
        },
    )

    return legend_layout


# Helper function to create a single legend item
def create_legend_item(yield_range, color):
    return html.Div(
        [
            html.Div(
                style={
                    "background-color": color,
                    "width": "20px",
                    "height": "20px",
                    "display": "inline-block",
                }
            ),
            html.Span(f" {yield_range}", style={"margin-left": "5px"}),
        ],
        style={"display": "flex", "align-items": "center", "margin": "2px 0"},
    )


# Map coloring
# -------------


# Define your color scale function
def get_color(yield_value):
    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    for i, threshold in enumerate(thresholds):
        if yield_value <= threshold:
            return colors[i]
    return colors[-1]


def prepare_for_merge(gjson, df, geojson_key="NAME_2", df_key="district"):
    # Strip whitespace and convert to consistent case
    gjson[geojson_key] = gjson[geojson_key].str.strip().str.lower()
    df[df_key] = df[df_key].str.strip().str.lower()

    # Check for special characters and remove them
    # Note: You might want to specify which characters to remove if any
    gjson[geojson_key] = gjson[geojson_key].str.replace("[^\w\s]", "", regex=True)
    df[df_key] = df[df_key].str.replace("[^\w\s]", "", regex=True)

    # Ensure data types are consistent
    gjson[geojson_key] = gjson[geojson_key].astype(str)
    df[df_key] = df[df_key].astype(str)

    # Return cleaned data
    return gjson, df


def color_map(df):
    # Load the GeoJSON data for Rwanda's districts
    rwanda_geojson_path = os.path.join(DATA_DIR, "RwandaDistricts.zip")
    rwanda_gjson = gpd.read_file(rwanda_geojson_path)

    # print(rwanda_gjson)

    # 'df' has district names and yield_tonne_ph values load it and merge it with the GeoJSON properties
    rwanda_gjson, df_aggregated = prepare_for_merge(rwanda_gjson, df)
    merged_gdf = rwanda_gjson.merge(
        df_aggregated, how="left", left_on="NAME_2", right_on="district"
    )
    merged_gdf = merged_gdf.dropna(subset=["district"])

    # print("\n\nMerged df")
    # print(merged_gdf)

    # Create the base map
    rwanda_map = create_rwanda_map()

    # Create a Folium GeoJson object, applying the color scale function to each feature
    folium.GeoJson(
        merged_gdf,
        style_function=lambda feature: {
            "fillColor": get_color(feature["properties"]["yield_tonne_ph"]),
            "color": "black",  # Boundary color
            "weight": 1,
            "fillOpacity": 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME_2", "yield_tonne_ph"],
            aliases=["District:", "Yield (tonnes/ha):"],
            localize=True,
        ),
    ).add_to(rwanda_map)

    return rwanda_map
