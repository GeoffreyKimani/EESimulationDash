from dash import Dash, html, dcc


def create_layout(app: Dash) -> html.Div:
    return html.Div(
        [
            dcc.Tabs(
                id="tabs",
                value="tab-1",
                children=[
                    dcc.Tab(
                        label="Satellite Data Extraction",
                        value="tab-1",
                        id="tab-1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        label="Data Exploration",
                        value="tab-2",
                        id="tab-2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        label="Data Analysis",
                        value="tab-3",
                        id="tab-3",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        label="Yield Gap", value="tab-yield-gap"
                    ),  # New tab for Yield Gap
                ],
                className="custom-tabs",
            ),
            html.Div(id="tabs-content"),
            # define the data storage globally
            dcc.Store(id="stored-data"),  # To store the filtered DataFrame
            dcc.Store(id="gdf-data"),  # To store the GDF DataFrame
            dcc.Store(id="features-df-store"),  # Stores the features for modeling
            dcc.Store(
                id="preprocessing-df-store"
            ),  # Stores the selected features for processing
            dcc.Store(id="modeling-df-store"),  # Stores the data after preprocessing
            dcc.Store(id="test-data-store"),  # Stores the data for model evaluation
            dcc.Store(
                id="selected-crop"
            ),  # Stores the selected crop from the first tab
        ],
        style={"textAlign": "center"},
    )
