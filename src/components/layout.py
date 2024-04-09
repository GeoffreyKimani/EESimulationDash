from dash import Dash, html, dcc, Input, Output, callback
from src.components.data_exploration_tab import tab_1_content
from src.components.data_analysis_tab import tab_3_content
from src.components.satellite_data_extraction_tab import tab_2_content
from src.components.yield_gap_tab import tab_yield_content


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


# ----------------------------------------------------- #
#                   TABS JOINER                        #
# ----------------------------------------------------- #
@callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "tab-1":
        return tab_2_content()  # for flipped pages
    elif tab == "tab-2":
        return tab_1_content()
    elif tab == "tab-3":
        return tab_3_content()
    elif tab == "tab-yield-gap":
        return tab_yield_content()
