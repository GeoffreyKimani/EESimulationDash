from ee_init import initialize_ee

initialize_ee()

from dash import Dash
from src.components.layout import create_layout
import dash_bootstrap_components as dbc


app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.8.1/css/all.css",
    ],
)

app.layout = create_layout(app)

if __name__ == "__main__":
    app.run_server(debug=True)
