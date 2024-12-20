import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load dataset
df = pd.read_csv('fire_nrt_J2V-C2_538155(data1).csv')  # Replace with your dataset path
df = df.drop_duplicates()

# Ensure relevant columns are present
df = df[['latitude', 'longitude', 'brightness', 'scan', 'track', 'confidence', 'bright_t31', 'daynight', 'frp']]

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Forest Fires Dashboard"),

    html.Label("Select Confidence Level:"),
    dcc.Dropdown(
        id='confidence-dropdown',
        options=[{'label': str(level), 'value': level} for level in df['confidence'].unique()],
        multi=True,
        placeholder="Select one or more confidence levels",
        value=None
    ),

    html.Label("Select Day/Night:"),
    dcc.RadioItems(
        id='daynight-radio',
        options=[
            {'label': 'Day', 'value': 'D'},
            {'label': 'Night', 'value': 'N'},
            {'label': 'Both', 'value': None}  # Corrected value for "Both"
        ],
        value=None,  # Default value for "Both"
        inline=True
    ),

    html.Br(),

    dcc.Graph(id='fire-map'),
    dcc.Graph(id='scatter-brightness-frp'),
    dcc.Graph(id='scatter-bright-t31-frp'),
    dcc.Graph(id='bar-chart-frp')
])

# Callbacks
@app.callback(
    Output('fire-map', 'figure'),
    [Input('confidence-dropdown', 'value'),
     Input('daynight-radio', 'value')]
)
def update_map(selected_confidence, selected_daynight):
    filtered_df = df
    if selected_confidence:
        filtered_df = filtered_df[filtered_df['confidence'].isin(selected_confidence)]
    if selected_daynight:
        filtered_df = filtered_df[filtered_df['daynight'] == selected_daynight]

    fig = px.scatter_mapbox(
        filtered_df, lat="latitude", lon="longitude",
        color="frp", size="brightness",
        hover_data=['confidence', 'bright_t31', 'scan', 'track', 'daynight'],
        zoom=3, mapbox_style="carto-positron",
        title="Fire Prediction Map"
    )
    return fig

@app.callback(
    Output('scatter-brightness-frp', 'figure'),
    [Input('confidence-dropdown', 'value'),
     Input('daynight-radio', 'value')]
)
def update_scatter_brightness_frp(selected_confidence, selected_daynight):
    filtered_df = df
    if selected_confidence:
        filtered_df = filtered_df[filtered_df['confidence'].isin(selected_confidence)]
    if selected_daynight:
        filtered_df = filtered_df[filtered_df['daynight'] == selected_daynight]

    fig = px.scatter(
        filtered_df, x='brightness', y='frp',
        color='confidence', size='bright_t31',
        hover_data=['scan', 'track', 'daynight'],
        title="Brightness vs Fire Radiative Power (FRP)"
    )
    return fig

@app.callback(
    Output('scatter-bright-t31-frp', 'figure'),
    [Input('confidence-dropdown', 'value'),
     Input('daynight-radio', 'value')]
)
def update_scatter_bright_t31_frp(selected_confidence, selected_daynight):
    filtered_df = df
    if selected_confidence:
        filtered_df = filtered_df[filtered_df['confidence'].isin(selected_confidence)]
    if selected_daynight:
        filtered_df = filtered_df[filtered_df['daynight'] == selected_daynight]

    fig = px.scatter(
        filtered_df, x='bright_t31', y='frp',
        color='confidence', size='brightness',
        hover_data=['scan', 'track', 'daynight'],
        title="Brightness Temperature vs Fire Radiative Power (FRP)"
    )
    return fig

@app.callback(
    Output('bar-chart-frp', 'figure'),
    [Input('confidence-dropdown', 'value'),
     Input('daynight-radio', 'value')]
)
def update_bar_chart(selected_confidence, selected_daynight):
    filtered_df = df
    if selected_confidence:
        filtered_df = filtered_df[filtered_df['confidence'].isin(selected_confidence)]

    # Aggregate average FRP for Day and Night
    avg_frp_daynight = filtered_df.groupby('daynight')['frp'].mean().reset_index()

    # Map daynight values to more descriptive labels for the chart
    avg_frp_daynight['daynight'] = avg_frp_daynight['daynight'].map({'D': 'Day', 'N': 'Night'})

    fig = px.bar(
        avg_frp_daynight, x='daynight', y='frp',
        title="Average Fire Radiative Power (FRP) for Day and Night",
        labels={'frp': 'Average Fire Radiative Power', 'daynight': 'Time of Day'},
        color='daynight',
        color_discrete_map={'Day': 'blue', 'Night': 'orange'}
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
