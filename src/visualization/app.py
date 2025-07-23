import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from plotly.subplots import make_subplots
import numpy as np

wandb_data = pd.read_csv('/workspaces/ml-fairness-thesis/src/visualization/wandb_export.csv', sep=',')
wandb_data['weights'] = wandb_data['weights'].fillna('')
wandb_data['dp_weight'] = wandb_data['weights'].map(fairopt.to_dp_weight)

group_by_keys = ['Name', 'dataset', 'sensitive_feature', 'dp_weight']
aggregation = [
    'train_classification_metrics.accuracy',
    'train_demographic_parity_difference',
    'train_demographic_parity_ratio',
    'test_classification_metrics.accuracy',
    'test_demographic_parity_difference',
    'test_demographic_parity_ratio',
]

datasets = pd.DataFrame(
    wandb_data[['dataset', 'sensitive_feature']].value_counts().index.values.tolist(), 
    columns=['dataset', 'sensitive_feature'])
wandb_data = wandb_data.groupby(group_by_keys)[aggregation].mean()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H4('Interactive scatter plot with Iris dataset'),
    dcc.Graph(id="plot"),
    html.P("Dataset:"),
    dcc.Dropdown(
        datasets['dataset'].unique(),
        'adult',
        id='dataset-dropdown'
    ),
    html.P("Feature:"),
    dcc.Dropdown(
        datasets['dataset'].unique(),
        'adult',
        id='feature-dropdown'
    ),
    html.P("Dataset split:"),
    dcc.Dropdown(
        ['train', 'test'],
        'train',
        id='split-dropdown'
    ),
])

@app.callback(
    dash.Output("feature-dropdown", "options"),  
    dash.Input("dataset-dropdown", "value")
)
def update_features(dataset_name):
    return datasets[datasets['dataset'] == dataset_name]['sensitive_feature']

@app.callback(
    dash.Output("feature-dropdown", "value"),
    dash.Input("feature-dropdown", "options")
)
def set_feature(features):
    return features[0]

@app.callback(
    dash.Output("plot", "figure"),  
    [
        dash.Input("dataset-dropdown", "value"), 
        dash.Input("feature-dropdown", "value"),
        dash.Input("split-dropdown", "value"),
    ]
)
def update_plot(dataset_name, sensitive_feature, split):
    accuracy_vs_dp = wandb_data.xs(dataset_name, level=1).xs(sensitive_feature, level=1).reset_index()
    fig_px = px.scatter(
        accuracy_vs_dp,
        y = f'{split}_classification_metrics.accuracy',
        x = f'{split}_demographic_parity_difference',
        color = 'Name',
        symbol = 'dp_weight'
    )

    if dataset_name == 'adult':
        aifds = AdultDataset()
    elif dataset_name == 'german':
        aifds = GermanDataset()
    else:
        aifds = CompasDataset()

    aifdf, aifattrs = aifds.convert_to_dataframe()
    aifmetadata = aifds.metadata

    # Sensitive feature values
    fvalues = list(aifmetadata['protected_attribute_maps'][aifattrs['protected_attribute_names'].index(sensitive_feature)].keys())
    maxfvalue = max(fvalues, key=lambda v: np.sum(aifdf[sensitive_feature] == v))

    A = (aifdf[sensitive_feature] == maxfvalue).astype(int)
    Y = (aifdf[aifds.label_names[0]] == [*aifmetadata['label_maps'][0]][0]).astype(int)
    
    lst_dp = np.linspace(0, 1, num=128)
    opts = [fairopt.curve_dp(A, Y, i, 'DP') for i in lst_dp]

    accs = [acc for acc, _ in opts]
    y1s = [np.sum(np.sum(cm, axis = 0)[0]) / A.shape[0] for _, cm in opts]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=lst_dp, y=accs, name="Accuracy"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=lst_dp, y=y1s, name="Y1 %"),
        secondary_y=True,
    )

    for d in fig_px.data:
        fig.add_trace(d)

    # Set x-axis title
    fig.update_xaxes(title_text="DI")

    fig.update_yaxes(range=[0, 1], secondary_y=True)

    # Set y-axes titles
    fig.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig.update_yaxes(title_text="Positive prediction %", secondary_y=True)

    return fig

app.run_server(debug=True)