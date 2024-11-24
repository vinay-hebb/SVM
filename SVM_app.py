import sys, os
import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from datetime import datetime
from tabulate import tabulate
from dash import dash_table
import random
def seed_everything(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

# TO DO:
# 1) Short introduction text in the web page
# 2) Add #samples to generate as UI button and add class separation button
# 3) Reduce marker size
# 4) Better plotting so that relevant {data points, lines} fills up best use of real estate(with guard distance)
# 5) Add interesting datasets for users to explore, and their nitry gritties
# 6) Reduce button width
# 7) Clean up extra memory, code
# 8) Write dual problem also

plot_button = dbc.Row([
    dcc.Graph(id='decision-boundary-plot', mathjax=True), 
    dbc.Button("Generate & Classify", id="id-plot", color="primary", size="sm")
    ])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    # https://dash.plotly.com/dash-core-components/markdown
    dcc.Markdown('''
        ### Introduction
                 
        This demo attempts to provide an insight into the various variables of SVM optimization problem. Reader is encouraged to play with demo for better insights. Few insights which a reader can understand are:  

        1. How would the values look like after finding optimal solution?
           1. $\\xi_n$ = 0 if $x_n$ lies on supporting hyperplane corresponding to its class
           2. 0 < $\\xi_n$ < 1 if $x_n$ lies between its own supporting and separating hyperplane  
           3. $\\xi_n$ > 1 if $x_n$ lies on other side of the separating hyperplane. In which case, $x_n$ is classified incorrectly  
        2. Margin also changes according to value $\\xi_n$

        ### Primal Optimization problem:        
                 
        $$
        \\begin{equation}
        \\begin{aligned}
        \\min_{\\mathbf{w},b,\\mathbf{\\xi}} \\quad & \\frac{1}{2} \\|\\mathbf{w}\\|^2 + C \\sum_{n=1}^N \\xi_n \\\\
        \\text{subject to} \\quad & y_n (<\\mathbf{w},\\mathbf{x_n}> + b) \\geq 1 - \\xi_n \\\\
        & \\xi_n \\geq 0
        \\end{aligned}
        \\end{equation}
        $$

    ''', mathjax=True),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dbc.Label("Number of Samples:"),
                dcc.Input(id='num-samples', type='number', placeholder='Enter number of samples', min=1, step=1, value=10, size="sm"),
            ], width='auto', className='mr-3'),
            
            dbc.Col([
                dbc.Label("C:"),
                dcc.Input(id='hyperparam-C', type='number', placeholder='Enter Hyperparameter C', min=0, value=1.0, size="sm"),
            ], width='auto', className='mr-3'),
            
            dbc.Col([
                dbc.Button("Generate & Classify", id="id-plot", color="primary", size="sm"),
                dbc.Button("Load data 1", id="load-data1", color="secondary", size="sm"),
                dbc.Button("Load data 2", id="load-data2", color="secondary", size="sm"),
            ], width='auto'),
        ], align='center'),
    ], fluid=True, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    dbc.Container([
        # dbc.Row([
            dcc.Graph(id='decision-boundary-plot', mathjax=True), 
            # plot_button,
            html.Div(dash_table.DataTable(id="update-table", style_header={'backgroundColor': 'white', 'fontWeight': 'bold'}))
        # ]),
    ], fluid=True, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    dcc.Markdown('''
        ## To Do:  
        1) Add provision for #samples as input  
        2) Better visualization  
        3) Add interesting datasets like moons,.. etc
        4) Discuss about nonlinear SVM  
        5) Ability to move points to get better insights into optimization problem
        '''),
    dcc.Store(id='my_state', storage_type='memory'),
])
server = app.server

def create_data(size, params):
    u, C = params
    return np.random.multivariate_normal(u, C, size=size)

def create_all_classes_data(n_samples, my_data = True):
    if my_data:
        cluster_1 = ((5,5), np.eye(2))
        cluster_2 = ((-5,-5), np.eye(2))
        X1 = create_data(n_samples//2, cluster_1)
        X2 = create_data(n_samples//2, cluster_2)
        X = np.vstack((X1, X2))
        y = np.hstack((np.ones(n_samples//2), -1*np.ones(n_samples//2)))
    else:
        X, y = make_classification(n_samples=n_samples, n_informative=2, n_redundant=0, n_features=2, n_classes=2, 
                                n_clusters_per_class=1, class_sep=2.5, flip_y=0)
    return X, y

def generate_hyperplanes(w, b, X, xx=None):
    a = -w[0] / w[1]
    if xx is None:
        xx = np.linspace(X[:, 0].min(), X[:, 0].max())
    y_hyp = a * xx - (b) / w[1]
    y_hyp1 = a * xx - (b-1) / w[1]
    y_hyp2 = a * xx - (b+1) / w[1]
    return xx, y_hyp, y_hyp1, y_hyp2

def get_plot_extremes(xx_arr, yy_arr):
    x_min, x_max, y_min, y_max = [xx_arr.min(), xx_arr.max(), yy_arr.min(), yy_arr.max()]
    mid_x, mid_y = x_min + (x_max-x_min)/2, y_min + (y_max-y_min)/2
    req_plot_side_length = max([(x_max-x_min), (y_max-y_min)])
    # print(f'{x_min, x_max, y_min, y_max}, {mid_x:.2f}, {mid_y:.2f}, {req_plot_side_length}')
    return mid_x - req_plot_side_length/2, mid_x + req_plot_side_length/2, mid_y - req_plot_side_length/2, mid_y + req_plot_side_length/2

def generate_decision_boundary(X, y, W, b, eq=True):
    df = pd.DataFrame({'X1':X[:, 0], 'X2':X[:, 1], 'y':y})
    fig = px.scatter(df, x="X1", y="X2", color="y")
    if eq:
        xx_arr, yy_arr = X[:, 0], X[:, 1]
        x_min, x_max, y_min, y_max = get_plot_extremes(xx_arr, yy_arr)
        pad_x, pad_y = 0.3, 0.3
        fig_minx, fig_maxx, fig_miny, fig_maxy = x_min - abs(pad_x*x_min), x_max + abs(pad_x*x_max), y_min - abs(pad_y*y_min), y_max + abs(pad_y*y_max)
        fig.update_xaxes(range=[fig_minx, fig_maxx])
        fig.update_yaxes(range=[fig_miny, fig_maxy])
        xx, y_hyp, y_hyp1, y_hyp2 = generate_hyperplanes(W, b, X, xx=np.linspace(fig_minx, fig_maxx))
    else:
        xx, y_hyp, y_hyp1, y_hyp2 = generate_hyperplanes(W, b, X)

    trace_hyperplane = go.Scatter(x=xx,y=y_hyp,mode='lines',line=dict(color='green', width=3),name='Hyperplane', showlegend=False)
    trace_hyperplane1 = go.Scatter(x=xx,y=y_hyp1,mode='lines',line=dict(color='green', width=3, dash='dash'),name='Hyperplane1', showlegend=False)
    trace_hyperplane2 = go.Scatter(x=xx,y=y_hyp2,mode='lines',line=dict(color='green', width=3, dash='dash'),name='Hyperplane2', showlegend=False)
    fig.add_trace(trace_hyperplane)
    fig.add_trace(trace_hyperplane1)
    fig.add_trace(trace_hyperplane2)
    # mid_idx = len(xx) // 2
    # W_vec_x, W_vec_y = xx[mid_idx] + np.array([0, W[0]]), yy[mid_idx] + np.array([0, W[1]])
    # print(f"{xx[mid_idx]}, {yy[mid_idx]}, {W_vec_x}, {W_vec_y}")
    # W_normal = go.Scatter(x=W_vec_x, y=W_vec_y, marker= dict(size=20,symbol= "arrow-bar-up", angleref="previous"), showlegend=False)
    # fig.add_trace(W_normal)
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    title_s = f'<br>Hyperplane Equations:<br>' + \
              f'{W[0]:.2f}x1 {W[1]:+.2f}x2 {b:+.2f} = 0<br>' + \
              f'{W[0]:.2f}x1 {W[1]:+.2f}x2 {b:+.2f} = -1<br>' + \
              f'{W[0]:.2f}x1 {W[1]:+.2f}x2 {b:+.2f} = 1<br>'
    fig.update_layout(title={'text': title_s, 'y': 1, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, 
                      title_font=dict(size=12), xaxis_title='$X1$', yaxis_title='$X2$', width=600, height=600, coloraxis_showscale=False)
    return fig

@app.callback(
    Output('decision-boundary-plot', 'figure'),
    Output("my_state", "data"),
    Output("update-table", "data"),
    Output("num-samples", "value"),
    Output("hyperparam-C", "value"),
    Input("id-plot", "n_clicks"),
    State("num-samples", "value"),
    State("hyperparam-C", "value"),
    State("my_state", "data"),
    Input("load-data1", "n_clicks"),
    Input("load-data2", "n_clicks"),
)
def process(n_clicks, n_samples, C, data, load_data1, load_data2):
    import pickle
    # state = np.random.get_state()
    # print("Numpy module state:", state)
    # with open('rng_state.pkl', 'wb') as f:
    #     pickle.dump(state, f)
    # with open('rng_state.pkl', 'rb') as f:
    #     loaded_state = pickle.load(f)
    # np.random.set_state(loaded_state)

    print(f'{datetime.now()} : Process : {n_clicks=}, {n_samples=}, {C=}, {load_data1=}, {load_data2=}')
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    print(changed_id)
    # import pdb; pdb.set_trace()
    if 'load-data1' in changed_id:
        with open('data1.pkl', 'rb') as f:
            X, y, n_samples, C = pickle.load(f)
        seed_everything(1)              # To keep the behavior cosnsistent when data is loaded from disk
    elif 'load-data2' in changed_id:
        with open('data2.pkl', 'rb') as f:
            X, y, n_samples, C = pickle.load(f)
        seed_everything(1)              # To keep the behavior cosnsistent when data is loaded from disk
    else:
        X, y = create_all_classes_data(n_samples, my_data=False)
    # print(X)
    # with open('data1.pkl', 'wb') as f:
    #     pickle.dump([X, y, n_samples, C], f)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    # clf = SVC(C=0.1,kernel='linear')
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    a, b = clf.coef_[0]
    c = clf.intercept_[0]
    # print(f'{a},{b},{c}')
    hyp_eqn = lambda x: np.dot(clf.coef_[0], x) + clf.intercept_[0]
    Xi_eqn = lambda x, y: 1-y*hyp_eqn(x)
    Margin = lambda x: np.abs(hyp_eqn(x)/np.linalg.norm(clf.coef_[0]))  # Considering perpendicular distance
    textbook_y = y_train
    textbook_y[textbook_y==0] = -1  # Using format as in textbook
    df = pd.DataFrame({'Support Vector: ' + r'$x_n$':[f"({x[0]:+.2f}, {x[1]:+.2f})" for x in clf.support_vectors_], 
                  'Margin': [Margin(x) for x in clf.support_vectors_],
                  r'$\\alpha_n$': clf.dual_coef_[0],
                  r'$\\xi_n': [Xi_eqn(x, textbook_y[idx]) for x, idx in zip(clf.support_vectors_, clf.support_)],
                  })
    df['On support hyperplane?'] = 0
    df['On support hyperplane?'] = df[r'$\\xi_n'] < 0.01
    print(f'Separting Hyperplane equation       : {a:.2f}x1 {b:+.2f}x2 {c:+.2f} = 0')
    print()
    print(f"Final Parameters after optimization : ")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("\nConfusion Matrix: ")
    print(confusion_matrix(y_test,y_pred))
    fig = generate_decision_boundary(X_train, y_train, clf.coef_[0], clf.intercept_[0])
    print()
    df = df.round(3).astype('str')      # https://stackoverflow.com/a/72322806/11471226
    return fig, data, df.to_dict("records"), n_samples, C

if __name__ == '__main__':
    # app.run_server(host='0.0.0.0', debug=False, port=7860)
    app.run_server(debug=True, port=7860, dev_tools_hot_reload=True)

