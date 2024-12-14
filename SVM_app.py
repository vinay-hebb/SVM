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
import pickle
from types import SimpleNamespace
def seed_everything(seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
seed_everything(1)              # To keep the behavior cosnsistent when data is loaded from disk

# TO DO:
# 2) Add interesting datasets for users to explore, and their nitry gritties
# 3) Write dual problem also

split = False
th_to_call_sample_on_hyp_plane = 0.001
pad_x, pad_y = 0.3, 0.3

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    # https://dash.plotly.com/dash-core-components/markdown
    dcc.Markdown('''
        ### Goal
                 
        This demo attempts to provide an insight into the various variables of SVM optimization problem. Reader is encouraged to play with demo for better insights. 

        ### Interactive Demo:
        Few points to Note:  
        1) Fewer samples makes it easier to get insights  
        2) Few existing datasets are generated to get quick insights about slack variables  
        3) When we load existing datasets, input varaibles can change in UI components, please keep an eye on that  
    '''),
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
                dbc.Button("Generate", id="id-generate", color="primary", size="sm"),
                dbc.Button("Classify", id="id-classify", color="primary", size="sm"),
            ], width='auto', className='mr-3'),
            
            dbc.Col([
                dbc.Button("Load dataset 1", id="load-data1", color="secondary", size="sm"),
                dbc.Button("Load dataset 2", id="load-data2", color="secondary", size="sm"),
            ], width='auto'),
        ], align='center'),
    ], fluid=True, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='decision-boundary-plot', mathjax=True)
            ], width=6),  # width=6 means it will take 6/12 of the row width
            dbc.Col([
                html.Div(dash_table.DataTable(id="update-table", style_header={'backgroundColor': 'white', 'fontWeight': 'bold'}))
            ], width=6),  # width=6 means it will take 6/12 of the row width
            html.Div(id='error-message'),
        ], align='center'),
    ], fluid=False, style={'display': 'flex', 'align-items': 'center', 'justify-content': 'left'}),
    dcc.Markdown('''
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

        ### Insights         
        Few insights which a reader can understand are:  

        1. How would the values look like after finding optimal solution?
           1. $\\xi_n$ = 0 if $x_n$ lies on supporting hyperplane corresponding to its class
           2. 0 < $\\xi_n$ < 1 if $x_n$ lies between its own supporting and separating hyperplane  
           3. $\\xi_n$ > 1 if $x_n$ lies on other side of the separating hyperplane. In which case, $x_n$ is classified incorrectly  
        2. Margin also changes according to value $\\xi_n$

        ## To Do:  
        1) Add interesting datasets like moons,.. etc  
        2) Discuss about nonlinear SVM  
        3) Ability to move points to get better insights into optimization problem  
        4) Ability to generate data as per the inputs of user (amount of overlap, variance, ...etc)  
        5) For extreme inputs, hyperplanes may not be visible (though they are plotted, they are just outside of 'meaningful' limits). This will be fixed soon.  
        ''', mathjax=True),
    dcc.Store(id='my_state', storage_type='memory'),
])
server = app.server

hyp_eqn = lambda clf, x: np.dot(clf.coef_[0], x) + clf.intercept_[0]
Xi_eqn = lambda clf, x, y: 1-y*hyp_eqn(clf, x)
Margin = lambda clf, x: np.abs(hyp_eqn(clf, x)/np.linalg.norm(clf.coef_[0]))  # Considering perpendicular distance

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

def generate_decision_boundary(fig, X, y, W, b, fig_minx, fig_maxx, eq=True):
    if eq:
        xx, y_hyp, y_hyp1, y_hyp2 = generate_hyperplanes(W, b, X, xx=np.linspace(fig_minx, fig_maxx))
    else:
        xx, y_hyp, y_hyp1, y_hyp2 = generate_hyperplanes(W, b, X)

    trace_hyperplane = go.Scatter(x=xx,y=y_hyp,mode='lines',line=dict(color='green', width=3),name='Separting Hyperplane', showlegend=False)
    trace_hyperplane1 = go.Scatter(x=xx,y=y_hyp1,mode='lines',line=dict(color='green', width=3, dash='dash'),name='Supporting Hyperplane1', showlegend=False)
    trace_hyperplane2 = go.Scatter(x=xx,y=y_hyp2,mode='lines',line=dict(color='green', width=3, dash='dash'),name='Supporting Hyperplane2', showlegend=False)
    fig.add_trace(trace_hyperplane)
    fig.add_trace(trace_hyperplane1)
    fig.add_trace(trace_hyperplane2)
    # mid_idx = len(xx) // 2
    # W_vec_x, W_vec_y = xx[mid_idx] + np.array([0, W[0]]), yy[mid_idx] + np.array([0, W[1]])
    # print(f"{xx[mid_idx]}, {yy[mid_idx]}, {W_vec_x}, {W_vec_y}")
    # W_normal = go.Scatter(x=W_vec_x, y=W_vec_y, marker= dict(size=20,symbol= "arrow-bar-up", angleref="previous"), showlegend=False)
    # fig.add_trace(W_normal)
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    title_s = f'<br>Separating hyperplane: {W[0]:.2f}x1 {W[1]:+.2f}x2 {b:+.2f} = 0<br>' + \
              f'Supporting hyperplane: {W[0]:.2f}x1 {W[1]:+.2f}x2 {b:+.2f} = -1<br>' + \
              f'Supporting hyperplane: {W[0]:.2f}x1 {W[1]:+.2f}x2 {b:+.2f} = 1<br>'
    fig.update_layout(title={'text': title_s, 'y': 1, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, 
                      title_font=dict(size=12), xaxis_title='$X1$', yaxis_title='$X2$', width=600, height=600, coloraxis_showscale=False)
    return fig

def classify(fig, data, n_samples, C):
    # print(data, n_samples, C)
    perpendicular_projections = True
    if C != data.C:
        data.C = C
    if n_samples != data.n_samples:
        # print('Regenerated data as num_samples was modified after Generate button and before classify button was clicked')
        # data.n_samples = n_samples
        # X, y = create_all_classes_data(n_samples, my_data=False)
        # data = SimpleNamespace(X=X, y=y, n_samples=n_samples, C=C)
        df_tmp_table = pd.DataFrame({'Support Vector: ' + r'$x_n$':[np.nan], 
                'Margin': [np.nan],
                r'$\\alpha_n$': [np.nan],
                r'$\\xi_n': [np.nan],
                })
        msg = html.Div(dcc.Markdown('*Number of samples was modified after Generating data. Please regenerate*'), style={'color': 'red', 'font-size': '24px'})
        return data, fig, df_tmp_table.to_dict("records"), n_samples, C, msg
    else:
        X, y, n_samples, C, fig_minx, fig_maxx, fig_miny, fig_maxy = data.X, data.y, data.n_samples, data.C, data.fig_minx, data.fig_maxx, data.fig_miny, data.fig_maxy
    
    X, y = np.array(X), np.array(y)
    if split == True:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    else:
        X_train, y_train = X, y
        
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_train, y_train)
    
    if split == True:
        y_pred = clf.predict(X_test)
    # print(f'{a},{b},{c}')
    textbook_y = y_train
    textbook_y[textbook_y==0] = -1  # Using format as in textbook
    df = pd.DataFrame({'Support Vector: ' + r'$x_n$':[f"({x[0]:+.2f}, {x[1]:+.2f})" for x in clf.support_vectors_], 
                  'Margin': [Margin(clf, x) for x in clf.support_vectors_],
                  r'$\\alpha_n$': clf.dual_coef_[0],
                  r'$\\xi_n': [Xi_eqn(clf, x, textbook_y[idx]) for x, idx in zip(clf.support_vectors_, clf.support_)],
                  })
    df['On support hyperplane?'] = df[r'$\\xi_n'] < th_to_call_sample_on_hyp_plane
    # print(f'Separting Hyperplane equation       : {a:.2f}x1 {b:+.2f}x2 {c:+.2f} = 0')
    # print()
    # print(f"Final Parameters after optimization : ")
    # print(tabulate(df, headers='keys', tablefmt='psql'))
    # print("\nConfusion Matrix: ")
    # print(confusion_matrix(y_test,y_pred))
    fig = generate_decision_boundary(fig, X_train, y_train, clf.coef_[0], clf.intercept_[0], fig_minx, fig_maxx)
    
    if perpendicular_projections:   # Draw perpendicular lines showing margin distance
        sv = clf.support_vectors_
        w = clf.coef_[0]
        b = clf.intercept_[0]
        w_norm = np.sqrt(np.sum(w**2))
        unit_normal = w / w_norm
        
        for point in sv:
            dist = (np.dot(w, point) + b) / w_norm
            proj_point = point - dist * unit_normal
            fig.add_trace(go.Scatter(x=[point[0], proj_point[0]], 
                                    y=[point[1], proj_point[1]],
                                    mode='lines',
                                    line=dict(color='red', width=1, dash='dot'),
                                    showlegend=False))

        fig.add_trace(go.Scatter(x=sv[:, 0], y=sv[:, 1], mode='markers', name='Support Vectors',
                                marker=dict(size=30, line=dict(width=3, color='red'),opacity=0.3,color='rgba(0,0,0,0)'), 
                                showlegend=False))

    df = df.round(3).astype('str')
    msg = html.Div(dcc.Markdown('Classified Samples'), style={'color': 'green'})
    return data, fig, df.to_dict("records"), n_samples, C, msg

def default_data(n_samples, C):
    msg = html.Div(dcc.Markdown(''))
    fig = go.Figure(data=[go.Scatter(x=[], y=[])])
    fig.update_layout(xaxis=dict(range=[-2, 2]), yaxis=dict(range=[-2, 2]), width=600, height=600)
    return SimpleNamespace().__dict__, fig, pd.DataFrame().to_dict("records"), n_samples, C, msg


@app.callback(
    Output("my_state", "data"),
    Output('decision-boundary-plot', 'figure'),
    Output("update-table", "data"),
    Output("num-samples", "value"),
    Output("hyperparam-C", "value"),
    Output("error-message", "children"),
    Input("id-generate", "n_clicks"),
    Input("id-classify", "n_clicks"),
    Input("load-data1", "n_clicks"),
    Input("load-data2", "n_clicks"),
    State("my_state", "data"),
    State("num-samples", "value"),
    State("hyperparam-C", "value"),
    State('decision-boundary-plot', 'figure'),
)
def callback_entry(generate_n_clicks, classify_n_clicks, 
                   load_data1_n_clicks, load_data2_n_clicks,
                   data, n_samples, C, existing_fig):
    print(f'{datetime.now()} : Starting callback_entry : {generate_n_clicks=}, {classify_n_clicks=}, {load_data1_n_clicks=}, {load_data2_n_clicks=}, {n_samples=}, {C=}')
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'id-generate' in changed_id or 'load-data1' in changed_id or 'load-data2' in changed_id:
        state_data = SimpleNamespace()
        if n_samples > 10000:
            msg = html.Div(dcc.Markdown('Lets not misuse free resource! ;)'), style={'color': 'red', 'font-size': '24px'})
            return (*default_data(n_samples, C)[:-1], msg)
        if 'id-generate' in changed_id:
            X, y = create_all_classes_data(n_samples, my_data=False)
        elif 'load-data1' in changed_id:
            with open('all_xi_ne_0.pkl', 'rb') as f:
                X, y, n_samples, C = pickle.load(f)
        elif 'load-data2' in changed_id:
            with open('some_xi_ne_0.pkl', 'rb') as f:
                X, y, n_samples, C = pickle.load(f)
        state_data = SimpleNamespace(X=X, y=y, n_samples=n_samples, C=C)
        df = pd.DataFrame({'X1':X[:, 0], 'X2':X[:, 1], 'y':y})
        fig = px.scatter(df, x="X1", y="X2", color="y")
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        fig.update_layout(xaxis_title='$X1$', yaxis_title='$X2$', width=600, height=600, coloraxis_showscale=False)
        equal_aspect=True
        if equal_aspect:
            x_min, x_max, y_min, y_max = get_plot_extremes(X[:, 0], X[:, 1])
            dist_x, dist_y = x_max - x_min, y_max - y_min
            fig_minx, fig_maxx, fig_miny, fig_maxy = x_min - abs(pad_x*dist_x), x_max + abs(pad_x*dist_x), y_min - abs(pad_y*dist_y), y_max + abs(pad_y*dist_y)
            # print(X)
            # print(fig_minx, fig_maxx, fig_miny, fig_maxy)
            fig.update_xaxes(range=[fig_minx, fig_maxx])
            fig.update_yaxes(range=[fig_miny, fig_maxy])
            state_data.__dict__.update(fig_minx=fig_minx, fig_maxx=fig_maxx, fig_miny=fig_miny, fig_maxy=fig_maxy)
        # else: TO DO: to be handled
        df_tmp_table = pd.DataFrame({'Support Vector: ' + r'$x_n$':[np.nan], 
                'Margin': [np.nan],
                r'$\\alpha_n$': [np.nan],
                r'$\\xi_n': [np.nan],
                })
        msg = html.Div(dcc.Markdown('Generated Samples'), style={'color': 'green'})
        print(f'{datetime.now()} : Ending callback_entry')
        return state_data.__dict__, fig, df_tmp_table.to_dict("records"), n_samples, C, msg
    elif 'id-classify' in changed_id:
        data = SimpleNamespace(**data)
        if not data.__dict__:
            msg = html.Div(dcc.Markdown('Please generate samples first and then classify'), style={'color': 'red', 'font-size': '24px'})
            return (*default_data(n_samples, C)[:-1], msg)
        existing_fig = go.Figure(existing_fig)
        existing_fig.data = [trace for trace in existing_fig.data if ('Hyperplane' not in trace.name) and ('Suppport Vectors' not in trace.name)]
        state, fig, df, n_samples, C, msg = classify(existing_fig, data, n_samples, C)
        print(f'{datetime.now()} : Ending callback_entry')
        return state.__dict__, fig, df, n_samples, C, msg
    else:
        print(f'{datetime.now()} : Ending callback_entry')
        return default_data(n_samples, C)
                   

if __name__ == '__main__':
    if 'SPACE_ID' in os.environ:
        app.run_server(host='0.0.0.0', debug=False, port=7860)
    else:
        app.run_server(debug=True, port=7860, dev_tools_hot_reload=True)

