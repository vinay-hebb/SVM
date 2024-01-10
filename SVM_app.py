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
        ### Primal Optimization problem:
        This demo tries to provide an insight into the various variables of SVM optimization problem
        Few things which can be understood are:
        1) How would the values look like after finding optimal solution?
           a) \xi_n = 0 if x_n lies on supporting hyperplane corresponding to its class
           b) 0 < \xi_n < 1 if x_n lies between its own supporting hyperplane and separating hyperplane
           c) \xi_n > 1 if x_n lies on other side of the separating hyperplane. In which case, x_n is classified incorrectly
        2) Margin can also according to value \xi_n
        To Do:
        1) Add provision for #samples as input
        2) Better visualization
        3) Add interesting datasets like moons,.. etc
        4) Discuss about nonlinear SVM
        
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
            dbc.Col(plot_button),
            dbc.Col(html.Div(dash_table.DataTable(id="update-table", style_header={'backgroundColor': 'white', 'fontWeight': 'bold'})))
        ]),
    ], fluid=True),
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

def generate_hyperplanes(w, b, X):
    a = -w[0] / w[1]
    xx = np.linspace(X[:, 0].min(), X[:, 0].max())
    yy = a * xx - (b) / w[1]
    yy1 = a * xx - (b-1) / w[1]
    yy2 = a * xx - (b+1) / w[1]
    return xx, yy, yy1, yy2

def generate_decision_boundary(X, y, W, b, eq=True):
    df = pd.DataFrame({'X1':X[:, 0], 'X2':X[:, 1], 'y':y})
    fig = px.scatter(df, x="X1", y="X2", color="y")
    xx, yy, yy1, yy2 = generate_hyperplanes(W, b, X)
    trace_hyperplane = go.Scatter(x=xx,y=yy,mode='lines',line=dict(color='green', width=3),name='Hyperplane', showlegend=False)
    trace_hyperplane1 = go.Scatter(x=xx,y=yy1,mode='lines',line=dict(color='green', width=3, dash='dash'),name='Hyperplane1', showlegend=False)
    trace_hyperplane2 = go.Scatter(x=xx,y=yy2,mode='lines',line=dict(color='green', width=3, dash='dash'),name='Hyperplane2', showlegend=False)
    fig.add_trace(trace_hyperplane)
    fig.add_trace(trace_hyperplane1)
    fig.add_trace(trace_hyperplane2)
    # mid_idx = len(xx) // 2
    # W_vec_x, W_vec_y = xx[mid_idx] + np.array([0, W[0]]), yy[mid_idx] + np.array([0, W[1]])
    # print(f"{xx[mid_idx]}, {yy[mid_idx]}, {W_vec_x}, {W_vec_y}")
    # W_normal = go.Scatter(x=W_vec_x, y=W_vec_y, marker= dict(size=20,symbol= "arrow-bar-up", angleref="previous"), showlegend=False)
    # fig.add_trace(W_normal)
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.update_layout(title=f'Hyp eqn : {W[0]:.2f}x1 {W[1]:+.2f}x2 {b:+.2f} = 0',xaxis_title='$X1$',yaxis_title='$X2$',width=600, height=600, coloraxis_showscale=False)
    if eq:
        xx_arr = yy_arr = np.empty((0))
        xx_arr = np.append(xx_arr, X[:, 0], axis=0)
        xx_arr = np.append(xx_arr, xx, axis=0)
        yy_arr = np.append(yy_arr, X[:, 1], axis=0)
        yy_arr = np.append(yy_arr, yy, axis=0)
        yy_arr = np.append(yy_arr, yy1, axis=0)
        yy_arr = np.append(yy_arr, yy2, axis=0)
        x_min, x_max, y_min, y_max = [xx_arr.min(), xx.max(), yy_arr.min(), yy_arr.max()]
        mid_x, mid_y = x_min + (x_max-x_min)/2, y_min + (y_max-y_min)/2
        s = max([(x_max-x_min), (y_max-y_min)])
        print(f'{x_min, x_max, y_min, y_max}, {mid_x:.2f}, {mid_y:.2f}, {s}')
        fig.update_xaxes(range=[mid_x - s/2, mid_x + s/2])
        fig.update_yaxes(range=[mid_y - s/2, mid_y + s/2])
    return fig

@app.callback(
    Output('decision-boundary-plot', 'figure'),
    Output("my_state", "data"),
    Output("update-table", "data"),
    Input("id-plot", "n_clicks"),
    State("my_state", "data"),
)
def process(n_clicks, data):
    size = 20

    X, y = create_all_classes_data(size, my_data=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    a, b = clf.coef_[0]
    c = clf.intercept_[0]
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
    return fig, data, df.to_dict("rows")

if __name__ == '__main__':
    app.run_server(debug=True, port=1111)
