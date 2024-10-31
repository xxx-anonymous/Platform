import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, dash_table
import zmq
import random

# Initialize data
data = {'nodes': {}, 'links': []}
rows = []
x_max = 80
iterations = []
accs = []
tips_num = []
client_num = 0
fig = go.Figure(layout={"template": "simple_white",
                        "xaxis": {"title": "迭代", "constrain": "domain"},
                        "yaxis": {"title": "", "constrain": "domain"},
                        "width": 1380,
                        "height": 500
                        })
acc_fig1 = go.Figure()
acc_fig2 = go.Figure()
trace1 = None
trace2 = None
Tr = [html.Tr([html.Td('服务器'), html.Td("在线", style={'color': 'green'}), html.Td('-'), html.Td('-'), html.Td('-')]), ]


def update_all(msg):
    '''生成节点'''
    global data, fig, rows, accs, acc_fig1, acc_fig2, iterations, trace1, trace2, client_num, Tr
    if msg['type'] == 1:
        data['nodes'][msg['ID']] = msg
        '''添加link'''
        link_ls = []
        if len(msg['previous']) != 0:
            for p in msg['previous']:
                link_temp = {'source': p, 'target': msg['ID']}
                data['links'].append(link_temp)
                link_ls.append(link_temp)

        '''绘制节点和边'''
        fig.add_trace(
            go.Scatter(
                x=[msg['located'][0]],
                y=[msg['located'][1]],
                text=msg['ID'] + '<br>Hash: ' + str(msg['Trans_hash']) + '<br>Accuracy: {:.4f}'.format(msg['Accuracy']),
                mode='markers',
                name=msg['ID'],
                marker=dict(symbol='square', size=20, line=dict(width=2))
            )
        )
        if len(msg['previous']) != 0:
            for link in link_ls:
                node_source = data['nodes'][link['source']]
                node_target = msg
                fig.add_annotation(
                    x=node_source['located'][0],
                    y=node_source['located'][1],
                    xref='x', yref='y',
                    ax=node_target['located'][0],
                    ay=node_target['located'][1],
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2, arrowsize=2,
                    arrowwidth=1, standoff=10,
                    arrowcolor='black', startstandoff=5
                )
        current_x_max = max([node['located'][0] for node in data['nodes'].values()])
        new_x_min = max(0, current_x_max - x_max)
        fig.update_xaxes(range=[new_x_min, current_x_max + 1])

        fig.update_layout(
            legend_traceorder='reversed'
        )
        # 限制横轴最大值
        if msg['located'][0] > x_max:
            fig.update_xaxes(range=[msg['located'][0] - x_max - 1, msg['located'][0] + 1])

        previous_trans_str = ''
        temp = 1
        for p in msg['previous']:
            previous_trans_str += 'Previous {}: '.format(temp) + p + ' '
            temp += 1
        rows.insert(0, {'T': msg['Trans_hash'], 'Pb': msg['ID'],
                        'PT': previous_trans_str, 'Acc': msg['Accuracy']})

    elif msg['type'] == 0:
        accs.append(round(msg['Accuracy'], 5))
        tips_num.append(msg['Tips_num'])
        iterations.append(len(accs) * 20)
        trace1 = go.Scatter(x=iterations, y=accs, mode="lines", name="Accuracy")
        trace2 = go.Scatter(x=iterations, y=tips_num, mode="lines", name="Tips_num")
        acc_fig1 = go.Figure(data=trace1, layout={
            "title": {"text": "全局模型精度", "font": {"size": 20}},
            "template": "ggplot2",
            "xaxis": {"title": "迭代"},
            "yaxis": {"title": "精度"}
        })
        acc_fig2 = go.Figure(data=trace2, layout={
            "title": {"text": "DAG区块链Tips数量", "font": {"size": 20}},
            "template": "seaborn",
            "xaxis": {"title": "迭代"},
            "yaxis": {"title": "Tips数量"}
        })

        client_num = msg['Client_num']
        Client_trans_num = msg['Client_trans_num']  # 列表
        Client_approved_num = msg['Client_approved_num']  # 列表

        td = [0] * (client_num + 1)
        for i in range(1, client_num + 1):
            if (Client_approved_num[i] / (Client_trans_num[i] + 0.00001)) < 0.8:
                td[i] = html.Td("异常！", style={'color': 'red'})
            else:
                td[i] = html.Td("正常", style={'color': 'green'})

        Tr = [html.Tr([html.Td(i), html.Td("在线", style={'color': 'green'}), html.Td(Client_trans_num[i]),
                       html.Td(round(Client_approved_num[i] / (Client_trans_num[i] + 0.00001), 2)),
                       td[i]])for i in range(1, client_num + 1)]


# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout with navigation tabs
tabs_content = dbc.Tabs(
    [
        dbc.Tab(label="DAG监控平台", tab_id="tab-1", children=[
            dbc.Row(dbc.Label('基于DAG区块链的异步FL可视化平台',
                              style={'font-size': '24px', 'text-align': 'center', 'margin': '0 auto'}),
                    justify='center'),
            dcc.Graph(id='graph', figure=fig,
                      style={'border-radius': '8px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                             'background-color': '#f9f9f9', 'padding': '10px', 'margin-bottom': '50px'}),
            html.Div([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("块高度"),
                    dbc.CardBody(html.H4("0", id="block-count"))
                ], color="info", inverse=True)),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("客户端数量"),
                    dbc.CardBody(html.H4("0", id="client-count"))
                ], color="success", inverse=True)),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("全局模型精度"),
                    dbc.CardBody(html.H4("0.00%", id="current-accuracy"))
                ], color="warning", inverse=True)),
                dbc.Col(dbc.Card([
                    dbc.CardHeader("当前Tips数量"),
                    dbc.CardBody(html.H4("0", id="tips-number"))
                ], color="danger", inverse=True)),
            ], style={'display': 'flex', 'justify-content': 'space-around', 'gap': '50px'})
        ]),
        dbc.Tab(label="性能曲线图", tab_id="tab-2", children=[
            html.Div([
                dcc.Graph(id='accuracy1', figure=acc_fig1,
                          style={'flex': '1', 'border-radius': '8px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                                 'background-color': '#f9f9f9', 'padding': '10px'}),
                dcc.Graph(id='accuracy2', figure=acc_fig2,
                          style={'flex': '1', 'border-radius': '8px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                                 'background-color': '#f9f9f9', 'padding': '10px'})
            ], style={'display': 'flex', 'flex-wrap': 'nowrap', 'justify-content': 'space-around', 'gap': '50px'}),
            html.Div([
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("全局模型精度", style={'color': 'black'}),
                        dbc.CardBody(html.H4("0.00%", id="real-time-accuracy", style={'color': 'black'}))
                    ], color="light", inverse=False)),  # inverse设为False以避免影响颜色
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("当前Tips数量", style={'color': 'black'}),
                        dbc.CardBody(html.H4("0", id="real-time-tips", style={'color': 'black'}))
                    ], color="light", inverse=False)),  # inverse设为False以避免影响颜色
                ], justify='center', style={'margin-top': '20px'})
            ])
        ]),
        dbc.Tab(label="交易信息监控", tab_id="tab-3", children=[
            dbc.Row(dbc.Label('DAG区块链交易信息表',
                              style={'font-size': '24px', 'text-align': 'center', 'margin': '0 auto'}),
                    justify='center'),
            dash_table.DataTable(
                id='table',
                columns=[{'name': '交易ID', 'id': 'Pb'},
                         {'name': '交易哈希', 'id': 'T'},
                         {'name': '前向交易', 'id': 'PT'},
                         {'name': '模型精度', 'id': 'Acc'}],
                page_size=50,
                data=rows,
                style_table={'maxWidth': '1400px', 'maxHeight': '1200px', 'overflowY': 'scroll', 'margin': '0 auto',
                             'border-radius': '8px', 'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                             'background-color': '#f9f9f9', 'padding': '10px'},
                style_cell={'textAlign': 'center', 'minWidth': '180px', 'width': '180px', 'maxWidth': '200px'},
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto'
                }
            )
        ]),
        dbc.Tab(label="客户端状态监控", tab_id="tab-4", children=[
            dbc.Table([
                html.Thead(
                    html.Tr([html.Th("设备 ID"), html.Th("设备状态"), html.Th("交易数量"), html.Th("验证率"), html.Th("预警状态")])),
                html.Tbody(Tr, id='Tr')
            ], bordered=True, hover=True, responsive=True, striped=True)
        ])
    ]
)

# App layout
app.layout = html.Div([
    tabs_content,
    dcc.Interval(id="interval", interval=1 * 1000, n_intervals=0),
], style={'background-color': '#f9f9f9', 'padding': '20px', 'border-radius': '8px',
          'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)'})


# ZMQ-related functions
def create_and_connect_zmq_socket(zmq_port="5411"):
    context = zmq.Context()
    zmq_socket = context.socket(zmq.SUB)
    zmq_socket.connect("tcp://localhost:%s" % zmq_port)
    zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    return zmq_socket


@app.callback(
    [Output('graph', 'figure'), Output('table', 'data'), Output('accuracy1', 'figure'), Output('accuracy2', 'figure'),
     Output('block-count', 'children'), Output('client-count', 'children'), Output('current-accuracy', 'children'),
     Output('tips-number', 'children'),
     Output('real-time-accuracy', 'children'), Output('real-time-tips', 'children'), Output('Tr', 'children')],
    [Input('interval', 'n_intervals')]
)
def update_components(n_intervals):
    if not hasattr(update_components, "zmq_socket"):
        update_components.zmq_socket = create_and_connect_zmq_socket()

    try:
        msg = update_components.zmq_socket.recv_pyobj(flags=zmq.NOBLOCK)
        update_all(msg)
    except zmq.Again:
        pass

    block_count = len(data['nodes'])
    client_count = client_num  # Update this based on actual client count
    current_accuracy = f"{round(accs[-1] * 100, 2)}%" if accs else "0.00%"
    tips_number = tips_num[-1] if tips_num else 0

    return (fig, rows, acc_fig1, acc_fig2, block_count, client_count, current_accuracy, tips_number,
            current_accuracy, tips_number, Tr)


if __name__ == '__main__':
    app.run_server(debug=False, port=8080)
