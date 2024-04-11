# Import packages
from dash import Dash, html, dcc, dash_table, callback, Output, Input
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import stats
from datetime import datetime

crs = [
'#48AFF4','#5DDA93',
'#B582D9','#F9B26C',
'#6797E0','#E76A87',
'#1CCEC5','#F4D35D',
'#F391D4','#9CA4B4',
'#9AD3F9','#8BDFB0',
'#FACA9B','#CCA6D9',
'#9FB5D6','#F598AE',
'#8DE2DE','#F9E494',
'#F4C2E4','#BFC4CF']

# Format title
def format_title(title, subtitle=None,font_size=20, subtitle_font_size=10):
    title =  f'<span style="font-size: {font_size}px;margin-bottom:50px;font-family:Roboto Medium;color:#2D426A">{title}</span>'
    if not subtitle:
        return title
    subtitle = f'<span style="font-size: {subtitle_font_size}px;font-family:Roboto;color:#899499">{subtitle}</span>'
    return f'{title}<br>{subtitle}' #removing a <br> for now

# for percentiles
def ordinal(x: str):
    if int(x) in range(11,20) or int(x) in [10*i for i in range(11)]:
        return x+('th' if x!='0' else '')
    elif x[-1] == '1':
        return x+'st'
    elif x[-1] == '2':
        return x+'nd'
    elif x[-1] == '3':
        return x+'rd'
    elif int(x[-1]) in range(4,10):
        return x+'th' 

# sample company: Asana (381043)
company = 'Asana'
users = pd.read_csv(Path(__file__).parent.parent/'data'/'pw_sample_users.csv')
users['percentile'] = users['prestige_v2'].apply(lambda x: stats.percentileofscore(users['prestige_v2'], x, kind='rank'))

prestige_ref = pd.read_csv(Path(__file__).parent.parent/'data'/'pw_prestige_percentiles.csv')
positions = pd.read_csv(Path(__file__).parent.parent/'data'/'pw_sample_positions.csv')
positions = positions[positions['ultimate_parent_rcid']==381043]

#initializing default cohorts
b1_in, t1_in, b2_in, t2_in = 0, 10, 90, 100
#lower prestige
group1_in = users[users['percentile'].between(b1_in,t1_in, inclusive='both')]#['user_id']
#higher prestige
group2_in = users[users['percentile'].between(b2_in,t2_in, inclusive='both')]#['user_id']

tmp1 = prestige_ref.copy()

raw_prestige1_in = np.mean(group1_in['prestige_v2'])
raw_prestige2_in = np.mean(group2_in['prestige_v2'])
tmp1['tmp_col1'] = np.abs(tmp1['PRESTIGE']-raw_prestige1_in)
tmp1['tmp_col2'] = np.abs(tmp1['PRESTIGE']-raw_prestige2_in)
prestige_avg1_in = int(round(tmp1[tmp1['tmp_col1']==(tmp1['tmp_col1'].min())]['PERCENTILE'].values[0]*100,0))
prestige_avg2_in = int(round(tmp1[tmp1['tmp_col2']==(tmp1['tmp_col2'].min())]['PERCENTILE'].values[0]*100,0))
age_avg1_in = round(np.mean((datetime.now() - pd.to_datetime(group1_in['birthday'], format='%Y-%m-%d')).apply(lambda x: x.days/365.25)), 1)
age_avg2_in = round(np.mean((datetime.now() - pd.to_datetime(group2_in['birthday'], format='%Y-%m-%d')).apply(lambda x: x.days/365.25)), 1)
positions1_in = positions[positions['user_id'].isin(group1_in['user_id'].unique())]
positions2_in = positions[positions['user_id'].isin(group2_in['user_id'].unique())]
salary_avg1_in = '$'+str(round(int(round(np.sum(positions1_in['weight_v2']*positions1_in['salary'])/np.sum(positions1_in['weight_v2']), 0))/1000, 1))+'K'
salary_avg2_in = '$'+str(round(int(round(np.sum(positions2_in['weight_v2']*positions2_in['salary'])/np.sum(positions2_in['weight_v2']), 0))/1000, 1))+'K'

median_t1 = pd.to_datetime(positions1_in[positions1_in['enddate'].isnull()]['startdate'], format='%Y-%m-%d').median().strftime('%Y-%m')
median_t2 = pd.to_datetime(positions2_in[positions2_in['enddate'].isnull()]['startdate'], format='%Y-%m-%d').median().strftime('%Y-%m')

stats_in = [
    {
        'Cohort': 'Higher Prestige', 'Prestige': prestige_avg2_in,
        'Avg Salary': salary_avg2_in, 'Avg Age': age_avg2_in, 'Median Start': median_t2
    },
    {
        'Cohort': 'Lower Prestige', 'Prestige': prestige_avg1_in, 
        'Salary': salary_avg1_in, 'Age': age_avg1_in, 'Median Start': median_t1
    }
    ]


# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# App layout Define Asana prestige quantile cohorts
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div("Define Asana prestige quantile cohorts:", className="me-0"),
            xs=2, sm=2, md=2, lg=2, xl=2, xxl=2, className="me-0", style={'margin-right': '0px', 'margin-top': '10px', 'font-size': '1.2em'}
        ),
        dbc.Col(
            dcc.RangeSlider(0, 100, value=[b1_in,t1_in,b2_in,t2_in],
                            marks=None, pushable=10, allowCross=False,
                            tooltip={"placement": "bottom", "always_visible": True,"style": {"fontSize": "12px"}},
                            id='slider-groups'),
            xs=4, sm=4, md=4, lg=4, xl=4, xxl=4, align="center", className="ms-0 float-start", style={'margin-left': '0px', 'margin-right': '0px'}
        ),
        dbc.Col(
            dash_table.DataTable(
                id='stats',
                data=stats_in,
                style_cell={'textAlign': 'center'}),
            xs=5, sm=5, md=5, lg=5, xl=5, xxl=5, align="left", style={'margin-left': '0px', 'margin-right': '0px'}
        )
    ], className="g-0 gap-0 justify-content-start align-bottom"
    ),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='roles_bar', config={'displayModeBar': False}),
            # width="auto"
            xs=4
            # xs=4, sm=4, md=4, lg=4, xl=4, xxl=4, style={"height": "100%"}, align="center"
        ),
        dbc.Col(
            dcc.Graph(id='seniority_bar', config={'displayModeBar': False}),
            # width="auto"
            xs=4, align="left"
            # xs=2, sm=2, md=2, lg=2, xl=2, xxl=2, style={"height": "100%"}, align="center"
        ),
        dbc.Col(
            dcc.Graph(id='msa_bar',  config={'displayModeBar': False}),
            # width="auto"
            xs=4
            # xs=4, sm=4, md=4, lg=4, xl=4, xxl=4, style={"height": "100%"}, align="right"
        )
    ], className="justify-content-start align-bottom h-75 g-0 gap-0"
    ),
], fluid = True, style={"height": "100vh"})


# Add controls to build the interaction
@callback(
    Output(component_id='stats', component_property='data'),
    Output(component_id='roles_bar', component_property='figure'),
    Output(component_id='seniority_bar', component_property='figure'),
    Output(component_id='msa_bar', component_property='figure'),
    Input(component_id='slider-groups', component_property='value')
)
def update_graph(ranges_chosen):
    b1, t1, b2, t2 = ranges_chosen 

    #lower prestige
    group1 = users[users['percentile'].between(b1,t1, inclusive='both')]#['user_id']
    #higher prestige
    group2 = users[users['percentile'].between(b2,t2, inclusive='both')]#['user_id']

    tmp2 = prestige_ref.copy()

    raw_prestige1 = np.mean(group1['prestige_v2'])
    raw_prestige2 = np.mean(group2['prestige_v2'])
    tmp2['tmp_col1'] = np.abs(tmp2['PRESTIGE']-raw_prestige1)
    tmp2['tmp_col2'] = np.abs(tmp2['PRESTIGE']-raw_prestige2)
    prestige_avg1 = int(round(tmp2[tmp2['tmp_col1']==(tmp2['tmp_col1'].min())]['PERCENTILE'].values[0]*100,0))
    prestige_avg2 = int(round(tmp2[tmp2['tmp_col2']==(tmp2['tmp_col2'].min())]['PERCENTILE'].values[0]*100,0))
    age_avg1 = round(np.mean((datetime.now() - pd.to_datetime(group1['birthday'], format='%Y-%m-%d')).apply(lambda x: x.days/365.25)), 1)
    age_avg2 = round(np.mean((datetime.now() - pd.to_datetime(group2['birthday'], format='%Y-%m-%d')).apply(lambda x: x.days/365.25)), 1)
    positions1 = positions[positions['user_id'].isin(group1['user_id'].unique())]
    positions2 = positions[positions['user_id'].isin(group2['user_id'].unique())]
    salary_avg1 = '$'+str(round(int(round(np.sum(positions1['weight_v2']*positions1['salary'])/np.sum(positions1['weight_v2']), 0))/1000, 1))+'K'
    salary_avg2 = '$'+str(round(int(round(np.sum(positions2['weight_v2']*positions2['salary'])/np.sum(positions2['weight_v2']), 0))/1000, 1))+'K'
    
    mediant1 = pd.to_datetime(positions1[positions1['enddate'].isnull()]['startdate'], format='%Y-%m-%d').median().strftime('%Y-%m')
    mediant2 = pd.to_datetime(positions2[positions2['enddate'].isnull()]['startdate'], format='%Y-%m-%d').median().strftime('%Y-%m')

    stats_out = [
    {
        'Cohort': 'Higher Prestige', 'Prestige': prestige_avg2,
        'Salary': salary_avg2, 'Age': age_avg2, 'Median Start': mediant2
    },
    {
        'Cohort': 'Lower Prestige', 'Prestige': prestige_avg1, 
        'Salary': salary_avg1, 'Age': age_avg1, 'Median Start': mediant1
    }
    ]

    roles2 = (positions2['role_k150'].value_counts()/positions2['role_k150'].value_counts().sum()).to_frame()
    roles2.columns = ['higher_prestige_pct']
    roles1 = (positions1['role_k150'].value_counts()/positions1['role_k150'].value_counts().sum()).to_frame()
    roles1.columns = ['lower_prestige_pct']
    roles = roles2.merge(roles1, how='outer', left_index=True, right_index=True).fillna(0)
    roles['diff'] = roles['higher_prestige_pct']-roles['lower_prestige_pct']
    roles_head = roles.sort_values('diff', ascending=False).head(5).sort_values('diff').reset_index()
    roles_tail = roles.sort_values('diff', ascending=True).head(5).reset_index()
    
    roles_fig = go.Figure()
    trace = go.Bar(
        y = roles_tail['role_k150'],
        x = roles_tail['diff'],
        textfont=dict(family="Roboto", size=10, color="white"),
        marker_color=crs[5],
        text=round(roles_tail["diff"]* 100, 1).astype(str) + " p.p.",
        customdata = round(roles_tail["diff"]* -100, 1).astype(str) + " p.p.",
        hovertemplate = '<b>%{y}</b> roles are a <b>%{customdata}</b> smaller share of Asana\'s higher prestige cohort.<extra></extra>',
        orientation = 'h',
        textposition = 'auto'
    )
    roles_fig.add_trace(trace)
    trace = go.Bar(
        y = [' '],
        x = [0], 
        orientation = 'h', 
        showlegend = False,
        marker_line_color='rgba(0,0,0,0)'
    )
    roles_fig.add_trace(trace)
    trace = go.Bar(
        y = roles_head['role_k150'],
        x = roles_head['diff'],
        textfont=dict(family="Roboto", size=10, color="white"),
        marker_color=crs[1],
        text=round(roles_head["diff"]* 100, 1).astype(str) + " p.p.",
        customdata = round(roles_head["diff"]* 100, 1).astype(str) + " p.p.",
        hovertemplate = '<b>%{y}</b> roles are a <b>%{customdata}</b> greater share of Asana\'s higher prestige cohort.<extra></extra>',
        orientation = 'h',
        textposition = 'auto'
    )
    roles_fig.add_trace(trace)
    t = f'<span style="color:{crs[1]};">High-</span> and <span style="color:{crs[5]};">low-</span>prestige roles'
    s = "Percentage-point difference in cohort makeup"
    roles_fig.update_layout(
        barmode = 'group', 
        yaxis = dict(
            zeroline = False,
            gridcolor = '#EAECF0', 
            gridwidth = 1, 
            showgrid = False,
            tickfont = dict(family="Roboto", size=10, color="#2D426A")
        ),
        xaxis = dict(
            visible = False, 
            zeroline = True, 
            showgrid = False, 
            zerolinecolor = '#EAECF0',
            zerolinewidth = 1,
            gridcolor = '#EAECF0',
            gridwidth = 1,
            tickformat = '.0%',
            showticklabels = True,
            tickfont = dict(family="Roboto", size=16, color="#2D426A")
        ),
        showlegend = False,
        title = dict(text = format_title(t,s), 
                    yanchor = 'top', y = 0.85, xanchor = 'left', x = 0.02),
        plot_bgcolor = 'rgba(0,0,0,0)',
        # height=500,
        hoverlabel = dict(bgcolor = 'white', bordercolor = '#2D426A', font = dict(size=10, family='Roboto Mono', color="#2D426A")),
        margin = dict(
            l = 0,  # default: 80
            r = 40,  # default: 80
            b = 0,  # default: 80
            t = 100, # default: 100
            pad = 0  # default: 0
            )
    )
    roles_fig.add_hline(y = 5, line_width = 2, line_color = 'black', line_dash = "dash"),
    roles_fig.update_traces(textangle=0)
    #roles_fig.update_xaxes(range=[roles_tail['diff'].min()*(1.5), roles_head['diff'].max()*(1.5)])

    seniorities2 = positions2.assign(seniority=np.where(positions2['seniority']>3, 'Manager+', 'Entry/Junior/Associate'))
    seniorities2 = (seniorities2['seniority'].value_counts()/seniorities2['seniority'].value_counts().sum()).to_frame().reset_index()
    seniorities1 = positions1.assign(seniority=np.where(positions1['seniority']>3, 'Manager+', 'Entry/Junior/Associate'))
    seniorities1 = (seniorities1['seniority'].value_counts()/seniorities1['seniority'].value_counts().sum()).to_frame().reset_index()

    seniorities_fig = make_subplots(rows=2, cols=1, 
                                    specs=[[{"type": "pie"}],[{"type": "pie"}]], 
                                    subplot_titles=("Higher prestige employees", "Lower prestige employees"))

    #cd2 = np.stack((seniorities2['count'].apply(lambda x: str(round(x*100, 1))+'%'), seniorities2['seniority']), axis=-1)
    seniorities_fig.add_trace(go.Pie(labels=seniorities2['seniority'], values=seniorities2['count'], domain = dict(y=[0, 0.5]),
                                     marker_colors=[crs[3], crs[4]], 
                                     textposition='outside', textinfo='percent+label',texttemplate='%{percent:.1%}',#texttemplate='%{label}<br>%{percent:.1%}',
                                     #customdata=[str(round(x*100, 1))+'%' for x in seniorities2.values],
                                     customdata=seniorities2['count'].apply(lambda x: str(round(x*100, 1))+'%'),
                                     hovertemplate= '<b>%{customdata}</b> of higher prestige Asana employees are <b>%{label}</b> level.<extra></extra>'
                                     ), 
                                     row=1,col=1)

    #cd1 = np.stack((seniorities1['count'].apply(lambda x: str(round(x*100, 1))+'%'), seniorities1['seniority']), axis=-1)
    seniorities_fig.add_trace(go.Pie(labels=seniorities1['seniority'], values=seniorities1['count'], domain = dict(y=[0.5, 1]),
                                     marker_colors=[crs[3], crs[4]], 
                                     textposition='outside', textinfo='percent+label',texttemplate='%{percent:.1%}',#texttemplate='%{label}<br>%{percent:.1%}'
                                     #customdata=[str(round(x*100, 1))+'%' for x in seniorities1.values],
                                     customdata=seniorities1['count'].apply(lambda x: str(round(x*100, 1))+'%'),
                                     hovertemplate= '<b>%{customdata}</b> of lower prestige Asana employees are <b>%{label}</b> level.<extra></extra>'
                                     ),
                                     row=2,col=1)
    
    #seniorities_fig.update_traces(textposition='outside', textinfo='percent+label',texttemplate='%{label}<br>%{percent:.1%}')



    t = f'Seniority makeup'
    seniorities_fig.update_layout(
        yaxis = dict(title = "",
                    zeroline = False,
                    gridcolor = '#EAECF0',
                    gridwidth = 1,
                    #side = "left", 
                    tickfont = dict(family = "Roboto", size = 14, color = "#2D426A")#,autorange = "reversed"
                    ),
        xaxis = dict(title = "",
                     zeroline = True,
                     zerolinecolor = '#EAECF0',
                     zerolinewidth = 1,
                     tickformat = '%',
                     gridcolor = '#EAECF0',
                     gridwidth = 1,
                     tickfont = dict(family = "Roboto Mono", size = 16, color = "#2D426A"),
                     visible = True,
                     tickangle=360
                     #title_standoff = 50
                ),
        showlegend = True, 
        legend=dict(bgcolor='rgba(0,0,0,0)', xanchor='center', yanchor='middle', x=0.5, y=0.53, font = dict(family="Roboto", size=10, color="#2D426A"),
                    traceorder='reversed'), 
        font = dict(family = "Roboto", size = 10, color = "#2D426A"),
        title = dict(text = format_title(t), 
                    yanchor = 'top', y = 0.87, xanchor = 'center', x = 0.5, xref='paper'),
        plot_bgcolor='rgba(0,0,0,0)',
        #height = 500,
        hoverlabel = dict(
            bgcolor = "white",
            bordercolor = "#2D426A",
            font = dict(size=10, family='Roboto Mono', color="#2D426A")
        ),
        margin = dict(
            l = 40,  # default: 80
            r = 40,  # default: 80
            b = 0,  # default: 80
            t = 100, # default: 100
            pad = 0  # default: 0
            )
    )
    seniorities_fig.update_annotations(font=dict(size=10))

    metros2 = (positions2['metro_area'].value_counts()/positions2['metro_area'].value_counts().sum()).to_frame()
    metros2.columns = ['higher_prestige_pct']
    metros1 = (positions1['metro_area'].value_counts()/positions1['metro_area'].value_counts().sum()).to_frame()
    metros1.columns = ['lower_prestige_pct']
    metros = metros2.merge(metros1, how='outer', left_index=True, right_index=True).fillna(0)
    metros['diff'] = metros['higher_prestige_pct']-metros['lower_prestige_pct']
    metros_head = metros.sort_values('diff', ascending=False).head(5).sort_values('diff').reset_index()
    metros_tail = metros.sort_values('diff', ascending=True).head(5).reset_index()

    metros_fig = go.Figure()
    trace = go.Bar(
        y = metros_tail['metro_area'].apply(lambda x: x.split(' metropolitan area')[0].title()),
        x = metros_tail['diff'],
        textfont=dict(family="Roboto", size=10, color="white"),
        marker_color=crs[5],
        text=round(metros_tail["diff"]* 100, 1).astype(str) + " p.p.",
        customdata = round(metros_tail["diff"]* -100, 1).astype(str) + " p.p.",
        hovertemplate = '<b>%{y}</b> employees are a <b>%{customdata}</b> smaller share of Asana\'s higher prestige cohort.<extra></extra>',
        orientation = 'h',
        textposition = 'auto'
    )
    metros_fig.add_trace(trace)
    trace = go.Bar(
        y = [' '],
        x = [0], 
        orientation = 'h', 
        showlegend = False,
        marker_line_color='rgba(0,0,0,0)'
    )
    metros_fig.add_trace(trace)
    trace = go.Bar(
        y = metros_head['metro_area'].apply(lambda x: x.split(' metropolitan area')[0].title()),
        x = metros_head['diff'],
        textfont=dict(family="Roboto", size=10, color="white"),
        marker_color=crs[1],
        text=round(metros_head["diff"]* 100, 1).astype(str) + " p.p.",
        customdata = round(metros_head["diff"]* 100, 1).astype(str) + " p.p.",
        hovertemplate = '<b>%{y}</b> employees are a <b>%{customdata}</b> greater share of Asana\'s higher prestige cohort.<extra></extra>',
        orientation = 'h',
        textposition = 'auto'
    )
    metros_fig.add_trace(trace)
    t = f'<span style="color:{crs[1]};">High-</span> and <span style="color:{crs[5]};">low-</span>prestige areas'
    s = "Percentage-point difference in cohort makeup"
    metros_fig.update_layout(
        barmode = 'group', 
        yaxis = dict(
            zeroline = False,
            gridcolor = '#EAECF0', 
            gridwidth = 1, 
            showgrid = False,
            tickfont = dict(family="Roboto", size=10, color="#2D426A")
        ),
        xaxis = dict(
            visible = False, 
            zeroline = True, 
            showgrid = False, 
            zerolinecolor = '#EAECF0',
            zerolinewidth = 1,
            gridcolor = '#EAECF0',
            gridwidth = 1,
            tickformat = '.0%',
            showticklabels = True,
            tickfont = dict(family="Roboto", size=16, color="#2D426A")
        ),
        showlegend = False,
        title = dict(text = format_title(t,s), 
                    yanchor = 'top', y = 0.85, xanchor = 'left', x = 0.02),
        plot_bgcolor = 'rgba(0,0,0,0)',
        # height=500,
        hoverlabel = dict(bgcolor = 'white', bordercolor = '#2D426A', font = dict(size=10, family='Roboto Mono', color="#2D426A")),
        margin = dict(
            l = 80,  # default: 80
            r = 0,  # default: 80
            b = 0,  # default: 80
            t = 100, # default: 100
            pad = 0  # default: 0
            )
    )
    metros_fig.add_hline(y = 5, line_width = 2, line_color = 'black', line_dash = "dash"),
    metros_fig.update_traces(textangle=0)
    #metros_fig.update_xaxes(range=[metros_tail['diff'].min()*(1.5), metros_head['diff'].max()*(1.5)])


    return stats_out, roles_fig, seniorities_fig, metros_fig
    # return stats_out, roles_fig, metros_fig











# Run the app
if __name__ == '__main__':
    app.run(debug=True)