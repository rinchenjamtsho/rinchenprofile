#!/usr/bin/env python
# coding: utf-8

# In[1]:


from jupyter_dash import JupyterDash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import random
from sklearn.linear_model import LinearRegression


# In[2]:


#!pip install dash==2.0.0


# In[3]:


def rinchendata(years):
    df = pd.read_csv('past_data.csv')
    df.astype(str)
    
    gp=df.groupby(['Year','months']).sum()[['Saving']]
    gp.reset_index(inplace=True)
    
    gp1=df.groupby(['Year','months']).sum()[['Number of drinks']]
    gp1.reset_index(inplace=True)
    
    if(years=='2019-2022'):
        data = gp.copy()
        data1 = gp1.copy()
        
        dr=data[['Year','months','Saving']].copy()
        month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        dr.months=pd.Categorical(dr.months,categories=month,ordered=True)
        dr.reset_index(inplace=True)
        dr.sort_values(by=['months','Year'],inplace=True)
        
        dr1=data1[['Year','months','Number of drinks']].copy()
        month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        dr1.months=pd.Categorical(dr1.months,categories=month,ordered=True)
        dr1.reset_index(inplace=True)
        dr1.sort_values(by=['months','Year'],inplace=True)
        
    else:
        data = gp[gp['Year']==int(years)]
        data1 = gp1[gp1['Year']==int(years)]
        
        dr=data[['Year','months','Saving']].copy() 
        month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        dr.months=pd.Categorical(dr.months,categories=month,ordered=True)
        dr.reset_index(inplace=True)
        dr=dr.groupby('months').sum()[['Saving']]
        dr.reset_index(inplace=True)
        
        dr1=data1[['Year','months','Number of drinks']].copy() 
        month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        dr1.months=pd.Categorical(dr1.months,categories=month,ordered=True)
        dr1.reset_index(inplace=True)
        dr1=dr1.groupby('months').sum()[['Number of drinks']]
        dr1.reset_index(inplace=True)
    return [dr,dr1]


# In[4]:


def drawfigure(years):
    [dr,dr1]=rinchendata(years)
    if(years=='2019-2022'):
        fig1=px.treemap(dr1,path=['Year','months'],values='Number of drinks',title='Total number drinks by Month')
        fig1.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        fig1.update_layout(autosize = False,width = 800,height=500)
        
        fig2=px.treemap(dr,path=['Year','months'],values='Saving',title='Total Saving by Month')
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        fig2.update_layout(autosize = False,width = 800,height=500)
    else:
        fig1 = px.bar(dr1, x='months', y='Number of drinks',color='months',color_discrete_sequence=['black','blue','red','pink','green','Gray','orange','purple','brown','cyan','olive','lime'],title='Total Number of drinks over the Months')
        fig1.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        fig1.update_layout(autosize = False,width = 800,height=500)
        
        fig2 = px.bar(dr, x='months', y='Saving',color='months',color_discrete_sequence=['black','blue','red','pink','green','Gray','orange','purple','brown','cyan','olive','lime'],title='Total Saving over the Months')
        fig2.update_layout(hoverlabel=dict(bgcolor="white",font_size=16,font_family="Rockwell"))
        fig2.update_layout(autosize = False,width = 800,height=500)
    return[fig1,fig2]


# In[5]:


# Create a dash application
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.SUPERHERO])
app.title="Rinchen jamtsho"
server=app.server


# In[6]:


df = pd.read_csv('past_data.csv')
df1 = pd.read_csv('present_data.csv')


# In[7]:


card_content = [
    dbc.CardBody(
        [
            html.H5("Facebook", className="card-title"),
            html.P(
                "If you want to connect me with facebook",
                className="card-text",
            ),
            dbc.Button("Details", color="primary", href="https://www.facebook.com/"),
        ],style={'font-family': 'Comic Sans MS','fontsize':20}
    ),
]

card_content1 = [
    dbc.CardBody(
        [
            html.H5("Whatsapp", className="card-title"),
            html.P(
                "If you want to connect me with Whatsapp",
                className="card-text",
            ),
            dbc.Button("Details", color="primary", href="https://www.whatsapp.com/"),
        ],style={'font-family': 'Comic Sans MS','fontsize':20}
    ),
]

card_content2 = [
    dbc.CardBody(
        [
            html.H5("Linkin", className="card-title"),
            html.P(
                "If you want to connect me with Linkin",
                className="card-text",
            ),
            dbc.Button("Details", color="primary", href="https://www.linkedin.com/checkpoint/lg/login"),
        ],style={'font-family': 'Comic Sans MS','fontsize':20}
    ),
]


# In[8]:


#html layout
year= ['2019','2020','2021','2022','2019-2022']

app.layout = html.Div([
    html.H1('Welcome to Personal Report',style={'font-family': 'Comic Sans MS','textAlign':'center','color':'#FC0A02','fontsize':50,'padding':'50px'}),
    
    dbc.Card([
        dbc.Row([
                dbc.Col(
                    dbc.CardImg(
                        src="https://scontent.fpbh1-1.fna.fbcdn.net/v/t1.6435-9/107568793_287536319230852_5532820151805126354_n.jpg?stp=c233.0.600.600a_dst-jpg_p600x600&_nc_cat=110&ccb=1-5&_nc_sid=174925&_nc_ohc=PnMJK_ZM17QAX_RYIUg&_nc_ht=scontent.fpbh1-1.fna&oh=00_AT-r5iuRCr5oLHQRo7AeHJf7WXyM0emmxGK3MSv3s8d0xw&oe=624DF947",
                        className="img-fluid rounded-start",
                    ),className="col-md-4"),
                dbc.Col(
                    dbc.CardBody([
                            html.H4("Rinchen Jamtsho", className="card-title"),
                            html.P("This is Dashboard Shows about "
                                "my personal information that I have done "
                                "in past and present.",
                                className="card-text"),
                        ],style={'font-family': 'Comic Sans MS','fontsize':20}),className="col-md-8"),
            ],className="g-0 d-flex align-items-center",)],
    className="mb-4",style={"maxWidth": "800px"}),
    
    html.Div([
        dbc.Spinner(color="primary"),
        dbc.Spinner(color="secondary"),
        dbc.Spinner(color="success"),
        dbc.Spinner(color="warning"),
        dbc.Spinner(color="danger"),
    ]),
    
    dbc.Tabs([
            dbc.Tab(label="Past", tab_id="past"),
            dbc.Tab(label="Present", tab_id="present"),
            ], id="tabs", active_tab=" ",style={'font-family': 'Comic Sans MS','fontsize':20}
        ),
    
    html.Br(),
    
    html.Div([
        html.Div([
            html.H2('Select_Year',style={'textAlign':'left','color':'#ffffff','fontsize':40}),
            dcc.Dropdown(id='year_id',clearable=False,
                    options=[{'label':i,'value':i} for i in year ],
                   placeholder='' 
                    ),
        ],id='dd',style={'font-family': 'Comic Sans MS','fontsize':20,'width':'40%','padding':'3px','fontsize':40,'color':'#9932CC'}),
    ]),
    html.Br(),
    
    html.Div([
        # output graphic (plot2)
        dcc.Graph(id='plot1'),
        # output graphic (plot3)
        dcc.Graph(id='plot2')
    ], style = {'display': 'flex'}), 
    
    html.Br(),
    dbc.Tabs([
            dbc.Tab(label="Past data", tab_id="past_data"),
            dbc.Tab(label="Present data", tab_id="present_data"),
            ], id="tabbs", active_tab=" ",style={'font-family': 'Comic Sans MS','fontsize':20}
        ),
    html.Div(id="table",style={'font-family': 'Comic Sans MS','fontsize':20}),
    
    html.Br(), 
    html.H1('For More Details',style={'font-family': 'Comic Sans MS','textAlign':'center','color':'#ff8000','fontsize':50,'padding':'50px'}),
    
    html.Div([
        dbc.Row([
            dbc.Col(dbc.Card(card_content, color="primary", inverse=True)),
            dbc.Col(dbc.Card(card_content1, color="secondary", inverse=True)),
            dbc.Col(dbc.Card(card_content2, color="info", inverse=True)),
        ],className="mb-4"),
    ]),
])


# In[9]:


@app.callback(
    Output("table", "children"),
    Input('tabbs','active_tab'),
)

def make_table(a):
    if a is None:
        raise PreventUpdate 
        
    elif(a == 'past_data'):
        df = pd.read_csv('past_data.csv')
        
    else:
        df = pd.read_csv('present_data.csv')
        
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, color = "secondary", hover=True)


# In[10]:


@app.callback([
    Output('plot1','figure'),
    Output('plot2','figure')
],[Input('year_id','value'),
  Input('tabs','active_tab')])

def draw_graph(years,tabs): 
    if years is None or tabs is None:
        raise PreventUpdate
    elif (tabs=='past'):
        [fig1,fig2]=drawfigure(years)
    elif(tabs=='present'):
        df1 = pd.read_csv('present_data.csv')
        df1.astype(str)
        X = df1.Workinghour.values.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, df1.WakingUptime)

        x_range = np.linspace(X.min(), X.max(), 100)
        y_range = model.predict(x_range.reshape(-1, 1))

        fig1 = px.scatter(df1, x='Workinghour', y='WakingUptime', opacity=0.5)
        fig1.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
        
        X1 = df1.Sleepingtime.values.reshape(-1, 1)

        model1 = LinearRegression()
        model1.fit(X1, df1.WakingUptime)

        x_range1 = np.linspace(X1.min(), X1.max(), 100)
        y_range1 = model1.predict(x_range1.reshape(-1, 1))

        fig2 = px.scatter(df1, x='Sleepingtime', y='WakingUptime', opacity=0.5)
        fig2.add_traces(go.Scatter(x=x_range1, y=y_range1, name='Regression Fit'))
        
    else:
        [fig1,fig2]=drawfigure(years)

    return [fig1,fig2]


# In[11]:


if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)    
    url = "http://127.0.0.1:{0}".format(port)    
    app.run_server(use_reloader=False, debug=True, port=port)

