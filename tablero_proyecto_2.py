import pandas as pd
import numpy as np
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import psycopg2
import tensorflow as tf
import keras
import joblib


#data = pd.read_csv("https://raw.githubusercontent.com/NicolayB/archivos/refs/heads/main/proyecto%202/bank-full.csv", delimiter=";")

# Conectarse a la base de datos de AWS
engine = psycopg2.connect(
    dbname="bankdb",
    user="postgres",
    password="actdproyecto2",
    host="proyecto2.cmavnazwpcep.us-east-1.rds.amazonaws.com",
    port='5432'
)
cursor = engine.cursor()

import pandas.io.sql as sqlio
query = """
SELECT *
FROM bank;"""
data = sqlio.read_sql_query(query, engine)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Modelo de redes
model = keras.models.load_model("red_bank.keras")


# Meses
orden = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
order_months = {month: i for i, month in enumerate(orden)}
df_months = (data.groupby("month")["duration"].mean().reset_index().sort_values(by='month', key=lambda x: x.map(order_months)))

# Lista de empleos
jobs = data["job"].unique()
jobs.sort()
# Lista de estado civil
maritals = data["marital"].unique()
maritals.sort()
# Lista de educación
educations = data["education"].unique()
educations.sort()
# Lista de contact
contacts = data["contact"].unique()
contacts.sort()
# Lista de meses
months = data["month"].unique()
months.sort()
# Lista de poutcome
poutcomes = data["poutcome"].unique()
poutcomes.sort()

# Lista de columnas
cols = data.columns

# Gráficos fijos
# Duración promedio de las llamadas por mes
duration_month = px.bar(df_months, x="month", y="duration",
                        title="Duración promedio de las llamadas por mes", 
                        height=300, width=500)
duration_month.update_layout(plot_bgcolor="white")
duration_month.update_xaxes(title="")
duration_month.update_yaxes(title="Duración")


# Se crea el app layout
app.layout = html.Div([
    html.H2("Clientes del banco"),
    html.H4("Análisis de la relación entre el cliente y la adquisición de productos bancarios"),
    html.Br(),
    html.Div([
        html.H6("Estadísticas generales"),
        html.Div([
            html.Div([dcc.Graph(id="duration-per-month",
                                figure=duration_month)
            ]),
            html.Br(),
            html.Div([
                dcc.RadioItems(id="variables-acept",
                               value=cols[0],
                               options=[dict(label=v, value=v) for v in cols if v not in ["y"]],
                               style={"display": "inline-block"}),
                dcc.Graph(id="stacked-graph", style={"display": "inline-block"})
            ])
        ])
    ]),
    html.Br(),
    html.Div([
        html.Header("Seleccione el empleo que desea analizar:"),
        dcc.Dropdown(id="jobs",
                options=[{"label": job,"value": job} for job in jobs],
                value=jobs[0],
                style={"width": "50%"}),
        html.Br(),
        html.Div([
            html.H6("Histogramas"),
            html.P("Los siguientes histgramas representan el estado marital y el nivel educativo de los clientes se encuentran en el área seleccionada"),
            dcc.Graph(id="marital-status", style={"display": "inline-block"}),
            dcc.Graph(id="education-level", style={"display": "inline-block"})
        ])
    ]),
    html.Div([
        html.H6("Comparación de empleos"),
        html.Header("Seleccione los empleos y las variables a comparar"),
        html.Div([
            dcc.Dropdown(id="job1",
                        options=[dict(label=job, value=job) for job in jobs],
                        value=jobs[0],
                        style={"display": "inline-block", "width": "100%", "margin-right": "10px"}),
            dcc.Dropdown(id="variable1",
                        options=[dict(label=variable,value=variable) for variable in ["marital","education","default","balance","housing","loan","y"]],
                        value="marital"),
            html.Div([
                    html.Div([
                        html.P(id="percentage1-job1", style={"text-align":"center"})
                    ]),
                    html.Div([
                        html.P(id="percentage2-job1", style={"text-align":"center"})
                    ]),
                    html.Div([
                        html.P(id="percentage3-job1", style={"text-align":"center"})
                    ]),
                    html.Div([
                        html.P(id="percentage4-job1", style={"text-align":"center"})
                    ])
            ], style={"margin-top":"10px"}
            ),
        ], style={"display": "inline-block", "margin-right": "5%"}
        ),
        html.Div([
            dcc.Dropdown(id="job2",
                        options=[dict(label=job, value=job) for job in jobs],
                        value=jobs[1],
                        style={"display": "inline-block", "width": "100%", "margin-right": "10px"}),
            dcc.Dropdown(id="variable2",
                        options=[dict(label=variable,value=variable) for variable in ["marital","education","default","balance","housing","loan","y"]],
                        value="marital"),
            html.Div([
                    html.Div([
                        html.P(id="percentage1-job2", style={"text-align":"center"})
                    ]),
                    html.Div([
                        html.P(id="percentage2-job2", style={"text-align":"center"})
                    ]),
                    html.Div([
                        html.P(id="percentage3-job2", style={"text-align":"center"})
                    ]),
                    html.Div([
                        html.P(id="percentage4-job2", style={"text-align":"center"})
                    ])
            ], style={"margin-top":"10px"}
            ),
        ], style={"display": "inline-block", "margin-right": "5%"}
        ),
        html.Div([
            html.H6("Definición de las variables"),
            html.P("marital: estado civil del cliente"),
            html.P("education: nivel académico del cliente"),
            html.P("default: estado de mora del cliente"),
            html.P("balance: balance anual promedio del cliente"),
            html.P("housing: prestamo de vivienda del cliente"),
            html.P("loan: prestamo personal del cliente"),
            html.P("y: aceptación del producto financiero por parte del cliente")
        ], style={"display": "inline-block"}
        )   
    ]),
    html.Div([
        html.H2("Predicción"),
        html.P("Ingrese los valores para realizar la predicción"),
        html.Div([
            html.Div([
                html.P("Edad"),
                dcc.Input(id="input-age",value="0", type="number")
            ]),
            html.Div([
                html.P("Empleo"),
                dcc.Dropdown(id="dd-job",
                            options=[dict(label=job, value=job) for job in jobs],
                            value=jobs[0],
                            style={"width": "45%"})
            ]),
            html.Div([
                html.P("Estado civil"),
                dcc.Dropdown(id="dd-marital",
                            options=[dict(label=marital, value=marital) for marital in maritals],
                            value=maritals[0],
                            style={"width": "45%"}),
            ]),
            html.Div([
                html.P("Nivel educativo"),
                dcc.Dropdown(id="dd-education",
                            options=[dict(label=education, value=education) for education in educations],
                            value=educations[0],
                            style={"width": "45%"}),
            ]),
            html.Div([
                html.P("Mora"),
                dcc.Dropdown(id="dd-default",
                            options=[dict(label=i, value=i) for i in ["yes","no"]],
                            value="no",
                            style={"width": "45%"}),
            ]),
            html.Div([
                html.P("Balance promedio"),
                dcc.Input(id="input-balance",value="0", type="number")
            ]),
            html.Div([
                html.P("Prestamo de vivienda"),
                dcc.Dropdown(id="dd-housing",
                            options=[dict(label=i, value=i) for i in ["yes","no"]],
                            value="no",
                            style={"width": "45%"}),
            ]),
            html.Div([
                html.P("Prestamo personal"),
                dcc.Dropdown(id="dd-loan",
                            options=[dict(label=i, value=i) for i in ["yes","no"]],
                            value="no",
                            style={"width": "45%"}),
            ]),
            html.Div([
                html.P("Contacto"),
                dcc.Dropdown(id="dd-contact",
                            options=[dict(label=contact, value=contact) for contact in contacts],
                            value=contacts[0],
                            style={"width": "45%"}),
            ]),
            html.Div([
                html.P("Día"),
                dcc.Dropdown(id="dd-day",
                            options=[dict(label=i, value=i) for i in range(1,32)],
                            value=1,
                            style={"width": "45%"}),
            ]),
            html.Div([
                html.P("Mes"),
                dcc.Dropdown(id="dd-month",
                            options=[dict(label=month, value=month) for month in months],
                            value=months[0],
                            style={"width": "45%"}),
            ]),
            html.Div([
                html.P("Duración"),
                dcc.Input(id="input-duration",value="0", type="number")
            ]),
            html.Div([
                html.P("Contactos durante esta campaña"),
                dcc.Input(id="input-campaign",value="0", type="number")
            ]),
            html.Div([
                html.P("Días desde el últmo contacto"),
                dcc.Input(id="input-pdays",value="0", type="number")
            ]),
            html.Div([
                html.P("Contactos antes de esta campaña"),
                dcc.Input(id="input-previous",value="0", type="number")
            ]),
            html.Div([
                html.P("Resultado última campaña"),
                dcc.Dropdown(id="dd-poutcome",
                            options=[dict(label=poutcome, value=poutcome) for poutcome in poutcomes],
                            value=poutcomes[0],
                            style={"width": "45%"})
            ])
        ]
        ),
        html.Div([
            html.H5("Según los datos ingresados", style={"text-align": "center"}),
            html.H6("Probabilidad de rechazar", style={"text-align": "center"}),
            html.Div(id="prob1", style={"text-align": "center"}),
            html.H6("Probabilidad de aceptar", style={"text-align": "center"}),
            html.Div(id="prob2", style={"text-align": "center"}),
            html.H6("Decisión del cliente", style={"text-align": "center"}),
            html.Div(id="prediction", style={"text-align": "center", "font-weight": "bold"})
        ]
        )
    ])
])


@app.callback(
    Output("stacked-graph", "figure"),
    [Input("variables-acept", "value")]  
)
def aceptation_bar(variable):
    grouped_data = data.groupby([variable, "y"]).size().reset_index(name="cantidad")
    figure = px.bar(grouped_data, x=variable, y="cantidad", color="y",barmode="stack")
    figure.update_layout(height=400, width=1000, plot_bgcolor="white",
                         title="Cantidad de productos aceptados y rechazados")

    return figure

@app.callback(
    Output("marital-status", "figure"),
    [Input("jobs", "value")]
)
def marital_state(job):
    filtered_data = data[data["job"] == job]
    figure = px.histogram(filtered_data, x="marital",
                          category_orders={"marital": ["single", "married", "divorced"]},
                          title="Estado marital")
    figure.update_layout(height=400, width=500, title_x=0.5,
                        plot_bgcolor="rgba(0,0,0,0)")
    figure.update_xaxes(title="")
    figure.update_yaxes(title="Cantidad")

    return figure

@app.callback(
    Output("education-level", "figure"),
    [Input("jobs", "value")]
)
def education_level(job):
    filtered_data = data[data["job"] == job]
    figure = px.histogram(filtered_data, x="education",
                          category_orders={"education": ["primary", "secondary", "tertiary", "unknown"]},
                          title="Nivel educativo")
    figure.update_layout(height=400, width=500, title_x=0.5,
                        plot_bgcolor="rgba(0,0,0,0)")
    figure.update_xaxes(title="")
    figure.update_yaxes(title="Cantidad")

    return figure

@app.callback(
    [Output("percentage1-job1", "children"),
     Output("percentage2-job1", "children"),
     Output("percentage3-job1", "children"),
     Output("percentage4-job1", "children")
    ],
    [Input("job1", "value"),
     Input("variable1", "value")]
)
def est_job1(job, variable):
    filtered_data = data[data["job"] == job]
    if variable == "marital":
        perc_single = (filtered_data["marital"].eq("single").sum() / filtered_data["marital"].count()) * 100
        perc_married = (filtered_data["marital"].eq("married").sum() / filtered_data["marital"].count()) * 100
        perc_divorced = (filtered_data["marital"].eq("divorced").sum() / filtered_data["marital"].count()) * 100

        return(
            f"Porcentaje solteros: {perc_single:.2f}%",
            f"Porcentaje casados: {perc_married:.2f}%",
            f"Porcentaje divorciados: {perc_divorced:.2f}%",
            ""
            )
    elif variable == "education":
        perc_primary = (filtered_data["education"].eq("primary").sum() / filtered_data["education"].count()) * 100
        perc_secondary = (filtered_data["education"].eq("secondary").sum() / filtered_data["education"].count()) * 100
        perc_terciary = (filtered_data["education"].eq("tertiary").sum() / filtered_data["education"].count()) * 100
        perc_unknown = (filtered_data["education"].eq("unknown").sum() / filtered_data["education"].count()) * 100

        return(
            f"Porcentaje primaria: {perc_primary:.2f}%",
            f"Porcentaje secundaria: {perc_secondary:.2f}%",
            f"Porcentaje terciaria: {perc_terciary:.2f}%",
            f"Porcentaje desconocida: {perc_unknown:.2f}%"
            )
    elif variable == "default":
        perc_yes = (filtered_data[variable].eq("yes").sum() / filtered_data[variable].count()) * 100
        perc_no = (filtered_data[variable].eq("no").sum() / filtered_data[variable].count()) * 100

        return(
            f"Porcentaje con mora: {perc_yes:.2f}%",
            f"Porcentaje sin mora: {perc_no:.2f}%",
            "",
            ""
            )
    elif variable == "balance":
        avg_balance = filtered_data[variable].mean()

        return(
            f"Promedio de balance anual: {avg_balance:.2f} euros",
            "",
            "",
            ""
            )
    elif variable == "housing":
        perc_yes = (filtered_data[variable].eq("yes").sum() / filtered_data[variable].count()) * 100
        perc_no = (filtered_data[variable].eq("no").sum() / filtered_data[variable].count()) * 100

        return(
            f"Porcentaje con prestamo de vivienda: {perc_yes:.2f}%",
            f"Porcentaje sin prestamo de vivienda: {perc_no:.2f}%",
            "",
            ""
            )
    elif variable == "loan":
        perc_yes = (filtered_data[variable].eq("yes").sum() / filtered_data[variable].count()) * 100
        perc_no = (filtered_data[variable].eq("no").sum() / filtered_data[variable].count()) * 100

        return(
            f"Porcentaje con prestamo personal: {perc_yes:.2f}%",
            f"Porcentaje sin prestamo personal: {perc_no:.2f}%",
            "",
            ""
            )
    elif variable == "y":
        perc_yes = (filtered_data[variable].eq("yes").sum() / filtered_data[variable].count()) * 100
        perc_no = (filtered_data[variable].eq("no").sum() / filtered_data[variable].count()) * 100

        return(
            f"Porcentaje que aceptó un producto: {perc_yes:.2f}%",
            f"Porcentaje que no aceptó un producto: {perc_no:.2f}%",
            "",
            ""
            )
    
@app.callback(
    [Output("percentage1-job2", "children"),
     Output("percentage2-job2", "children"),
     Output("percentage3-job2", "children"),
     Output("percentage4-job2", "children")
    ],
    [Input("job2", "value"),
     Input("variable2", "value")]
)
def est_job2(job, variable):
    filtered_data = data[data["job"] == job]
    if variable == "marital":
        perc_single = (filtered_data["marital"].eq("single").sum() / filtered_data["marital"].count()) * 100
        perc_married = (filtered_data["marital"].eq("married").sum() / filtered_data["marital"].count()) * 100
        perc_divorced = (filtered_data["marital"].eq("divorced").sum() / filtered_data["marital"].count()) * 100

        return(
            f"Porcentaje solteros: {perc_single:.2f}%",
            f"Porcentaje casados: {perc_married:.2f}%",
            f"Porcentaje divorciados: {perc_divorced:.2f}%",
            ""
            )
    elif variable == "education":
        perc_primary = (filtered_data["education"].eq("primary").sum() / filtered_data["education"].count()) * 100
        perc_secondary = (filtered_data["education"].eq("secondary").sum() / filtered_data["education"].count()) * 100
        perc_terciary = (filtered_data["education"].eq("tertiary").sum() / filtered_data["education"].count()) * 100
        perc_unknown = (filtered_data["education"].eq("unknown").sum() / filtered_data["education"].count()) * 100

        return(
            f"Porcentaje primaria: {perc_primary:.2f}%",
            f"Porcentaje secundaria: {perc_secondary:.2f}%",
            f"Porcentaje terciaria: {perc_terciary:.2f}%",
            f"Porcentaje desconocida: {perc_unknown:.2f}%"
            )
    elif variable == "default":
        perc_yes = (filtered_data[variable].eq("yes").sum() / filtered_data[variable].count()) * 100
        perc_no = (filtered_data[variable].eq("no").sum() / filtered_data[variable].count()) * 100

        return(
            f"Porcentaje con mora: {perc_yes:.2f}%",
            f"Porcentaje sin mora: {perc_no:.2f}%",
            "",
            ""
            )
    elif variable == "balance":
        avg_balance = filtered_data[variable].mean()

        return(
            f"Promedio de balance anual: {avg_balance:.2f} euros",
            "",
            "",
            ""
            )
    elif variable == "housing":
        perc_yes = (filtered_data[variable].eq("yes").sum() / filtered_data[variable].count()) * 100
        perc_no = (filtered_data[variable].eq("no").sum() / filtered_data[variable].count()) * 100

        return(
            f"Porcentaje con prestamo de vivienda: {perc_yes:.2f}%",
            f"Porcentaje sin prestamo de vivienda: {perc_no:.2f}%",
            "",
            ""
            )
    elif variable == "loan":
        perc_yes = (filtered_data[variable].eq("yes").sum() / filtered_data[variable].count()) * 100
        perc_no = (filtered_data[variable].eq("no").sum() / filtered_data[variable].count()) * 100

        return(
            f"Porcentaje con prestamo personal: {perc_yes:.2f}%",
            f"Porcentaje sin prestamo personal: {perc_no:.2f}%",
            "",
            ""
            )
    elif variable == "y":
        perc_yes = (filtered_data[variable].eq("yes").sum() / filtered_data[variable].count()) * 100
        perc_no = (filtered_data[variable].eq("no").sum() / filtered_data[variable].count()) * 100

        return(
            f"Porcentaje que aceptó un producto: {perc_yes:.2f}%",
            f"Porcentaje que no aceptó un producto: {perc_no:.2f}%",
            "",
            ""
            )

@app.callback(
    [Output("prob1", "children"),
     Output("prob2", "children"),
     Output("prediction", "children")],
    [Input("input-age", "value"),
     Input("dd-job", "value"),
     Input("dd-marital", "value"),
     Input("dd-education", "value"),
     Input("dd-default", "value"),
     Input("input-balance", "value"),
     Input("dd-housing", "value"),
     Input("dd-loan", "value"),
     Input("dd-contact", "value"),
     Input("dd-day", "value"),
     Input("dd-month", "value"),
     Input("input-duration", "value"),
     Input("input-campaign", "value"),
     Input("input-pdays", "value"),
     Input("input-previous", "value"),
     Input("dd-poutcome", "value")]
)
def prediction(age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome):
    # Documentos de escalamiento y codificación
    import os
    import joblib
    path = os.getcwd()
    #std_scl = joblib.load(path+"/scaler.pkl")
    #codif = joblib.load(path+"/encoder.pkl")
    std_scl = joblib.load("scaler.pkl")
    codif = joblib.load("encoder.pkl")
    # Función para deserializar el modelo
    def predict(lista):
        Xnew = pd.DataFrame([{
            "age": lista[0],
            "job": lista[1],
            "marital": lista[2],
            "education": lista[3],
            "default": lista[4],
            "balance": lista[5],
            "housing": lista[6],
            "loan": lista[7],
            "contact": lista[8],
            "day": lista[9],
            "month": lista[10],
            "duration": lista[11],
            "campaign": lista[12],
            "pdays": lista[13],
            "previous": lista[14],
            "poutcome": lista[15]
        }])

        cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
        Xnew_codif = codif.transform(Xnew[cat_cols])
        
        Xnew_num = Xnew.drop(cat_cols, axis=1)

        new_data = np.hstack((Xnew_num, Xnew_codif))

        # Escalar los datos
        new_data_escalada = std_scl.transform(new_data)

        # Predicción
        ypred = model.predict(new_data_escalada)

        opciones_pred = np.argmax(ypred, axis=1)
        opciones_pred = "yes" if opciones_pred == 1 else "no"

        return ypred, opciones_pred

    lista = [age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome]
    pred = predict(lista)
    prob1 = pred[0][0]
    prob2 = 1 - prob1
    acep = pred[1]

    return(prob1, prob2, acep)

if __name__ == '__main__':
    app.run_server(debug=True)