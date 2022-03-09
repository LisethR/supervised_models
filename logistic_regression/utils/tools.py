# base de datos
import pyodbc
import sqlalchemy as db
import numpy as np
import pandas as pd
# graficas
import matplotlib.pyplot as plt
import plotnine as p9
# modelaje
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

# predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# data consulting in sql server
def connection_db_sql(database: str, consult_sql_server: str):
    '''
    Permite consultar la informacion que se tiene sobre la base de datos
    SQL Server, cual depende de la siguiente informacion:
    Vars:
    ---- 
    las variables de entra son:
    - database: es la base de datos particular, la cual guarda las 
        tablas de interes en formato str.
    - consult_sql_server: los caracateristicas y formato 
        de la consulta en SQL Server.
    result:
    ------
    se obtiene un df con la informacion de la consulta con las caracteristicas de interes
    nota:
    ---- 
    la consulta es exactamente igual a la consulta en SQL Server
    '''

    # connection
    server = 'LAPTOP-V50CPP72' 

    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                            'SERVER='+server+';DATABASE='+database+';'
                            'TRUSTED_CONNECTION=yes')

    # consultar data ----
    data_from_sqlserver = pd.read_sql_query(consult_sql_server, cnxn)

    # return dataframe with consulting
    return data_from_sqlserver


# server = 'LAPTOP-V50CPP72' 
# database = 'wines_data'

# cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
#                             'SERVER='+server+';DATABASE='+database+';'
#                             'TRUSTED_CONNECTION=yes')

# # consultar data ----
# data = pd.read_sql_query("SELECT density, quality, residual_sugar, alcohol FROM winequality", cnxn) 
# median_data = data.median()
# data['gt_density'] = data.density > median_data.density