# base de datos
from scipy.stats import uniform
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
from sklearn.model_selection import RandomizedSearchCV

# predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve



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

def graf_boxplot(data_aviable, v1: str, v2: str):
    print(  p9.ggplot(data_aviable, p9.aes(v1,v2, color = v1)) +
        p9.theme(figure_size=(20, 5)) +
        p9.geom_boxplot() +
        p9.coord_flip())

def graf_density(data_aviable, v1: str, v2: str):
    print(  p9.ggplot(data_aviable, p9.aes(v1, fill = v2)) +
    p9.theme(figure_size=(20, 5)) +
    p9.geom_density(alpha = .3))

def graf_bar(data_aviable, v1: str):
    """descripttive"""
    print( p9.ggplot(data_aviable, p9.aes(v1, fill = v1)) +
    p9.theme(figure_size=(20, 5)) +
    p9.geom_bar(alpha =0.3))


def graf_roc(Y_test, x_test, lr):
    logit_roc_auc = roc_auc_score(Y_test, lr.predict(x_test))
    fpr, tpr, thresholds = roc_curve(Y_test, lr.predict_proba(x_test)[:,1])

    data_roc = pd.DataFrame({'logit_roc_auc':logit_roc_auc,'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds})

    print(
        p9.ggplot(data_roc, p9.aes('fpr', 'tpr')) +
        p9.theme(figure_size=(20, 5)) +
        p9.geom_line(alpha =0.3) +
        p9.geom_abline(intercept=0, slope=1,color="#8D1137") +
        p9.labs(title = "'Receiver operating characteristic'",
                x = "False Positive Rate",
                y = "True Positive Rate")
    )

def look_optimezed_param(value_x, value_y):
    # División de los datos en train y test
    # https://vitalflux.com/class-imbalance-class-weight-python-sklearn/

    # Creación del modelo
    lr=LogisticRegression()

    # Create regularization penalty space
    penalty = ['l1', 'l2']

    # Create regularization hyperparameter distribution using uniform distribution
    C = uniform(loc=0, scale=4)

    # Create hyperparameter options
    hyperparameters = dict(C=C, penalty=penalty)

    # Create randomized search 5-fold cross validation and 100 iterations
    clf = RandomizedSearchCV(lr, hyperparameters, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

    # Fit randomized search
    best_model = clf.fit(value_x, value_y)

    # View best hyperparameters
    print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
    print('Best C:', best_model.best_estimator_.get_params()['C'])

    return best_model

def graf_lr(x_value, y_value, data):
    import seaborn as sns; sns.set_theme(color_codes=True)
    # plot logistic regression curve with black points and red line
    sns.regplot(x=x_value, y=y_value, data=data, logistic=True, ci=95)

# https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/