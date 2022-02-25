# base de datos
import pyodbc
import sqlalchemy as db
import numpy as np
import pandas as pd
# graficas
import matplotlib.pyplot as plt
import seaborn as sns
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

# TODO:
# 1. Arreglar el script
# 2. y verificar los pasos

# conn
server = 'LAPTOP-V50CPP72' 
database = 'wines_data'

cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                            'SERVER='+server+';DATABASE='+database+';'
                            'TRUSTED_CONNECTION=yes')

# consultar data ----
data = pd.read_sql_query("SELECT density, quality, residual_sugar, alcohol FROM winequality", cnxn) 
median_data = data.median()
data['gt_density'] = data.density > median_data.density

# plot1
(
    p9.ggplot(data, p9.aes('gt_density','alcohol', color = 'gt_density')) +
    p9.geom_boxplot()
)

(
    p9.ggplot(data, p9.aes('alcohol', fill = 'gt_density')) +
    p9.geom_density(alpha = .3)
)

# Se puede apreciar, menor densidad mayor el grado de alcohols

# plot3
(
    p9.ggplot(data, p9.aes('gt_density', fill = 'gt_density')) +
    p9.geom_bar()
)
data[['quality', 'gt_density']].groupby('gt_density').count()
# se puede observar una data balanceada

# División de los datos en train y test
X = data[['alcohol']]
y = data['gt_density']

X_train, X_test, y_train, y_test = train_test_split(
                                        X.values.reshape(-1,1),
                                        y.values.reshape(-1,1),
                                        train_size   = 0.8,
                                        random_state = 1234,
                                        shuffle      = True
                                    )

# Creación del modelo
lr=LogisticRegression()
#tuning weight for minority class then weight for majority class will be 1-weight of minority class
#Setting the range for class weights
weights = np.linspace(0.0,0.99,500)
#specifying all hyperparameters with possible values
param = {'C': [0.1, 0.5, 1,10,15,20], 'penalty': ['l1','l2'],"class_weight":[{0:x ,1:1.0 -x} for x in weights]}
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
#Gridsearch for hyperparam tuning
model= GridSearchCV(estimator= lr,param_grid=param,scoring="f1",cv=folds,return_train_score=True)
#train model to learn relationships between x and y
model.fit(X_train,y_train)

# print best hyperparameters
print("Best F1 score: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)

#Building Model again with best params
lr2=LogisticRegression(class_weight={0:0.27,1:0.73},C=20,penalty="l2")
lr2.fit(X_train,y_train)

#Building Model again with best params
lr2=LogisticRegression(class_weight={0:0.3749699398797595,1:0.6250300601202405},C=20,penalty="l2")
lr2.fit(X_train,y_train)



# predict probabilities on Test and take probability for class 1([:1])
y_pred_prob_test = lr2.predict_proba(X_test)[:, 1]
#predict labels on test dataset
y_pred_test = lr2.predict(X_test)
# create onfusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("confusion Matrix is :nn",cm)
print("n")
# ROC- AUC score
print("ROC-AUC score  test dataset:  t", roc_auc_score(y_test,y_pred_prob_test))
#Precision score
print("precision score  test dataset:  t", precision_score(y_test,y_pred_test))
#Recall Score
print("Recall score  test dataset:  t", recall_score(y_test,y_pred_test))

