from utils.tools import *

data = connection_db_sql('wines_data', "SELECT density, quality, residual_sugar, alcohol FROM winequality")
data['gt_density'] = data.density > data.density.median()

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
weights = np.linspace(0.0,0.99,100)
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

the_best_params = model.best_params_
h_c = list(the_best_params.values())[0]
h_class = list(the_best_params.values())[1]
h_penalty = list(the_best_params.values())[2]

#Building Model again with best params
lr2=LogisticRegression(class_weight={0:0.27,1:0.73},C=20,penalty="l2")
lr2.fit(X_train,y_train)

#Building Model again with best params
lr2=LogisticRegression(class_weight={0:0.39877755511022045,1:0.6012224448897796},C=0.1,penalty="l2")
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

