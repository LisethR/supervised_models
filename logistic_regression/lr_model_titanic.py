# import statements
from logistic_regression.utils.tools import *
from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Get the data
titanic = data('titanic')
titanic.sample(5)

# Feature engineering (one hot encoding)
titanic = pd.get_dummies(titanic, drop_first=True)
titanic.sample(5)

# Test train split
X_train, X_test, y_train, y_test = train_test_split(titanic.drop('survived_yes', axis=1), titanic['survived_yes'])

# Train the model using the training data
LogReg = LogisticRegression(solver='lbfgs')
LogReg.fit(X_train, y_train)

# Prediciting if a class-1 child-age girl survived
LogReg.predict(np.array([[0,0,1,1]]))[0]

# predict if a class-3 adult-age male surrvived
LogReg.predict(np.array([[0,1,0,0]]))[0]

# Scoring the model
LogReg.score(X_test, y_test)

# undersatanting the score
prediction = (LogReg.predict(X_test) > .5).astype(int)
np.sum(prediction == y_test) / len(y_test)

graf_roc(y_test, X_test, LogReg)
