import pyodbc
import sqlalchemy as db
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9

# conn
server = 'LAPTOP-V50CPP72' 
database = 'wines_data'

cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                            'SERVER='+server+';DATABASE='+database+';'
                            'TRUSTED_CONNECTION=yes')


# consultar data ----
# fixed_acidity,pH, density, citric_acid, residual_sugar, alcohol, quality
data = pd.read_sql_query("SELECT * FROM winequality", cnxn) # density, residual_sugar, quality
data['group_q'] = data['quality']>5

# grafico de dispersion entre las variables
sns.pairplot(data, corner=True,hue="group_q", palette = "hls")
plt.show()


corrmat = data.corr()

f, ax = plt.subplots(figsize =(9, 8))
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
plt.show()

data_inters = data[['fixed_acidity','density','group_q']]

# plot de la data de todas las variables
sns.countplot(x ='group_q', data=data_inters, palette = "hls")
plt.show()

# por ahora las de intesres
(
    p9.ggplot(data_inters, p9.aes('density', 'fixed_acidity', color = 'group_q')) +
    p9.geom_point()
)


from sklearn import linear_model
data_inters



model = linear_model.LogisticRegression()
model.fit(X,y)

