

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as unPlot
```

    Populating the interactive namespace from numpy and matplotlib
    

#### El problema del Tittanic, si bien es un dataset bastante conocido, nos permite poder ver facilmente una aplicación práctica de Árboles de Clasificación.  ####
#### Lo primero que vamos a hacer es leer el dataset, para poder ver con qué tipo de datos estamos tratando. Otro problema que tengo, es transformar los atributos del tipo 'object' a string. ####



```python
unDatasetEntr = pd.read_csv("train.csv")
unDatasetTest = pd.read_csv("test.csv")
from sklearn.preprocessing import LabelEncoder
unLb = LabelEncoder()
unDatasetEntr['Embarked'] = unLb.fit_transform(unDatasetEntr['Embarked'].astype(str))
unDatasetEntr['Sex'] = unLb.fit_transform(unDatasetEntr['Sex'].astype(str))
unDatasetEntr['Cabin'] = unLb.fit_transform(unDatasetEntr['Cabin'].astype(str))
unDatasetEntr = unDatasetEntr.fillna(unDatasetEntr.median())
unLabel = unDatasetEntr[['Survived']]
entr = unDatasetEntr[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']]
test = unDatasetTest[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']]
```

#### Luego imprimo los tipos de datos, donde podemos ver que tenemos 2 tipos de datos: numéricos (int y float) y strings (pandas los ve como 'object'). ####


```python
unDatasetEntr.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex              int64
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin            int64
    Embarked         int64
    dtype: object




```python
unDatasetEntr.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>147</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>81</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>147</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>55</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>147</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



#### Creo un dataframe a partir del dataset, para no alterar los datos del dataset original, y trabajar directo con el dataframe. En mi caso me voy a quedar con las columnas más relevantes para la predicción (en éste caso los atributos: Sex, Age, y Survived). ####


```python
unDataFrame = pd.DataFrame()
unDataFrame['Sex'] = unDataset['Sex']
unDataFrame['Age'] = unDataset['Age']
unDataFrame['Survived'] = unDataset['Survived']
unDataFrame.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Age</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>22.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>26.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>male</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>female</td>
      <td>27.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>female</td>
      <td>58.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>female</td>
      <td>14.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>female</td>
      <td>55.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Como podemos ver en la fila 5 del dataset, tenemos datos faltantes. Aquí voy a eliminar los valores faltantes, para simplificar el tratamiento de los mismos. Si bien no es la mejor estrategia, en este caso nos centramos en los CART, y no en el tratamiento de datos faltantes, a modo de simplificación. ####


```python
unDataset = unDataFrame.dropna(axis=0)
pd.get_dummies(unDataset.Sex)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>female</th>
      <th>male</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>856</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>857</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>858</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>860</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>861</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>862</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>864</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>865</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>866</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>867</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>869</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>870</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>871</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>872</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>873</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>874</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>875</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>876</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>877</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>879</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>880</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>881</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>882</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>883</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>884</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>885</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>714 rows × 2 columns</p>
</div>



#### Como mi atributo de salida (o label) es 'Survived', voy a declararlo como variable dependiente, y luego la saco del dataframe. ####


```python
X = unDataset.iloc[:, 1:2].values
y = unDataset.iloc[:, 2].values
```


```python
from sklearn.tree import DecisionTreeRegressor
unRT = DecisionTreeRegressor(random_state = 0)
unRT.fit(X,y)
```




    DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_decrease=0.0,
               min_impurity_split=None, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               presort=False, random_state=0, splitter='best')




```python
from sklearn.tree import DecisionTreeClassifier
unArbolDeDecision = DecisionTreeClassifier(random_state = 10)
from sklearn.model_selection import cross_val_score
cross_val_score(unArbolDeDecision, entr, unLabel, cv=30).mean()
```




    0.796421950315165



#### Con esto obtuvimos una precisión mayor al 79,6% ####


```python

```
