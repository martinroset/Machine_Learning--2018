
#### Éste dataset reune datos del análisis químico de vinos porducidos en Italia (todos en una misma región) a partir de tres cosechas diferentes. Si bien el dataset original tiene 30 atributos, el de UCI (https://archive.ics.uci.edu/ml/datasets/wine) fué reducido a los 13 atributos. Éste problema trata de a partir de un dataset de vinos, poder predecir de qué cosecha es el vino.


```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score
%matplotlib inline
```

#### Cargamos el dataset, y observamos los primeros 15 registros del mismo.


```python
wineDS = pd.read_csv("wine_con_nombres.csv")
wineDS.head(15)
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
      <th>Class</th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280OD315</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>14.20</td>
      <td>1.76</td>
      <td>2.45</td>
      <td>15.2</td>
      <td>112</td>
      <td>3.27</td>
      <td>3.39</td>
      <td>0.34</td>
      <td>1.97</td>
      <td>6.75</td>
      <td>1.05</td>
      <td>2.85</td>
      <td>1450</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>14.39</td>
      <td>1.87</td>
      <td>2.45</td>
      <td>14.6</td>
      <td>96</td>
      <td>2.50</td>
      <td>2.52</td>
      <td>0.30</td>
      <td>1.98</td>
      <td>5.25</td>
      <td>1.02</td>
      <td>3.58</td>
      <td>1290</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>14.06</td>
      <td>2.15</td>
      <td>2.61</td>
      <td>17.6</td>
      <td>121</td>
      <td>2.60</td>
      <td>2.51</td>
      <td>0.31</td>
      <td>1.25</td>
      <td>5.05</td>
      <td>1.06</td>
      <td>3.58</td>
      <td>1295</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>14.83</td>
      <td>1.64</td>
      <td>2.17</td>
      <td>14.0</td>
      <td>97</td>
      <td>2.80</td>
      <td>2.98</td>
      <td>0.29</td>
      <td>1.98</td>
      <td>5.20</td>
      <td>1.08</td>
      <td>2.85</td>
      <td>1045</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>13.86</td>
      <td>1.35</td>
      <td>2.27</td>
      <td>16.0</td>
      <td>98</td>
      <td>2.98</td>
      <td>3.15</td>
      <td>0.22</td>
      <td>1.85</td>
      <td>7.22</td>
      <td>1.01</td>
      <td>3.55</td>
      <td>1045</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>14.10</td>
      <td>2.16</td>
      <td>2.30</td>
      <td>18.0</td>
      <td>105</td>
      <td>2.95</td>
      <td>3.32</td>
      <td>0.22</td>
      <td>2.38</td>
      <td>5.75</td>
      <td>1.25</td>
      <td>3.17</td>
      <td>1510</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>14.12</td>
      <td>1.48</td>
      <td>2.32</td>
      <td>16.8</td>
      <td>95</td>
      <td>2.20</td>
      <td>2.43</td>
      <td>0.26</td>
      <td>1.57</td>
      <td>5.00</td>
      <td>1.17</td>
      <td>2.82</td>
      <td>1280</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>13.75</td>
      <td>1.73</td>
      <td>2.41</td>
      <td>16.0</td>
      <td>89</td>
      <td>2.60</td>
      <td>2.76</td>
      <td>0.29</td>
      <td>1.81</td>
      <td>5.60</td>
      <td>1.15</td>
      <td>2.90</td>
      <td>1320</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>14.75</td>
      <td>1.73</td>
      <td>2.39</td>
      <td>11.4</td>
      <td>91</td>
      <td>3.10</td>
      <td>3.69</td>
      <td>0.43</td>
      <td>2.81</td>
      <td>5.40</td>
      <td>1.25</td>
      <td>2.73</td>
      <td>1150</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>14.38</td>
      <td>1.87</td>
      <td>2.38</td>
      <td>12.0</td>
      <td>102</td>
      <td>3.30</td>
      <td>3.64</td>
      <td>0.29</td>
      <td>2.96</td>
      <td>7.50</td>
      <td>1.20</td>
      <td>3.00</td>
      <td>1547</td>
    </tr>
  </tbody>
</table>
</div>



#### Aquí podemos ver que el dataset no tiene atributos faltantes, entre otras cosas (media, mín, máx, etc.). 


```python
wineDS.describe()
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
      <th>Class</th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280OD315</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
      <td>178.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.938202</td>
      <td>13.000618</td>
      <td>2.336348</td>
      <td>2.366517</td>
      <td>19.494944</td>
      <td>99.741573</td>
      <td>2.295112</td>
      <td>2.029270</td>
      <td>0.361854</td>
      <td>1.590899</td>
      <td>5.058090</td>
      <td>0.957449</td>
      <td>2.611685</td>
      <td>746.893258</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.775035</td>
      <td>0.811827</td>
      <td>1.117146</td>
      <td>0.274344</td>
      <td>3.339564</td>
      <td>14.282484</td>
      <td>0.625851</td>
      <td>0.998859</td>
      <td>0.124453</td>
      <td>0.572359</td>
      <td>2.318286</td>
      <td>0.228572</td>
      <td>0.709990</td>
      <td>314.907474</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>11.030000</td>
      <td>0.740000</td>
      <td>1.360000</td>
      <td>10.600000</td>
      <td>70.000000</td>
      <td>0.980000</td>
      <td>0.340000</td>
      <td>0.130000</td>
      <td>0.410000</td>
      <td>1.280000</td>
      <td>0.480000</td>
      <td>1.270000</td>
      <td>278.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>12.362500</td>
      <td>1.602500</td>
      <td>2.210000</td>
      <td>17.200000</td>
      <td>88.000000</td>
      <td>1.742500</td>
      <td>1.205000</td>
      <td>0.270000</td>
      <td>1.250000</td>
      <td>3.220000</td>
      <td>0.782500</td>
      <td>1.937500</td>
      <td>500.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>13.050000</td>
      <td>1.865000</td>
      <td>2.360000</td>
      <td>19.500000</td>
      <td>98.000000</td>
      <td>2.355000</td>
      <td>2.135000</td>
      <td>0.340000</td>
      <td>1.555000</td>
      <td>4.690000</td>
      <td>0.965000</td>
      <td>2.780000</td>
      <td>673.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>13.677500</td>
      <td>3.082500</td>
      <td>2.557500</td>
      <td>21.500000</td>
      <td>107.000000</td>
      <td>2.800000</td>
      <td>2.875000</td>
      <td>0.437500</td>
      <td>1.950000</td>
      <td>6.200000</td>
      <td>1.120000</td>
      <td>3.170000</td>
      <td>985.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000000</td>
      <td>14.830000</td>
      <td>5.800000</td>
      <td>3.230000</td>
      <td>30.000000</td>
      <td>162.000000</td>
      <td>3.880000</td>
      <td>5.080000</td>
      <td>0.660000</td>
      <td>3.580000</td>
      <td>13.000000</td>
      <td>1.710000</td>
      <td>4.000000</td>
      <td>1680.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Aquí podemos ver que ninguno de los atributos tiene valores faltantes, y que todos son del tipo numérico.


```python
wineDS.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 14 columns):
    Class                   178 non-null int64
    Alcohol                 178 non-null float64
    Malic acid              178 non-null float64
    Ash                     178 non-null float64
    Alcalinity of ash       178 non-null float64
    Magnesium               178 non-null int64
    Total phenols           178 non-null float64
    Flavanoids              178 non-null float64
    Nonflavanoid phenols    178 non-null float64
    Proanthocyanins         178 non-null float64
    Color intensity         178 non-null float64
    Hue                     178 non-null float64
    OD280OD315              178 non-null float64
    Proline                 178 non-null int64
    dtypes: float64(11), int64(3)
    memory usage: 19.5 KB
    

#### En la siguiente matriz de correlación, podemos ver que tan correlacionados están unos atributos con otros. Los 2 atributos más coorelacionados son "Flavanoids" y "Total Phenols". 


```python
import numpy as np
wineDF = pd.DataFrame(wineDS)
rs = np.random.RandomState(0)
unaCorrM = wineDF.corr()
unaCorrM.style.background_gradient()
```




<style  type="text/css" >
    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col0 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col1 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col2 {
            background-color:  #308cbe;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col3 {
            background-color:  #fcf4fa;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col4 {
            background-color:  #2685bb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col5 {
            background-color:  #faf2f8;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col6 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col7 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col8 {
            background-color:  #2685bb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col9 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col10 {
            background-color:  #6ba5cd;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col11 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col12 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col13 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col0 {
            background-color:  #c6cce3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col1 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col2 {
            background-color:  #94b6d7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col3 {
            background-color:  #cacee5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col4 {
            background-color:  #f1ebf5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col5 {
            background-color:  #94b6d7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col6 {
            background-color:  #4897c4;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col7 {
            background-color:  #4897c4;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col8 {
            background-color:  #d1d2e6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col9 {
            background-color:  #93b5d6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col10 {
            background-color:  #187cb6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col11 {
            background-color:  #b3c3de;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col12 {
            background-color:  #7bacd1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col13 {
            background-color:  #056aa6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col0 {
            background-color:  #197db7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col1 {
            background-color:  #b9c6e0;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col2 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col3 {
            background-color:  #d7d6e9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col4 {
            background-color:  #71a8ce;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col5 {
            background-color:  #e4e1ef;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col6 {
            background-color:  #d6d6e9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col7 {
            background-color:  #d3d4e7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col8 {
            background-color:  #60a1ca;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col9 {
            background-color:  #dfddec;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col10 {
            background-color:  #71a8ce;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col11 {
            background-color:  #faf3f9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col12 {
            background-color:  #d3d4e7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col13 {
            background-color:  #c9cee4;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col0 {
            background-color:  #8fb4d6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col1 {
            background-color:  #99b8d8;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col2 {
            background-color:  #83afd3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col3 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col4 {
            background-color:  #3b92c1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col5 {
            background-color:  #8fb4d6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col6 {
            background-color:  #76aad0;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col7 {
            background-color:  #69a5cc;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col8 {
            background-color:  #80aed2;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col9 {
            background-color:  #b3c3de;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col10 {
            background-color:  #6da6cd;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col11 {
            background-color:  #b4c4df;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col12 {
            background-color:  #8bb2d4;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col13 {
            background-color:  #67a4cc;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col0 {
            background-color:  #0872b1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col1 {
            background-color:  #fdf5fa;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col2 {
            background-color:  #5ea0ca;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col3 {
            background-color:  #7bacd1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col4 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col5 {
            background-color:  #e9e5f1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col6 {
            background-color:  #d4d4e8;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col7 {
            background-color:  #cacee5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col8 {
            background-color:  #4a98c5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col9 {
            background-color:  #dbdaeb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col10 {
            background-color:  #adc1dd;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col11 {
            background-color:  #d9d8ea;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col12 {
            background-color:  #c4cbe3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col13 {
            background-color:  #ede8f3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col0 {
            background-color:  #b0c2de;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col1 {
            background-color:  #88b1d4;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col2 {
            background-color:  #b7c5df;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col3 {
            background-color:  #b3c3de;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col4 {
            background-color:  #d1d2e6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col5 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col6 {
            background-color:  #5ea0ca;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col7 {
            background-color:  #549cc7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col8 {
            background-color:  #e0dded;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col9 {
            background-color:  #78abd0;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col10 {
            background-color:  #7eadd1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col11 {
            background-color:  #96b6d7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col12 {
            background-color:  #7dacd1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col13 {
            background-color:  #358fc0;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col0 {
            background-color:  #f5eef6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col1 {
            background-color:  #83afd3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col2 {
            background-color:  #e7e3f0;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col3 {
            background-color:  #dedcec;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col4 {
            background-color:  #f2ecf5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col5 {
            background-color:  #a7bddb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col6 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col7 {
            background-color:  #034b76;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col8 {
            background-color:  #f7f0f7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col9 {
            background-color:  #0872b1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col10 {
            background-color:  #bdc8e1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col11 {
            background-color:  #2c89bd;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col12 {
            background-color:  #046198;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col13 {
            background-color:  #1b7eb7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col0 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col1 {
            background-color:  #93b5d6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col2 {
            background-color:  #f1ebf4;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col3 {
            background-color:  #e0deed;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col4 {
            background-color:  #f6eff7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col5 {
            background-color:  #abbfdc;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col6 {
            background-color:  #034d79;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col7 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col8 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col9 {
            background-color:  #056dab;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col10 {
            background-color:  #d5d5e8;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col11 {
            background-color:  #1278b4;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col12 {
            background-color:  #04588a;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col13 {
            background-color:  #1c7fb8;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col0 {
            background-color:  #0f76b3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col1 {
            background-color:  #ebe6f2;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col2 {
            background-color:  #5c9fc9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col3 {
            background-color:  #d2d2e7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col4 {
            background-color:  #589ec8;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col5 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col6 {
            background-color:  #e5e1ef;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col7 {
            background-color:  #e3e0ee;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col8 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col9 {
            background-color:  #f2ecf5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col10 {
            background-color:  #8eb3d5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col11 {
            background-color:  #d7d6e9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col12 {
            background-color:  #e5e1ef;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col13 {
            background-color:  #dcdaeb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col0 {
            background-color:  #dedcec;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col1 {
            background-color:  #afc1dd;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col2 {
            background-color:  #d8d7e9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col3 {
            background-color:  #f3edf5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col4 {
            background-color:  #e2dfee;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col5 {
            background-color:  #9fbad9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col6 {
            background-color:  #056ba9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col7 {
            background-color:  #05659f;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col8 {
            background-color:  #eee9f3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col9 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col10 {
            background-color:  #b7c5df;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col11 {
            background-color:  #549cc7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col12 {
            background-color:  #0c74b2;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col13 {
            background-color:  #4697c4;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col0 {
            background-color:  #4094c3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col1 {
            background-color:  #2987bc;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col2 {
            background-color:  #6ba5cd;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col3 {
            background-color:  #bcc7e1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col4 {
            background-color:  #b9c6e0;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col5 {
            background-color:  #abbfdc;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col6 {
            background-color:  #a2bcda;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col7 {
            background-color:  #a9bfdc;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col8 {
            background-color:  #8cb3d5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col9 {
            background-color:  #bbc7e0;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col10 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col11 {
            background-color:  #f6eff7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col12 {
            background-color:  #dbdaeb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col13 {
            background-color:  #4c99c5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col0 {
            background-color:  #ede7f2;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col1 {
            background-color:  #dddbec;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col2 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col3 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col4 {
            background-color:  #eee8f3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col5 {
            background-color:  #d1d2e6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col6 {
            background-color:  #2484ba;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col7 {
            background-color:  #056faf;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col8 {
            background-color:  #e0deed;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col9 {
            background-color:  #65a3cb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col10 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col11 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col12 {
            background-color:  #056fae;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col13 {
            background-color:  #63a2cb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col0 {
            background-color:  #faf3f9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col1 {
            background-color:  #bfc9e1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col2 {
            background-color:  #ede7f2;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col3 {
            background-color:  #f4eef6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col4 {
            background-color:  #eee8f3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col5 {
            background-color:  #ced0e6;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col6 {
            background-color:  #04629a;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col7 {
            background-color:  #045788;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col8 {
            background-color:  #fcf4fa;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col9 {
            background-color:  #2182b9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col10 {
            background-color:  #f6eff7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col11 {
            background-color:  #0c74b2;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col12 {
            background-color:  #023858;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col13 {
            background-color:  #4c99c5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col0 {
            background-color:  #eee8f3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col1 {
            background-color:  #0c74b2;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col2 {
            background-color:  #d3d4e7;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col3 {
            background-color:  #c6cce3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col4 {
            background-color:  #fff7fb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col5 {
            background-color:  #6ba5cd;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col6 {
            background-color:  #157ab5;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col7 {
            background-color:  #0f76b3;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col8 {
            background-color:  #e7e3f0;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col9 {
            background-color:  #5a9ec9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col10 {
            background-color:  #5c9fc9;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col11 {
            background-color:  #65a3cb;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col12 {
            background-color:  #3b92c1;
        }    #T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col13 {
            background-color:  #023858;
        }</style>  
<table id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >Class</th> 
        <th class="col_heading level0 col1" >Alcohol</th> 
        <th class="col_heading level0 col2" >Malic acid</th> 
        <th class="col_heading level0 col3" >Ash</th> 
        <th class="col_heading level0 col4" >Alcalinity of ash</th> 
        <th class="col_heading level0 col5" >Magnesium</th> 
        <th class="col_heading level0 col6" >Total phenols</th> 
        <th class="col_heading level0 col7" >Flavanoids</th> 
        <th class="col_heading level0 col8" >Nonflavanoid phenols</th> 
        <th class="col_heading level0 col9" >Proanthocyanins</th> 
        <th class="col_heading level0 col10" >Color intensity</th> 
        <th class="col_heading level0 col11" >Hue</th> 
        <th class="col_heading level0 col12" >OD280OD315</th> 
        <th class="col_heading level0 col13" >Proline</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row0" class="row_heading level0 row0" >Class</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col0" class="data row0 col0" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col1" class="data row0 col1" >-0.328222</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col2" class="data row0 col2" >0.437776</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col3" class="data row0 col3" >-0.0496432</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col4" class="data row0 col4" >0.517859</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col5" class="data row0 col5" >-0.209179</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col6" class="data row0 col6" >-0.719163</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col7" class="data row0 col7" >-0.847498</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col8" class="data row0 col8" >0.489109</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col9" class="data row0 col9" >-0.49913</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col10" class="data row0 col10" >0.265668</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col11" class="data row0 col11" >-0.617369</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col12" class="data row0 col12" >-0.78823</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row0_col13" class="data row0 col13" >-0.633717</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row1" class="row_heading level0 row1" >Alcohol</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col0" class="data row1 col0" >-0.328222</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col1" class="data row1 col1" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col2" class="data row1 col2" >0.0943969</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col3" class="data row1 col3" >0.211545</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col4" class="data row1 col4" >-0.310235</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col5" class="data row1 col5" >0.270798</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col6" class="data row1 col6" >0.289101</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col7" class="data row1 col7" >0.236815</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col8" class="data row1 col8" >-0.155929</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col9" class="data row1 col9" >0.136698</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col10" class="data row1 col10" >0.546364</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col11" class="data row1 col11" >-0.0717472</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col12" class="data row1 col12" >0.0723432</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row1_col13" class="data row1 col13" >0.64372</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row2" class="row_heading level0 row2" >Malic acid</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col0" class="data row2 col0" >0.437776</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col1" class="data row2 col1" >0.0943969</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col2" class="data row2 col2" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col3" class="data row2 col3" >0.164045</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col4" class="data row2 col4" >0.2885</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col5" class="data row2 col5" >-0.0545751</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col6" class="data row2 col6" >-0.335167</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col7" class="data row2 col7" >-0.411007</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col8" class="data row2 col8" >0.292977</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col9" class="data row2 col9" >-0.220746</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col10" class="data row2 col10" >0.248985</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col11" class="data row2 col11" >-0.561296</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col12" class="data row2 col12" >-0.36871</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row2_col13" class="data row2 col13" >-0.192011</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row3" class="row_heading level0 row3" >Ash</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col0" class="data row3 col0" >-0.0496432</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col1" class="data row3 col1" >0.211545</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col2" class="data row3 col2" >0.164045</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col3" class="data row3 col3" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col4" class="data row3 col4" >0.443367</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col5" class="data row3 col5" >0.286587</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col6" class="data row3 col6" >0.12898</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col7" class="data row3 col7" >0.115077</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col8" class="data row3 col8" >0.18623</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col9" class="data row3 col9" >0.00965194</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col10" class="data row3 col10" >0.258887</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col11" class="data row3 col11" >-0.0746669</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col12" class="data row3 col12" >0.00391123</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row3_col13" class="data row3 col13" >0.223626</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row4" class="row_heading level0 row4" >Alcalinity of ash</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col0" class="data row4 col0" >0.517859</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col1" class="data row4 col1" >-0.310235</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col2" class="data row4 col2" >0.2885</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col3" class="data row4 col3" >0.443367</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col4" class="data row4 col4" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col5" class="data row4 col5" >-0.0833331</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col6" class="data row4 col6" >-0.321113</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col7" class="data row4 col7" >-0.35137</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col8" class="data row4 col8" >0.361922</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col9" class="data row4 col9" >-0.197327</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col10" class="data row4 col10" >0.018732</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col11" class="data row4 col11" >-0.273955</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col12" class="data row4 col12" >-0.276769</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row4_col13" class="data row4 col13" >-0.440597</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row5" class="row_heading level0 row5" >Magnesium</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col0" class="data row5 col0" >-0.209179</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col1" class="data row5 col1" >0.270798</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col2" class="data row5 col2" >-0.0545751</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col3" class="data row5 col3" >0.286587</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col4" class="data row5 col4" >-0.0833331</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col5" class="data row5 col5" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col6" class="data row5 col6" >0.214401</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col7" class="data row5 col7" >0.195784</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col8" class="data row5 col8" >-0.256294</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col9" class="data row5 col9" >0.236441</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col10" class="data row5 col10" >0.19995</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col11" class="data row5 col11" >0.0553982</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col12" class="data row5 col12" >0.0660039</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row5_col13" class="data row5 col13" >0.393351</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row6" class="row_heading level0 row6" >Total phenols</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col0" class="data row6 col0" >-0.719163</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col1" class="data row6 col1" >0.289101</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col2" class="data row6 col2" >-0.335167</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col3" class="data row6 col3" >0.12898</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col4" class="data row6 col4" >-0.321113</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col5" class="data row6 col5" >0.214401</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col6" class="data row6 col6" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col7" class="data row6 col7" >0.864564</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col8" class="data row6 col8" >-0.449935</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col9" class="data row6 col9" >0.612413</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col10" class="data row6 col10" >-0.0551364</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col11" class="data row6 col11" >0.433681</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col12" class="data row6 col12" >0.699949</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row6_col13" class="data row6 col13" >0.498115</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row7" class="row_heading level0 row7" >Flavanoids</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col0" class="data row7 col0" >-0.847498</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col1" class="data row7 col1" >0.236815</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col2" class="data row7 col2" >-0.411007</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col3" class="data row7 col3" >0.115077</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col4" class="data row7 col4" >-0.35137</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col5" class="data row7 col5" >0.195784</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col6" class="data row7 col6" >0.864564</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col7" class="data row7 col7" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col8" class="data row7 col8" >-0.5379</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col9" class="data row7 col9" >0.652692</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col10" class="data row7 col10" >-0.172379</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col11" class="data row7 col11" >0.543479</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col12" class="data row7 col12" >0.787194</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row7_col13" class="data row7 col13" >0.494193</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row8" class="row_heading level0 row8" >Nonflavanoid phenols</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col0" class="data row8 col0" >0.489109</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col1" class="data row8 col1" >-0.155929</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col2" class="data row8 col2" >0.292977</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col3" class="data row8 col3" >0.18623</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col4" class="data row8 col4" >0.361922</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col5" class="data row8 col5" >-0.256294</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col6" class="data row8 col6" >-0.449935</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col7" class="data row8 col7" >-0.5379</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col8" class="data row8 col8" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col9" class="data row8 col9" >-0.365845</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col10" class="data row8 col10" >0.139057</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col11" class="data row8 col11" >-0.26264</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col12" class="data row8 col12" >-0.50327</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row8_col13" class="data row8 col13" >-0.311385</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row9" class="row_heading level0 row9" >Proanthocyanins</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col0" class="data row9 col0" >-0.49913</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col1" class="data row9 col1" >0.136698</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col2" class="data row9 col2" >-0.220746</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col3" class="data row9 col3" >0.00965194</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col4" class="data row9 col4" >-0.197327</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col5" class="data row9 col5" >0.236441</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col6" class="data row9 col6" >0.612413</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col7" class="data row9 col7" >0.652692</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col8" class="data row9 col8" >-0.365845</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col9" class="data row9 col9" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col10" class="data row9 col10" >-0.0252499</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col11" class="data row9 col11" >0.295544</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col12" class="data row9 col12" >0.519067</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row9_col13" class="data row9 col13" >0.330417</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row10" class="row_heading level0 row10" >Color intensity</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col0" class="data row10 col0" >0.265668</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col1" class="data row10 col1" >0.546364</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col2" class="data row10 col2" >0.248985</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col3" class="data row10 col3" >0.258887</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col4" class="data row10 col4" >0.018732</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col5" class="data row10 col5" >0.19995</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col6" class="data row10 col6" >-0.0551364</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col7" class="data row10 col7" >-0.172379</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col8" class="data row10 col8" >0.139057</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col9" class="data row10 col9" >-0.0252499</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col10" class="data row10 col10" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col11" class="data row10 col11" >-0.521813</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col12" class="data row10 col12" >-0.428815</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row10_col13" class="data row10 col13" >0.3161</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row11" class="row_heading level0 row11" >Hue</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col0" class="data row11 col0" >-0.617369</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col1" class="data row11 col1" >-0.0717472</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col2" class="data row11 col2" >-0.561296</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col3" class="data row11 col3" >-0.0746669</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col4" class="data row11 col4" >-0.273955</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col5" class="data row11 col5" >0.0553982</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col6" class="data row11 col6" >0.433681</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col7" class="data row11 col7" >0.543479</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col8" class="data row11 col8" >-0.26264</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col9" class="data row11 col9" >0.295544</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col10" class="data row11 col10" >-0.521813</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col11" class="data row11 col11" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col12" class="data row11 col12" >0.565468</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row11_col13" class="data row11 col13" >0.236183</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row12" class="row_heading level0 row12" >OD280OD315</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col0" class="data row12 col0" >-0.78823</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col1" class="data row12 col1" >0.0723432</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col2" class="data row12 col2" >-0.36871</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col3" class="data row12 col3" >0.00391123</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col4" class="data row12 col4" >-0.276769</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col5" class="data row12 col5" >0.0660039</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col6" class="data row12 col6" >0.699949</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col7" class="data row12 col7" >0.787194</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col8" class="data row12 col8" >-0.50327</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col9" class="data row12 col9" >0.519067</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col10" class="data row12 col10" >-0.428815</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col11" class="data row12 col11" >0.565468</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col12" class="data row12 col12" >1</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row12_col13" class="data row12 col13" >0.312761</td> 
    </tr>    <tr> 
        <th id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055level0_row13" class="row_heading level0 row13" >Proline</th> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col0" class="data row13 col0" >-0.633717</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col1" class="data row13 col1" >0.64372</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col2" class="data row13 col2" >-0.192011</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col3" class="data row13 col3" >0.223626</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col4" class="data row13 col4" >-0.440597</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col5" class="data row13 col5" >0.393351</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col6" class="data row13 col6" >0.498115</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col7" class="data row13 col7" >0.494193</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col8" class="data row13 col8" >-0.311385</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col9" class="data row13 col9" >0.330417</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col10" class="data row13 col10" >0.3161</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col11" class="data row13 col11" >0.236183</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col12" class="data row13 col12" >0.312761</td> 
        <td id="T_48b9f730_eb91_11e8_a49f_9eb6d0674055row13_col13" class="data row13 col13" >1</td> 
    </tr></tbody> 
</table> 



#### Ahora debemos dividir el dataset en entrenamiento y testing. 


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)
rfc = RandomForestClassifier
X = wineDS.drop('Class', axis = 1)
y = wineDS['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
```


```python
from sklearn.metrics import accuracy_score
modelos = [RandomForestClassifier(random_state=77), GradientBoostingClassifier(random_state=77), AdaBoostClassifier(random_state=77)]
```

#### Aquí realizamos entrenamiento y predicción:


```python
from sklearn.model_selection import cross_val_score, GridSearchCV

for model in modelos:
    unResultado = cross_val_score(model, X_train, y_train, cv=5)
    unMensaje = ("{0}:\n\tMedia de Precisión (training) \t= {1:.3f} "
           "(+/- {2:.3f})".format(model.__class__.__name__,
                                  unResultado.mean(),
                                  unResultado.std()))
    print(unMensaje)
    model.fit(X_train, y_train)
    unaPrediccion_test = model.predict(X_test)
    unaPrecision_test = accuracy_score(y_test, unaPrediccion_test)
    print("\tPrecisión (test)\t\t= {0:.3f}".format(unaPrecision_test))
```

    RandomForestClassifier:
    	Media de Precisión (training) 	= 0.965 (+/- 0.023)
    	Precisión (test)		= 1.000
    GradientBoostingClassifier:
    	Media de Precisión (training) 	= 0.944 (+/- 0.036)
    	Precisión (test)		= 0.944
    AdaBoostClassifier:
    	Media de Precisión (training) 	= 0.915 (+/- 0.018)
    	Precisión (test)		= 0.917
    

#### Como vemos en los resultados anteriores, el mejor  modelo para éste caso fué "Random Forest".
