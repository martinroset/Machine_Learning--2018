

```python
import pandas as pd
import numpy as np
import scipy
from sklearn import preprocessing
```

# importando el dataset


```python
dataset = pd.read_csv('wine_con_cabezales.csv')
dataset.head(10)
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
      <th>OD280/OD315 of diluted wines</th>
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
  </tbody>
</table>
</div>




```python
dataset.min()

```




    Class                             1.00
     Alcohol                         11.03
     Malic acid                       0.74
     Ash                              1.36
     Alcalinity of ash               10.60
    Magnesium                        70.00
     Total phenols                    0.98
    Flavanoids                        0.34
    Nonflavanoid phenols              0.13
    Proanthocyanins                   0.41
    Color intensity                   1.28
    Hue                               0.48
    OD280/OD315 of diluted wines      1.27
    Proline                         278.00
    dtype: float64




```python
dataset.max()
```




    Class                              3.00
     Alcohol                          14.83
     Malic acid                        5.80
     Ash                               3.23
     Alcalinity of ash                30.00
    Magnesium                        162.00
     Total phenols                     3.88
    Flavanoids                         5.08
    Nonflavanoid phenols               0.66
    Proanthocyanins                    3.58
    Color intensity                   13.00
    Hue                                1.71
    OD280/OD315 of diluted wines       4.00
    Proline                         1680.00
    dtype: float64




```python
dataset.describe()
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
      <th>OD280/OD315 of diluted wines</th>
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




```python
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 178 entries, 0 to 177
    Data columns (total 14 columns):
    Class                           178 non-null int64
     Alcohol                        178 non-null float64
     Malic acid                     178 non-null float64
     Ash                            178 non-null float64
     Alcalinity of ash              178 non-null float64
    Magnesium                       178 non-null int64
     Total phenols                  178 non-null float64
    Flavanoids                      178 non-null float64
    Nonflavanoid phenols            178 non-null float64
    Proanthocyanins                 178 non-null float64
    Color intensity                 178 non-null float64
    Hue                             178 non-null float64
    OD280/OD315 of diluted wines    178 non-null float64
    Proline                         178 non-null int64
    dtypes: float64(11), int64(3)
    memory usage: 19.5 KB
    


```python
dataset.Proanthocyanins.head(15)
```




    0     2.29
    1     1.28
    2     2.81
    3     2.18
    4     1.82
    5     1.97
    6     1.98
    7     1.25
    8     1.98
    9     1.85
    10    2.38
    11    1.57
    12    1.81
    13    2.81
    14    2.96
    Name: Proanthocyanins, dtype: float64



#### Acá vemos que todos los atributos son del tipo float o int, por lo tanto no es necesario convertir dichos atributos


```python
dataset[dataset["Proanthocyanins"]==1.25]
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
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
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
      <th>45</th>
      <td>1</td>
      <td>14.21</td>
      <td>4.04</td>
      <td>2.44</td>
      <td>18.9</td>
      <td>111</td>
      <td>2.85</td>
      <td>2.65</td>
      <td>0.30</td>
      <td>1.25</td>
      <td>5.24</td>
      <td>0.87</td>
      <td>3.33</td>
      <td>1080</td>
    </tr>
    <tr>
      <th>134</th>
      <td>3</td>
      <td>12.51</td>
      <td>1.24</td>
      <td>2.25</td>
      <td>17.5</td>
      <td>85</td>
      <td>2.00</td>
      <td>0.58</td>
      <td>0.60</td>
      <td>1.25</td>
      <td>5.45</td>
      <td>0.75</td>
      <td>1.51</td>
      <td>650</td>
    </tr>
    <tr>
      <th>148</th>
      <td>3</td>
      <td>13.32</td>
      <td>3.24</td>
      <td>2.38</td>
      <td>21.5</td>
      <td>92</td>
      <td>1.93</td>
      <td>0.76</td>
      <td>0.45</td>
      <td>1.25</td>
      <td>8.42</td>
      <td>0.55</td>
      <td>1.62</td>
      <td>650</td>
    </tr>
    <tr>
      <th>150</th>
      <td>3</td>
      <td>13.50</td>
      <td>3.12</td>
      <td>2.62</td>
      <td>24.0</td>
      <td>123</td>
      <td>1.40</td>
      <td>1.57</td>
      <td>0.22</td>
      <td>1.25</td>
      <td>8.60</td>
      <td>0.59</td>
      <td>1.30</td>
      <td>500</td>
    </tr>
  </tbody>
</table>
</div>



### Acá vamos a normalizar los valores del dataset


```python
vectorDeWine = dataset.values
minmaxSC= preprocessing.MinMaxScaler()
wineSC = minmaxSC.fit_transform(vectorDeWine)
wineDSC = pd.DataFrame(wineSC)
wineDSC.head(15)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.842105</td>
      <td>0.191700</td>
      <td>0.572193</td>
      <td>0.257732</td>
      <td>0.619565</td>
      <td>0.627586</td>
      <td>0.573840</td>
      <td>0.283019</td>
      <td>0.593060</td>
      <td>0.372014</td>
      <td>0.455285</td>
      <td>0.970696</td>
      <td>0.561341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.571053</td>
      <td>0.205534</td>
      <td>0.417112</td>
      <td>0.030928</td>
      <td>0.326087</td>
      <td>0.575862</td>
      <td>0.510549</td>
      <td>0.245283</td>
      <td>0.274448</td>
      <td>0.264505</td>
      <td>0.463415</td>
      <td>0.780220</td>
      <td>0.550642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.560526</td>
      <td>0.320158</td>
      <td>0.700535</td>
      <td>0.412371</td>
      <td>0.336957</td>
      <td>0.627586</td>
      <td>0.611814</td>
      <td>0.320755</td>
      <td>0.757098</td>
      <td>0.375427</td>
      <td>0.447154</td>
      <td>0.695971</td>
      <td>0.646933</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.878947</td>
      <td>0.239130</td>
      <td>0.609626</td>
      <td>0.319588</td>
      <td>0.467391</td>
      <td>0.989655</td>
      <td>0.664557</td>
      <td>0.207547</td>
      <td>0.558360</td>
      <td>0.556314</td>
      <td>0.308943</td>
      <td>0.798535</td>
      <td>0.857347</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.581579</td>
      <td>0.365613</td>
      <td>0.807487</td>
      <td>0.536082</td>
      <td>0.521739</td>
      <td>0.627586</td>
      <td>0.495781</td>
      <td>0.490566</td>
      <td>0.444795</td>
      <td>0.259386</td>
      <td>0.455285</td>
      <td>0.608059</td>
      <td>0.325963</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.834211</td>
      <td>0.201581</td>
      <td>0.582888</td>
      <td>0.237113</td>
      <td>0.456522</td>
      <td>0.789655</td>
      <td>0.643460</td>
      <td>0.396226</td>
      <td>0.492114</td>
      <td>0.466724</td>
      <td>0.463415</td>
      <td>0.578755</td>
      <td>0.835949</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.884211</td>
      <td>0.223320</td>
      <td>0.582888</td>
      <td>0.206186</td>
      <td>0.282609</td>
      <td>0.524138</td>
      <td>0.459916</td>
      <td>0.320755</td>
      <td>0.495268</td>
      <td>0.338737</td>
      <td>0.439024</td>
      <td>0.846154</td>
      <td>0.721826</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.797368</td>
      <td>0.278656</td>
      <td>0.668449</td>
      <td>0.360825</td>
      <td>0.554348</td>
      <td>0.558621</td>
      <td>0.457806</td>
      <td>0.339623</td>
      <td>0.264984</td>
      <td>0.321672</td>
      <td>0.471545</td>
      <td>0.846154</td>
      <td>0.725392</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.177866</td>
      <td>0.433155</td>
      <td>0.175258</td>
      <td>0.293478</td>
      <td>0.627586</td>
      <td>0.556962</td>
      <td>0.301887</td>
      <td>0.495268</td>
      <td>0.334471</td>
      <td>0.487805</td>
      <td>0.578755</td>
      <td>0.547076</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.744737</td>
      <td>0.120553</td>
      <td>0.486631</td>
      <td>0.278351</td>
      <td>0.304348</td>
      <td>0.689655</td>
      <td>0.592827</td>
      <td>0.169811</td>
      <td>0.454259</td>
      <td>0.506826</td>
      <td>0.430894</td>
      <td>0.835165</td>
      <td>0.547076</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.807895</td>
      <td>0.280632</td>
      <td>0.502674</td>
      <td>0.381443</td>
      <td>0.380435</td>
      <td>0.679310</td>
      <td>0.628692</td>
      <td>0.169811</td>
      <td>0.621451</td>
      <td>0.381399</td>
      <td>0.626016</td>
      <td>0.695971</td>
      <td>0.878745</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.813158</td>
      <td>0.146245</td>
      <td>0.513369</td>
      <td>0.319588</td>
      <td>0.271739</td>
      <td>0.420690</td>
      <td>0.440928</td>
      <td>0.245283</td>
      <td>0.365931</td>
      <td>0.317406</td>
      <td>0.560976</td>
      <td>0.567766</td>
      <td>0.714693</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.715789</td>
      <td>0.195652</td>
      <td>0.561497</td>
      <td>0.278351</td>
      <td>0.206522</td>
      <td>0.558621</td>
      <td>0.510549</td>
      <td>0.301887</td>
      <td>0.441640</td>
      <td>0.368601</td>
      <td>0.544715</td>
      <td>0.597070</td>
      <td>0.743224</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.978947</td>
      <td>0.195652</td>
      <td>0.550802</td>
      <td>0.041237</td>
      <td>0.228261</td>
      <td>0.731034</td>
      <td>0.706751</td>
      <td>0.566038</td>
      <td>0.757098</td>
      <td>0.351536</td>
      <td>0.626016</td>
      <td>0.534799</td>
      <td>0.621969</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0.881579</td>
      <td>0.223320</td>
      <td>0.545455</td>
      <td>0.072165</td>
      <td>0.347826</td>
      <td>0.800000</td>
      <td>0.696203</td>
      <td>0.301887</td>
      <td>0.804416</td>
      <td>0.530717</td>
      <td>0.585366</td>
      <td>0.633700</td>
      <td>0.905136</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
