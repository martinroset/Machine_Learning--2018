

```python
import pandas as pd
import numpy
```

### Primero importamos los datasets (entrenamiento y validación), para después poder empezar a tratar los datos. ###


```python
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB

golf = "golf.csv"
golfT = "golf-test.csv"
golfFH = open(golf, "r")
golfEntFH = open(golfT, "r")
golfD = pd.read_csv(golfFH, sep=",")
golfEntD = pd.read_csv(golfEntFH, sep=",")
golfFH.close()
golfEntFH.close()
golfD.head(15)
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
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Wind</th>
      <th>Play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sunny</td>
      <td>85</td>
      <td>85</td>
      <td>False</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sunny</td>
      <td>80</td>
      <td>90</td>
      <td>True</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>overcast</td>
      <td>83</td>
      <td>78</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rain</td>
      <td>70</td>
      <td>96</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rain</td>
      <td>68</td>
      <td>80</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rain</td>
      <td>65</td>
      <td>70</td>
      <td>True</td>
      <td>no</td>
    </tr>
    <tr>
      <th>6</th>
      <td>overcast</td>
      <td>64</td>
      <td>65</td>
      <td>True</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sunny</td>
      <td>72</td>
      <td>95</td>
      <td>False</td>
      <td>no</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sunny</td>
      <td>69</td>
      <td>70</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>9</th>
      <td>rain</td>
      <td>75</td>
      <td>80</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>sunny</td>
      <td>75</td>
      <td>70</td>
      <td>True</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>overcast</td>
      <td>72</td>
      <td>90</td>
      <td>True</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>12</th>
      <td>overcast</td>
      <td>81</td>
      <td>75</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>rain</td>
      <td>71</td>
      <td>80</td>
      <td>True</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>



### Para éste ejercicio voy a utilizar Naives-Bayes Gauciano, por lo cual todos mis valores deben ser numéricos. ###
### Para esto convierto los valores categóricos (clases) en números, para poder trabajar con ellos en Naives Bayes Gauciano. ###


```python
a = {'sunny': 1, 'overcast': 2, 'rain': 3}
golfD.Outlook = [a[item] for item in golfD.Outlook.astype(str)]

b = {'False': 0, 'True': 1,}
golfD.Wind = [b[item] for item in golfD.Wind.astype(str)]

c = {'no': 0, 'yes': 1,}
golfD.Play = [c[item] for item in golfD.Play.astype(str)]

golfD.head(15)
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
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Wind</th>
      <th>Play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>85</td>
      <td>85</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>80</td>
      <td>90</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>83</td>
      <td>78</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>70</td>
      <td>96</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>68</td>
      <td>80</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>65</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>64</td>
      <td>65</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>72</td>
      <td>95</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>69</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>75</td>
      <td>80</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>75</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>72</td>
      <td>90</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>81</td>
      <td>75</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>71</td>
      <td>80</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
d = {'sunny': 1, 'overcast': 2, 'rain': 3}
golfEntD.Outlook = [d[item] for item in golfEntD.Outlook.astype(str)]

e = {'False': 0, 'True': 1,}
golfEntD.Wind = [e[item] for item in golfEntD.Wind.astype(str)]

f = {'no': 0, 'yes': 1,}
golfEntD.Play = [f[item] for item in golfEntD.Play.astype(str)]


golfEntD.head(15)

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
      <th>Outlook</th>
      <th>Temperature</th>
      <th>Humidity</th>
      <th>Wind</th>
      <th>Play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>85</td>
      <td>85</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>80</td>
      <td>90</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>83</td>
      <td>78</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>70</td>
      <td>96</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>68</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>65</td>
      <td>70</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>64</td>
      <td>65</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>72</td>
      <td>95</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>69</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>75</td>
      <td>80</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>68</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>72</td>
      <td>90</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2</td>
      <td>81</td>
      <td>75</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>71</td>
      <td>80</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Aquí selecciono los atributos de entrada y el label (Jugar) ###


```python
entrenamiento = golfD 
validacion = golfEntD

naivesBayes = GaussianNB()
entrenamientoF = entrenamiento.loc[:,['Outlook','Temperature','Humidity','Wind','Play']]
entrenamientoL = entrenamiento.iloc[:,4]
validacionF = validacion.loc[:,['Outlook','Temperature','Humidity','Wind','Play']]
validacionL = validacion.iloc[:,4]
```

### Entreno el modelo y muestro las labels, las predicciones ###


```python
naivesBayes.fit(entrenamientoF, entrenamientoL)
entrenamientoD = pd.concat([validacionF, validacionL], axis=1)
entrenamientoD["prediction"] = naivesBayes.predict(entrenamientoF)
print(entrenamientoD)
print ("Naive Bayes Accuracy:", naivesBayes.score(validacionF,validacionL))
```

        Outlook  Temperature  Humidity  Wind  Play  Play  prediction
    0         1           85        85     0     1     1           0
    1         2           80        90     1     0     0           0
    2         2           83        78     0     1     1           1
    3         3           70        96     0     1     1           1
    4         3           68        80     1     1     1           1
    5         3           65        70     1     0     0           0
    6         2           64        65     1     1     1           1
    7         1           72        95     0     0     0           0
    8         1           69        70     0     1     1           1
    9         1           75        80     0     0     0           1
    10        1           68        70     1     1     1           1
    11        2           72        90     1     1     1           1
    12        2           81        75     1     0     0           1
    13        3           71        80     1     1     1           0
    Naive Bayes Accuracy: 1.0
    


```python

```
