

```python
import pandas as pd
```

#### Cargo el Dataset Iris, y muestro las primeras 10 filas


```python
unDataset = pd.read_csv('Dataset_Iris_UCI.csv')
unDataset.head(10)
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
      <th>sepal length in cm</th>
      <th>sepal width in cm</th>
      <th>petal length in cm</th>
      <th>petal width in cm</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.6</td>
      <td>3.4</td>
      <td>1.4</td>
      <td>0.3</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



#### Calculo la media para cada atributo


```python
unDataset.mean()
```




    sepal length in cm    5.843333
    sepal width in cm     3.054000
    petal length in cm    3.758667
    petal width in cm     1.198667
    dtype: float64



#### Calculo la varianza para cada atributo


```python
unDataset.var()
```




    sepal length in cm    0.685694
    sepal width in cm     0.188004
    petal length in cm    3.113179
    petal width in cm     0.582414
    dtype: float64



#### Calculo la desviación standard para cada atributo


```python
unDataset.std()
```




    sepal length in cm    0.828066
    sepal width in cm     0.433594
    petal length in cm    1.764420
    petal width in cm     0.763161
    dtype: float64



#### Información acerca de los atributos 

|       Dato            |              Descripción                         | 
|-----------------------|--------------------------------------------------| 
 sepal length in cm	    | Largo del sépalo de la flor (en centímetros) 	   
 sepal width in cm	    | Ancho del sépalo de la flor (en centímetros) 	    				
 petal length in cm	    | Largo del pétalo de la flor (en centímetros)      					
 petal width in cm	    | Ancho del pétalo de la flor (en centímetros) 	    				
 class 	                | Es el atributo a predecir, o "Label" (variable de salida),
 .                      | la cual hace referencia a la especie de la flor. 
 .                      | Esto se divide en 3 clases: Iris Setosa, 
 .                      |  Iris Versicolour, e Iris Virgínica.



```python

```
