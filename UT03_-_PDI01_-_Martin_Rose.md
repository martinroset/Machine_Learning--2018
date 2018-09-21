

```python
import pandas as pd
import os
```


```python
unDir = os.getcwd()
print(unDir)
os.listdir('.')

```

    C:\Users\usuario\Desktop\Machine_Learning--2018\Jupyter_Notebook
    




    ['.ipynb_checkpoints',
     'Dataset_Iris_UCI.csv',
     'gender_submission.csv',
     'iris.data',
     'test.csv',
     'titanic.csv',
     'train.csv',
     'UT01_-_PDI02_-_Ej01_Martin_Rose.ipynb',
     'UT02_-_PDI04_-_Ej_01_-_Martín_Rose.ipynb',
     'UT02_-_PDI04_-_Ej_02_-_Martín_Rose.ipynb',
     'UT03_-_PDI01_-_Martin_Rose.xlsx',
     'UT03_-_PDI01_Martin_Rose.ipynb']




```python
import pandas as pd
unArchivo = pd.ExcelFile('UT03_-_PDI01_-_Martin_Rose.xlsx')
df = unArchivo.parse()
df.head(160)
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
      <th>Unnamed: 0</th>
      <th>x</th>
      <th>y</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>4</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>6</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NaN</td>
      <td>5</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>mean (x)</td>
      <td>mean (y)</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NaN</td>
      <td>3.5</td>
      <td>2.66667</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NaN</td>
      <td>xi- mean(x)</td>
      <td>yi - mean(y)</td>
      <td>(xi- mean(x)) * (yi- mean(y))</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NaN</td>
      <td>-2.5</td>
      <td>-1.66667</td>
      <td>4.16667</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NaN</td>
      <td>-0.5</td>
      <td>-0.666667</td>
      <td>0.333333</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NaN</td>
      <td>-1.5</td>
      <td>0.333333</td>
      <td>-0.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NaN</td>
      <td>0.5</td>
      <td>0.333333</td>
      <td>0.166667</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>NaN</td>
      <td>2.5</td>
      <td>-0.666667</td>
      <td>-1.66667</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NaN</td>
      <td>1.5</td>
      <td>2.33333</td>
      <td>3.5</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NaN</td>
      <td>Sumatoria=</td>
      <td>NaN</td>
      <td>6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NaN</td>
      <td>suma ^2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>NaN</td>
      <td>6.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>NaN</td>
      <td>0.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>NaN</td>
      <td>2.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>NaN</td>
      <td>0.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>NaN</td>
      <td>6.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>NaN</td>
      <td>2.25</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>NaN</td>
      <td>17.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>b1=</td>
      <td>0.342857</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>39</th>
      <td>NaN</td>
      <td>6</td>
      <td>3.52381</td>
      <td>2</td>
      <td>2.322</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NaN</td>
      <td>5</td>
      <td>3.18095</td>
      <td>5</td>
      <td>3.30893</td>
    </tr>
    <tr>
      <th>41</th>
      <td>NaN</td>
      <td>RMSE=</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.10123</td>
    </tr>
    <tr>
      <th>42</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>43</th>
      <td>NaN</td>
      <td>x</td>
      <td>yPrima</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>NaN</td>
      <td>0</td>
      <td>1.46667</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45</th>
      <td>NaN</td>
      <td>0.1</td>
      <td>1.50095</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>46</th>
      <td>NaN</td>
      <td>0.2</td>
      <td>1.53524</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>47</th>
      <td>NaN</td>
      <td>0.3</td>
      <td>1.56952</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>48</th>
      <td>NaN</td>
      <td>0.4</td>
      <td>1.60381</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>49</th>
      <td>NaN</td>
      <td>0.5</td>
      <td>1.6381</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50</th>
      <td>NaN</td>
      <td>0.6</td>
      <td>1.67238</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>51</th>
      <td>NaN</td>
      <td>0.7</td>
      <td>1.70667</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>52</th>
      <td>NaN</td>
      <td>0.8</td>
      <td>1.74095</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>53</th>
      <td>NaN</td>
      <td>0.9</td>
      <td>1.77524</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>54</th>
      <td>NaN</td>
      <td>1</td>
      <td>1.80952</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>55</th>
      <td>NaN</td>
      <td>1.1</td>
      <td>1.84381</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>56</th>
      <td>NaN</td>
      <td>1.2</td>
      <td>1.8781</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>57</th>
      <td>NaN</td>
      <td>1.3</td>
      <td>1.91238</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>58</th>
      <td>NaN</td>
      <td>1.4</td>
      <td>1.94667</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>59</th>
      <td>NaN</td>
      <td>1.5</td>
      <td>1.98095</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60</th>
      <td>NaN</td>
      <td>1.6</td>
      <td>2.01524</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>61</th>
      <td>NaN</td>
      <td>1.7</td>
      <td>2.04952</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>62</th>
      <td>NaN</td>
      <td>1.8</td>
      <td>2.08381</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>63</th>
      <td>NaN</td>
      <td>1.9</td>
      <td>2.1181</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>64</th>
      <td>NaN</td>
      <td>2</td>
      <td>2.15238</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>65</th>
      <td>NaN</td>
      <td>2.1</td>
      <td>2.18667</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>66</th>
      <td>NaN</td>
      <td>2.2</td>
      <td>2.22095</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>67</th>
      <td>NaN</td>
      <td>2.3</td>
      <td>2.25524</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>68</th>
      <td>NaN</td>
      <td>2.4</td>
      <td>2.28952</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>69 rows × 5 columns</p>
</div>




```python
df1 = unArchivo.parse()
print(df1.iloc[0:6,1:3])
```

       x  y
    0  1  1
    1  3  2
    2  2  3
    3  4  3
    4  6  2
    5  5  5
    


```python
print(df1.iloc[7:9,1:3])
```

              x         y
    7  mean (x)  mean (y)
    8       3.5   2.66667
    


```python
print(df1.iloc[10:16,1:4])
```

                  x             y                     Unnamed: 3
    10  xi- mean(x)  yi - mean(y)  (xi- mean(x)) * (yi- mean(y))
    11         -2.5      -1.66667                        4.16667
    12         -0.5     -0.666667                       0.333333
    13         -1.5      0.333333                           -0.5
    14          0.5      0.333333                       0.166667
    15          2.5     -0.666667                       -1.66667
    


```python
print(df1.iloc[17:18,1:4])
```

                 x    y Unnamed: 3
    17  Sumatoria=  NaN          6
    


```python
print(df1.iloc[19:26,1:2])
```

              x
    19  suma ^2
    20     6.25
    21     0.25
    22     2.25
    23     0.25
    24     6.25
    25     2.25
    


```python
print(df1.iloc[29:30,1:3])
print(df1.iloc[31:32,1:3])
```

          x         y
    29  b1=  0.342857
          x        y
    31  b0=  1.46667
    


```python
print(df1.iloc[34:42,1:5])
```

            x        y Unnamed: 3 Unnamed: 4
    34      x       y'          y   error ^2
    35      1  1.80952          1   0.655329
    36      3  2.49524          2   0.245261
    37      2  2.15238          3   0.718458
    38      4   2.8381          3  0.0262132
    39      6  3.52381          2      2.322
    40      5  3.18095          5    3.30893
    41  RMSE=      NaN        NaN    1.10123
    


```python
print(df1.iloc[43:71,1:3])
```

          x        y
    43    x   yPrima
    44    0  1.46667
    45  0.1  1.50095
    46  0.2  1.53524
    47  0.3  1.56952
    48  0.4  1.60381
    49  0.5   1.6381
    50  0.6  1.67238
    51  0.7  1.70667
    52  0.8  1.74095
    53  0.9  1.77524
    54    1  1.80952
    55  1.1  1.84381
    56  1.2   1.8781
    57  1.3  1.91238
    58  1.4  1.94667
    59  1.5  1.98095
    60  1.6  2.01524
    61  1.7  2.04952
    62  1.8  2.08381
    63  1.9   2.1181
    64    2  2.15238
    65  2.1  2.18667
    66  2.2  2.22095
    67  2.3  2.25524
    68  2.4  2.28952
    


```python
import matplotlib.pyplot as plt
import numpy as np
print(df1.iloc[43:71,1:3])

```

          x        y
    43    x   yPrima
    44    0  1.46667
    45  0.1  1.50095
    46  0.2  1.53524
    47  0.3  1.56952
    48  0.4  1.60381
    49  0.5   1.6381
    50  0.6  1.67238
    51  0.7  1.70667
    52  0.8  1.74095
    53  0.9  1.77524
    54    1  1.80952
    55  1.1  1.84381
    56  1.2   1.8781
    57  1.3  1.91238
    58  1.4  1.94667
    59  1.5  1.98095
    60  1.6  2.01524
    61  1.7  2.04952
    62  1.8  2.08381
    63  1.9   2.1181
    64    2  2.15238
    65  2.1  2.18667
    66  2.2  2.22095
    67  2.3  2.25524
    68  2.4  2.28952
    


```python

```
