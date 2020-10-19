# Exporting `TabularPandas` for Inference (Intermediate)
> A guide for exporting `TabularPandas` and use it for inference with non-neural networks






---
This article is also a Jupyter Notebook available to be run from the top down. There
will be code snippets that you can then run in any environment.

Below are the versions of `fastai`, `fastcore`, and `wwf` currently running at the time of writing this:
* `fastai`: 2.0.16 
* `fastcore`: 1.1.2 
* `wwf`: 0.0.4 
---



## Using [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) as a Preprocessor

As mentioned in the [documentation](https://docs.fast.ai/tutorial.tabular#fastai-with-Other-Libraries) using `fastai` to preprocess our tabular data can be a nice way in which the library integrates with XGBoost and Random Forests. 

The issue though is when doing inference currently there is no way to export our [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object so we can do inference without building [`DataLoaders`](https://docs.fast.ai/data.core#DataLoaders) and exporting a [`Learner`](https://docs.fast.ai/learner#Learner). We'll solve this problem here and explain what we are doing. 

This is a much shorter article as it's currently an active [PR](https://github.com/fastai/fastai/pull/2857), but it will live here until the functionality is merged. 

## Grab the Data

Let's grab the `ADULT_SAMPLE` dataset quickly and work with it:

```python
from fastai.tabular.all import *
```

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
```

```python
df.head()
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>Private</td>
      <td>101320</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>1902</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>Private</td>
      <td>236746</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>10520</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>96185</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Divorced</td>
      <td>NaN</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>Self-emp-inc</td>
      <td>112847</td>
      <td>Prof-school</td>
      <td>15.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>Self-emp-not-inc</td>
      <td>82297</td>
      <td>7th-8th</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>



## Building our [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas)

Next we'll want to make our sample [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object. For our added `export` and `import` functionalities we will use the `@patch` method out of `fastcore` which means we can add them on later.

Let's build our `to` object:

```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [FillMissing, Categorify, Normalize]
splits = RandomSplitter()(range_of(df))

to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                   y_names=['salary'], splits=splits)
```

The nice part about [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) is now our data is completely preprocessed, as we can see blow by looking at a few rows of our `xs`:

```python
to.train.xs.head()
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
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18966</th>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0.030856</td>
      <td>-0.858388</td>
      <td>1.146335</td>
    </tr>
    <tr>
      <th>11592</th>
      <td>5</td>
      <td>13</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>-0.702314</td>
      <td>-0.021613</td>
      <td>1.537389</td>
    </tr>
    <tr>
      <th>548</th>
      <td>5</td>
      <td>16</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>-0.115778</td>
      <td>-0.336138</td>
      <td>-0.026827</td>
    </tr>
    <tr>
      <th>32008</th>
      <td>8</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0.764027</td>
      <td>0.129564</td>
      <td>-1.199989</td>
    </tr>
    <tr>
      <th>23657</th>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>13</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0.324124</td>
      <td>1.550315</td>
      <td>1.146335</td>
    </tr>
  </tbody>
</table>
</div>



## Exporting our [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas)

The next bit we want to do is actually add our export funcionality. We'll save it away as a pickle file:


<h4 id="TabularPandas.export" class="doc_header"><code>TabularPandas.export</code><a href="https://github.com/walkwithfastai/walkwithfastai.github.io/tree/master/wwf/tab/export.py#L9" class="source_link" style="float:right">[source]</a></h4>

> <code>TabularPandas.export</code>(**`fname`**=*`'export.pkl'`*, **`pickle_protocol`**=*`2`*)

Export the contents of `self` without the items


```python
@patch
def export(self:TabularPandas, fname='export.pkl', pickle_protocol=2):
    "Export the contents of `self` without the items"
    old_to = self
    self = self.new_empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickle.dump(self, open(Path(fname), 'wb'), protocol=pickle_protocol)
        self = old_to
```

And now we can directly use it:

```python
to.export('to.pkl')
```

## Loading It Back In

Now that we have exported our [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas), how do we use it in deployment? We'll make a [`load_pandas`](/tab.export.html#load_pandas) function to bring our pickle in:


<h4 id="load_pandas" class="doc_header"><code>load_pandas</code><a href="https://github.com/walkwithfastai/walkwithfastai.github.io/tree/master/wwf/tab/export.py#L20" class="source_link" style="float:right">[source]</a></h4>

> <code>load_pandas</code>(**`fname`**)

Load in a [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object from `fname`


```python
def load_pandas(fname):
    "Load in a `TabularPandas` object from `fname`"
    distrib_barrier()
    res = pickle.load(open(fname, 'rb'))
    return res
```

Let's do so for our newly exported `to`

```python
to_load = load_pandas('to.pkl')
```

And we can see it has no data:

```python
len(to_load)
```




    0



So how do we process some new data? the key is a combination of two functions:

* `to.train.new()`
* `to.process()`

The first will setup our data as though it is based on our training data and the second will run our `procs` through it. Let's try it out on a subset of our `DataFrame`:

```python
to_new = to_load.train.new(df.iloc[:10])
to_new.process()
```

And if we examine our data, we can see it's processed!

```python
to_new.xs.head()
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
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>8</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>0.764027</td>
      <td>-0.840572</td>
      <td>0.755281</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>13</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0.397441</td>
      <td>0.451042</td>
      <td>1.537389</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>-0.042461</td>
      <td>-0.889547</td>
      <td>-0.026827</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>15</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>-0.042461</td>
      <td>-0.730635</td>
      <td>1.928443</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>6</td>
      <td>3</td>
      <td>9</td>
      <td>6</td>
      <td>3</td>
      <td>2</td>
      <td>0.250807</td>
      <td>-1.022003</td>
      <td>-0.026827</td>
    </tr>
  </tbody>
</table>
</div>



To use this with your own projects simply make sure you've `pip` installed `wwf` and do:

```python
from wwf.tabular.export import *

to.export(fname)
```

After training and do what we did above for using your exported [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object with new data
