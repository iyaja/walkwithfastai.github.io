# Using Custom Transform Statistics (Intermediate)
> A guide for showing how to bring in pregenerated statistics for tabular






---
This article is also a Jupyter Notebook available to be run from the top down. There
will be code snippets that you can then run in any environment.

Below are the versions of `fastai`, `fastcore`, and `wwf` currently running at the time of writing this:
* `fastai`: 2.0.16 
* `fastcore`: 1.1.2 
* `wwf`: 0.0.4 
---



## Why Use Predetermined Stats?

If `fastai` will simply let us pass everything to a [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object to preprocess and train on, why should having custom statistics for our data? 

Let's try to think of a scenario.

My data is a few *trillion* rows, so there is no way (currently) I can load this `DataFrame` into memory at once. What do I do? Perhaps I would want to train on batches of my data at a time (article on this will come soon). To do this though I would **need** all of my `procs` predetermined from the start so every transform is done the same across all of our mini-batches of data. 

Currently there is an [open PR](https://github.com/fastai/fastai/pull/2837) for this integration, so for now this will live inside of Walk with fastai and we'll show how to use it as well!

Before we begin, let's import the tabular module:

## Modifying the `procs`

Now let's modify each of our `procs` to have this ability, as right now it's currently not there!

### Categorify

The first one we will look at is [`Categorify`](/tab.stats.html#Categorify). Currently the source code looks like so:

```python
class Categorify(TabularProc):
    "Transform the categorical variables to something similar to `pd.Categorical`"
    order = 1
    def setups(self, to):
        store_attr(classes={n:CategoryMap(to.iloc[:,n].items, add_na=(n in to.cat_names)) for n in to.cat_names})

    def encodes(self, to): to.transform(to.cat_names, partial(_apply_cats, self.classes, 1))
    def decodes(self, to): to.transform(to.cat_names, partial(_decode_cats, self.classes))
    def __getitem__(self,k): return self.classes[k]
```

What our modification needs to do is on the `__init__` we need an option to pass in a dictionary of class mappings, and [`setups`](/tab.stats.html#setups) needs to generate class mappings for those not passed in. Let's do so below:


<h2 id="Categorify" class="doc_header"><code>class</code> <code>Categorify</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L237" class="source_link" style="float:right">[source]</a></h2>

> <code>Categorify</code>(**`enc`**=*`None`*, **`dec`**=*`None`*, **`split_idx`**=*`None`*, **`order`**=*`None`*) :: [`TabularProc`](https://docs.fast.ai/tabular.core#TabularProc)

Transform the categorical variables to something similar to `pd.Categorical`


```python
class Categorify(TabularProc):
    "Transform the categorical variables to something similar to `pd.Categorical`"
    order = 1
    def __init__(self, classes=None):
        if classes is None: classes = defaultdict(L)
        store_attr()
        super().__init__()
    def setups(self, to):
        for n in to.cat_names:
            if n not in self.classes or is_categorical_dtype(to[n]):
                self.classes[n] = CategoryMap(to.iloc[:,n].items, add_na=n)

    def encodes(self, to): to.transform(to.cat_names, partial(_apply_cats, self.classes, 1))
    def decodes(self, to): to.transform(to.cat_names, partial(_decode_cats, self.classes))
    def __getitem__(self,k): return self.classes[k]
```

Now we have successfully set up our [`Categorify`](/tab.stats.html#Categorify). Let's look at a quick example below.

We'll make a `DataFrame` with two category columns:

```python
df = pd.DataFrame({'a':[0,1,2,0,2], 'b': ['a', 'b', 'a', 'c', 'b']})
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
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>a</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>c</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>b</td>
    </tr>
  </tbody>
</table>
</div>



Next we want to specify specific classes for `a`. We'll set a maximum range up to `4` rather than `2` shown in our `DataFrame`:

```python
tst_classes = {'a':L(['#na#',0,1,2,3,4])}; tst_classes
```




    {'a': (#6) ['#na#',0,1,2,3,4]}



Finally we will build a [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object with a modified version of [`Categorify`](/tab.stats.html#Categorify):

```python
to = TabularPandas(df, Categorify(classes=tst_classes), ['a','b'])
```

How do we tell it worked though? Let's check out `to.classes`, which is a shortcut for `to.procs.categorify.classes`. 

What we should see is our dictionary we mapped for `a`, which we do!

```python
to.classes
```




    {'a': (#6) ['#na#',0,1,2,3,4], 'b': ['#na#', 'a', 'b', 'c']}



### Normalize

Next let's move onto [`Normalize`](https://docs.fast.ai/data.transforms#Normalize). Things get a bit tricky here because we also need to update the base [`Normalize`](https://docs.fast.ai/data.transforms#Normalize) transform as well.

Why? 

Currently `fastai`'s [`Normalize`](https://docs.fast.ai/data.transforms#Normalize) tabular proc overrides the [`setups`](/tab.stats.html#setups) for [`Normalize`](https://docs.fast.ai/data.transforms#Normalize) by storing away our `means` and `stds`. What we need to do is have an option to pass in our `means` and `stds` in the base [`Normalize`](https://docs.fast.ai/data.transforms#Normalize). 

Let's do so here with `@patch`


<h4 id="Normalize.__init__" class="doc_header"><code>Normalize.__init__</code><a href="https://github.com/walkwithfastai/walkwithfastai.github.io/tree/master/wwf/tab/stats.py#L29" class="source_link" style="float:right">[source]</a></h4>

> <code>Normalize.__init__</code>(**`x`**:[`Normalize`](https://docs.fast.ai/data.transforms#Normalize), **`mean`**=*`None`*, **`std`**=*`None`*, **`axes`**=*`(0, 2, 3)`*, **`means`**=*`None`*, **`stds`**=*`None`*)

Initialize self.  See help(type(self)) for accurate signature.


Very nice little one-liner.

Integrating with tabular though will not be so nice as a one-liner. Our user scenario looks something like so:

We can pass in custom means *or* custom standard deviations, and these should be in the form of a dictionary similar to how we had our `classes` earlier. Let's modify [`setups`](/tab.stats.html#setups) to account for this:


<h4 id="setups" class="doc_header"><code>setups</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L371" class="source_link" style="float:right">[source]</a></h4>

> <code>setups</code>(**`to`**:[`Tabular`](https://docs.fast.ai/tabular.core#Tabular))




How do we test this? 

We'll do a similar scenario to our [`Categorify`](/tab.stats.html#Categorify) example earlier. We'll have one column:

```python
df = pd.DataFrame({'a':[0,1,2,3,4]})
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
      <th>a</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



And normalize them with some custom statistics. In our case we'll make them `3` for a mean and `1` for the standard deviation

```python
tst_means,tst_stds = {'a':3.}, {'a': 1.}
```

We'll pass this into [`Normalize`](https://docs.fast.ai/data.transforms#Normalize) and build a [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object:

```python
norm = Normalize(means=tst_means, stds=tst_stds)
to = TabularPandas(df, norm, cont_names='a')
```

We can then check our `mean` and `std` values:

```python
to.means, to.stds
```




    ({'a': 3.0}, {'a': 1.0})



And they line up!

### FillMissing

The last preprocesser is [`FillMissing`](/tab.stats.html#FillMissing). For this one we want to give `fastai` the ability to accept custom `na_dicts`, as this is where the information is stored on what continuous columns contains missing values! 

Compared to the last two, this integration is pretty trivial. First we'll give `__init__` the ability to accept a `na_dict`, then our [`setups`](/tab.stats.html#setups) needs to check if we have an `na_dict` already and what columns aren't there from it. First let's look at the old:

```python
class FillMissing(TabularProc):
    "Fill the missing values in continuous columns."
    def __init__(self, fill_strategy=FillStrategy.median, add_col=True, fill_vals=None):
        if fill_vals is None: fill_vals = defaultdict(int)
        store_attr()

    def setups(self, dsets):
        missing = pd.isnull(dsets.conts).any()
        store_attr(na_dict={n:self.fill_strategy(dsets[n], self.fill_vals[n])
                            for n in missing[missing].keys()})
        self.fill_strategy = self.fill_strategy.__name__

    def encodes(self, to):
        missing = pd.isnull(to.conts)
        for n in missing.any()[missing.any()].keys():
            assert n in self.na_dict, f"nan values in `{n}` but not in setup training set"
        for n in self.na_dict.keys():
            to[n].fillna(self.na_dict[n], inplace=True)
            if self.add_col:
                to.loc[:,n+'_na'] = missing[n]
                if n+'_na' not in to.cat_names: to.cat_names.append(n+'_na')
```

Followed by our new:


<h2 id="FillMissing" class="doc_header"><code>class</code> <code>FillMissing</code><a href="https://github.com/fastai/fastai/tree/master/fastai/tabular/core.py#L293" class="source_link" style="float:right">[source]</a></h2>

> <code>FillMissing</code>(**`fill_strategy`**=*`median`*, **`add_col`**=*`True`*, **`fill_vals`**=*`None`*) :: [`TabularProc`](https://docs.fast.ai/tabular.core#TabularProc)

Fill the missing values in continuous columns.


We can see our [`setups`](/tab.stats.html#setups) checks for what new `cont_names` we have and then updates our `na_dict` with those missing keys. Let's test it out below:

```python
df = pd.DataFrame({'a':[0,1,np.nan,1,2,3,4], 'b': [np.nan,1,2,3,4,5,6]})
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
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



We'll pass in a dictionary for `a` but not `b`:

```python
fill = FillMissing(na_dict={'a': 2.0}) 
to = TabularPandas(df, fill, cont_names=['a', 'b'])
```

And now let's look at our `na_dict`:

```python
to.na_dict
```




    {'a': 2.0, 'b': 3.5}



We can see that it all works!

## Full Integration Example

Nor for those folks that don't particularly care about how we get to this point and simply want to use it, we'll do the following:

```python
from wwf.tab.stats import *
from fastai.tabular.all import *
```

We'll make an example from the `ADULT_SAMPLE` dataset:

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



We'll set everything up as we normally would for our [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas):

```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
splits = RandomSplitter()(range_of(df))
```

Except we'll define every proc ourselves. For our [`Categorify`](/tab.stats.html#Categorify) example we will use `relationship`, [`Normalize`](https://docs.fast.ai/data.transforms#Normalize) will use `age`, and [`FillMissing`](/tab.stats.html#FillMissing) will use `education-num`:

### Categorify

First let's find those values:

```python
df['relationship'].unique()
```




    array([' Wife', ' Not-in-family', ' Unmarried', ' Husband', ' Own-child',
           ' Other-relative'], dtype=object)



And we'll set that as a dictionary with a ` Single ` class as well:

```python
classes = {'relationship': df['relationship'].unique() + [' Single ']}
```

And pass it to our [`Categorify`](/tab.stats.html#Categorify):

```python
cat = Categorify(classes=classes)
```

### Normalize

Next we have normalize. We'll use a (very) wrong mean and standard deviation of 15. and 7.:

```python
means,stds = {'age':15.}, {'age': 7.}
```

And pass it to [`Normalize`](https://docs.fast.ai/data.transforms#Normalize):

```python
norm = Normalize(means=means, stds=stds)
```

### FillMissing

Lastly we have our [`FillMissing`](/tab.stats.html#FillMissing), which we will simply fill with 5.:

```python
na_dict = {'education-num':5.}
```

And pass it in:

```python
fill = FillMissing(na_dict=na_dict) 
```

### Bringing it together

Now let's build our [`TabularPandas`](https://docs.fast.ai/tabular.core#TabularPandas) object:

```python
procs = [cat, norm, fill]
to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                   y_names=['salary'], splits=splits)
```

{% include note.html content='you may need to redefine your `cat_names` and `cont_names` here, this is because [`FillMissing`](/tab.stats.html#FillMissing) may override them' %}

And we're done! 

Thanks again for reading, and I hope this article helps you with your tabular endeavors!
