#!/usr/bin/env python
# coding: utf-8

# # <font color='#d35400'> Lab 2 </font> | Gathering & Cleaning Data
# Welcome to Lab 2! This lab is about dealing with the Titanic dataset. For background information, the Titanic was a passenger ship that famously sank in 1912, with an estimated 1,500 people lost in the wreck.
# 
# ### <font color='#FF8C00'> About the Dataset </font>
# The dataset contains demographic information about passengers who were on board the Titanic as well as whether they survived or not. We focus on loading the data, building an understanding of the dataset, and cleaning the data for later use.
# 

# ## <font color = '#FF8C00'> Section 1 </font> | Reading the Data using Pandas & Manually
# To start off, we need to access the dataset. The dataset is in a `.csv` format, and is assumed to be in the same directory as this Jupyter notebook. In the cells below, we access the `.csv` file using Pandas first.

# In[1]:


# importing the pandas library
import pandas as pd

# importing the seaborn library
import seaborn as sns

# importing the matplotlib library
import matplotlib.pyplot as plt

# importing the csv library
import csv

# importing the IPython library
from IPython.display import display


# In[2]:


# reading the .csv file
titanic_df = pd.read_csv('titanic.csv')

# viewing the first 5 rows of the data frame
display(titanic_df.head(5))

# viewing the information of the data frame
titanic_df.info()

# viewing the description of the data frame
titanic_df.describe()


# Great! Now that we've accessed the `.csv` file using pandas, we can now try to acces the file manually using basic Python. We attempt this by reading the data into a dictionary where the <font color = '#e74c3c '> **keys** </font> are variable names and the <font color = '#e74c3c '> **values** </font> are lists.

# In[3]:


# initializing a dictionary
data_dictionary = {}

# extracting the keys from the dictionary
with open('titanic.csv') as file:
    first_line = file.readline().strip().split(",")

# allowing each key to have a empty list
data_dictionary = {key: [] for key in first_line}

# printing out the keys to check if done right
print("List of Keys: ", first_line, "\n")

# checking if the keys are added
print("Dictionary Keys: ", data_dictionary.keys())


# Cool! Now that we have set up the keys for the dictionary, we would like to <font color = '#e74c3c '> **add** </font> in the values. We achieve this by creating a list from each line and then <font color = '#e74c3c '> **adding** </font> them to each key using a for loop. We then <font color = '#e74c3c '> **print** </font> out the details of the dictionary to check if everything went right.

# In[4]:


# opening the titanic csv file
with open('titanic.csv') as file:
        
        # using the csv reader to read the file
        reader = csv.reader(file)

        # skipping the keys
        next(reader)

        for line in reader: 
            # appending each value to each key 
            for key, value in zip(data_dictionary.keys(), line):
                data_dictionary[key].append(value)

# setting up a test string
test_string = ""

# printing out the key and values of the dictionary
for key, value in data_dictionary.items():
    test_string += key + " "

    for val in value:
        test_string += val + " "
    
    test_string += "\n"

# printing out the test string
print(test_string)


# Now that we've manually created a dictionary, we will now convert the `data_dictionary` into a Pandas Dataframe. 

# In[5]:


# converting the dictionary into a data frame
data_dictionary_df = pd.DataFrame(data_dictionary)

# displaying the data frame
data_dictionary_df


# In[6]:


# displaying the info
display(data_dictionary_df.info())

# describing the data frame
display(data_dictionary_df.describe())


# ### <font color='#FF8C00'> Sources Used For Section One </font>
# - https://stackoverflow.com/questions/19746350/how-to-change-color-in-markdown-cells-ipython-jupyter-notebook
# - https://stackoverflow.com/questions/1904394/read-only-the-first-line-of-a-file
# - https://stackoverflow.com/questions/43375884/how-to-print-dictionary-values-line-by-line-in-python
# - https://stackoverflow.com/questions/4435169/how-do-i-append-one-string-to-another-in-python
# - https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
# 

# ## <font color = '#FF8C00'> Section 2 </font> | Representing the Data
# In this section, we work on <font color = '#e74c3c '> **manipulating** </font> the data that we have so that we can best represent it. Here, we make <font color = '#e74c3c '> **decisions** </font> on whether the variable should be represented as categorical or numerical, and convert them accordingly. We also work on <font color = '#e74c3c'> **plotting** </font> using Seaborn. We start by viewing the dataset below.

# In[7]:


# viewing the titanic dataset 
titanic_df.head(5)


# For the dataset above, we need to <font color = '#e74c3c '> **consider** </font> the `PClass`, `SibSp`, `Parch`, `Fare` and `Cabin` variables. We need to <font color = '#e74c3c '> **count** </font> the number of unique values for each and decide the best way to <font color = '#e74c3c '> **represent** </font> each - as integer, float or categorical.

# In[8]:


# initializing a list of column headings
column_headings = ["Pclass", "SibSp", "Parch", "Fare", "Cabin"]

# iterating through target column headings
for heading in column_headings:
    unique_value = titanic_df[heading].nunique()
    print(heading + ": " , unique_value , "unique values" )


# ### <font color = '#FF8C00'> Representation of Data </font>
# Details about the dataset can be found at [Kaggle](https://www.kaggle.com/c/titanic/data). According to the dataset description:
# - <font color = '#e74c3c'> **Pclass** </font> is the Ticket Class. The best way to represent this feature is categorically, as it is a categorical feature representing the class of ticket using 1, 2 and 3.
# - <font color = '#e74c3c'> **SibSp** </font> is the Number of Sibling / Spouses. The best way to represent this is as a integer, as it isn't possible to have floating point values for a discrete feature.
# - <font color = '#e74c3c'> **Parch** </font> is the Number of Parents / Children. The best way to represent this is as a integer, as it isn't possible to have floating point values for a discrete feature.
# - <font color = '#e74c3c'> **Fare** </font> is the Passenger Fare. The best way to represent this is as a floating point value, as currency can have floating point values.
# - <font color = '#e74c3c'> **Cabin** </font> is a Number of the Cabin. The best way to represent this is as a integer, as it is a discrete numerical feature.
# 
# Next, we need to convert a few features into <font color = '#e74c3c'> **categorical** </font> variables. The following features are Pclass, Sex, Cabin & Embarked. We achieve this using <font color = '#e74c3c'> **.astype()**  </font>.

# In[9]:


# converting the Pclass variable into a categorical feature
titanic_df['Pclass'] = titanic_df['Pclass'].astype('category')

# converting the Sex variable into a cateorical feature
titanic_df['Sex'] = titanic_df['Sex'].astype('category')

# converting the Cabin variable into a categorical feature
titanic_df['Cabin'] = titanic_df['Cabin'].astype('category')

# converting the Embarked variable into a categorical feature
titanic_df['Embarked'] = titanic_df['Embarked'].astype('category')

#viewing the titanic data frame
titanic_df.head(3)


# Next up, we plot a <font color = '#e74c3c'> **bar chart** </font> of the Survived column heading. For visualization, we use <font color = '#e74c3c'> **Seaborn** </font> to create a visually striking bar chart

# In[10]:


# converting the survived column into a list
survived_list = titanic_df['Survived'].values.tolist()
print("Survived Numbers: ", survived_list)

# counting the number of people who survived and didn't
alive = 0
dead = 0

# for looping through the list of survivors
for i in range(0, len(survived_list)):
    if survived_list[i] == 0:
        dead += 1
    else:
        alive += 1

# printing out the values of people alive and dead
print("Alive: ", alive)
print("Dead: ", dead)

# creating a new list
survival_categories = ["Alive", "Dead"]
survial_numbers = [alive, dead]

# creating a bar plot
sns.barplot(x=survival_categories, y =survial_numbers)

# adding the title and labels
plt.title("Survival Numbers on the Titanic")
plt.xlabel("Alive or Dead")
plt.ylabel("Numbers")


# We can also do this in an alternative manner, where we use Seaborn's  <font color = '#e74c3c'> **countplot** </font> function.

# In[11]:


# bar colors
bar_colors = {'0': '#AA4A44', '1':'#89CFF0'}

# counting the plot on a single categorical variable
sns.countplot(x='Survived', data=titanic_df, palette=bar_colors)

# labeling the axes and displaying the title
plt.title("Titanic Survival Numbers")
plt.ylabel("Numbers")

# displaying the plot
plt.show()


# ### <font color='#FF8C00'> Sources Used For Section Two </font>
# - https://www.kaggle.com/c/titanic/data
# - https://stackoverflow.com/questions/45759966/counting-unique-values-in-a-column-in-pandas-dataframe-like-in-qlik
# - https://stackoverflow.com/questions/54165508/convert-categorical-column-into-specific-integers
# - https://pandas.pydata.org/docs/user_guide/categorical.html
# - https://www.datacamp.com/tutorial/one-hot-encoding-python-tutorial
# - https://www.geeksforgeeks.org/countplot-using-seaborn-in-python/
# - https://seaborn.pydata.org/archive/0.11/generated/seaborn.countplot.html

# ## <font color = '#FF8C00'> Section 3 </font> | Missing Values
# In this section, we have the variable Age that appears to have a number of<font color = '#e74c3c'> **missing values** </font>. In this section, we work on filling these missing values using the <font color = '#e74c3c'> **mean, median and mode** </font>. We then work on using a simple <font color = '#e74c3c'> **supervised machine learning model (KNN)** </font> to fill the missing values as a alternative method.
# 

# ### <font color = '#FF8C00'> Mean, Median & Mode </font>
# We start by creating <font color = '#e74c3c'> **three** </font> new columns in the data set labeled as `Age_fill_mean`, `Age_fill_median`, `Age_fill_mode` by copying the original <font color = '#e74c3c'> **Age** </font> variable.

# In[12]:


# viewing the titanic dataset 
titanic_df.head(3)


# In[13]:


# creating the mean, median and mode column headings 
titanic_df['Age_fill_mean'] = titanic_df['Age']
titanic_df['Age_fill_median'] = titanic_df['Age']
titanic_df['Age_fill_mode'] = titanic_df['Age']

# checking for the presence of the columns
titanic_column_headings = ['Age_fill_mean', 'Age_fill_median', 'Age_fill_mode']
for column in titanic_column_headings:
    if column in titanic_df.columns:
        print(column, " is present")


# ### <font color = '#FF8C00'> Fill in Missing Values </font>
# Now that we have copied the `Age` column, we now need to work on filling out the <font color = '#e74c3c'> **missing values** </font>. Here are the following requirements:
# - Fill the missing values in `Age_ﬁll_mean` with the mean `Age` in the data
# - Fill the missing values in the `Age_fill_median` with the median `Age` in the data
# - Fill the missing values in the `Age_fill_mode` with the mode `Age` in the data

# In[14]:


# filling out the missing with the mean, median and mode
titanic_df['Age_fill_mean'] = titanic_df['Age_fill_mean'].fillna(titanic_df['Age_fill_mean'].mean())
titanic_df['Age_fill_median'] = titanic_df['Age_fill_median'].fillna(titanic_df['Age_fill_median'].median())
titanic_df['Age_fill_mode'] = titanic_df['Age_fill_mode'].fillna(titanic_df['Age_fill_mode'].mode()[0])

# checking for any null values
mean_null =  titanic_df['Age_fill_mean'].isnull().any()
median_null = titanic_df['Age_fill_median'].isnull().any()
mode_null = titanic_df['Age_fill_mode'].isnull().any()

# printing out the null values
print("Mean: ", mean_null)
print("Median: ", median_null)
print("Mode: ", mode_null)

# checking the titanic dataset
titanic_df.head(3)


# ### <font color = '#FF8C00'> Supervised Learning with KNN </font>
# We now move on to using a <font color = '#e74c3c'> **supervised learning model (KNN)** </font> to fill in the missing values. In order to accomplish this, we perform the <font color = '#e74c3c'> **following** </font> steps first:
# 1. Creating a new column in the data set called `Age_fill_knn`
# 2. Importing `KNeighborsRegressor` from `sklearn.neighbors`
# 

# In[15]:


# creating a new column in the data set
titanic_df['Age_fill_knn'] = titanic_df['Age']

# importing the knn
from sklearn.neighbors import KNeighborsRegressor


# Great! Now that we've created a new column and imported <font color = '#e74c3c'> **KNeighborsRegressor** </font>, we now need to divide our dataset into two datasets:
# 1. One called `data_drop_age_na` where there are no missing values for `Age`
# 2. One called `data_age_na` that contains all the rows with missing values for `Age`

# In[16]:


# new data frame where there are no missing values for age
data_drop_age_na = titanic_df[titanic_df['Age'].notna()]

# new dataframe where there are missing values for age
data_age_na = titanic_df[titanic_df['Age'].isna()]

# cross-checking the lengths
titanic_age_length = len(titanic_df['Age'])
drop_age_length = len(data_drop_age_na) 
missing_age_length = len(data_age_na)

# using if condition to check if the length are equal
if titanic_age_length == (drop_age_length + missing_age_length):
    print("Valid!")
else:
    print("Not Valid!")

# viewing the dataframe for no missing values
print("No Missing Values for Age")
display(data_drop_age_na.head(3))
print("\n")

# viewing the dataframe for missing values
print("All the Rows with Missing Values")
display(data_age_na.head(3))


# With the two datasets produced, we instantiate and fit a <font color = '#e74c3c'> **KNN** </font> model using `data_drop_age_na` where `n_neighbors` is set to 5, and the <font color = '#e74c3c'> **features** </font> used are `SibSp`, `Parch`, and `Fare`, and the response is `Age`. We then use this fitted model to <font color = '#e74c3c'> **predict** </font> in the `data_age_na` data frame.

# In[17]:


# the training dataset and the response variable
x_train = data_drop_age_na[['SibSp', 'Parch', 'Fare']]
y_train = data_drop_age_na['Age']

# the feature to predict
x_predict = data_age_na[['SibSp', 'Parch', 'Fare']]


# In[18]:


# instantiating the KNN model with 5 neighbors
titanic_knn = KNeighborsRegressor(n_neighbors=5)

# fitting the model with the training data
titanic_knn.fit(x_train, y_train)

# performing the predictions
age_predict = titanic_knn.predict(x_predict)

# printing out the results
print(age_predict)
print("Type: ", type(age_predict))


# Next, we would like to check the total number of <font color = '#e74c3c'> **missing** </font> values in `Age_fill_knn`  and cross check it with the number of values in the numpy array produced by our <font color = '#e74c3c'> **KNN** </font> model. This is to just check if the number of elements are the same on <font color = '#e74c3c'> **both** </font> sides. 

# In[19]:


# finding the total number of elements including missing values
total_value_knn = len(titanic_df['Age_fill_knn'])
print("Total is: ", total_value_knn)

missing_value_knn = titanic_df['Age_fill_knn'].isna().sum()
print("Missing Values: ", missing_value_knn)

# finding the length of the numpy array
total_elements = age_predict.size
print("Total Elements of Numpy Array: ", total_elements)


# Now that we have confirmed that <font color = '#e74c3c'> both </font> the missing values in the `age_fill_knn` and the number of values in the `age_predict` is the same, this means that we can go ahead and <font color = '#e74c3c'> fill </font> the missing values in the `age_fll_knn` from the numpy array.

# In[20]:


# boolean masking for missing values in age
missing_age_mask = titanic_df['Age_fill_knn'].isna()

# filling in missing values using the numpy array
titanic_df.loc[missing_age_mask, 'Age_fill_knn'] = age_predict

# checking the dataset again for missing values
new_missing_value_knn = titanic_df['Age_fill_knn'].isna().sum()
print("Missing Values: ", new_missing_value_knn)

# converting the pandas data frame into a .csv file for cross checking
titanic_df.to_csv("new_titanic.csv")


# Lastly, we would like to <font color = '#e74c3c'> compare </font> the results of the different methods of filling values using the mean, median, mode and using a supervised learning model (KNN). We achieve this using <font color = '#e74c3c'> seaborn </font>.

# In[21]:


# plotting the method of filling in the values using mode
sns.kdeplot(data=titanic_df['Age_fill_mode'], label="Mode Method")

# plotting the method of filling in the value using median
sns.kdeplot(data=titanic_df['Age_fill_median'], label="Median Method")

# plotting the method of filling in the value using mean
sns.kdeplot(data=titanic_df['Age_fill_mean'], label="Mean Method")

# plotting the method of filling in the value using knn
sns.kdeplot(data=titanic_df['Age_fill_knn'], label="KNN Method")

# setting the legend
plt.legend()

# setting the x and y axes and title
plt.title("Kernel Density Estimation Plots")
plt.xlabel("Age")
plt.ylabel("Probability Density")


# ### <font color='#FF8C00'> Sources Used For Section Three </font>
# - https://www.geeksforgeeks.org/adding-new-column-to-existing-dataframe-in-pandas/
# - https://www.quora.com/How-do-you-check-if-a-column-has-a-null-value-in-Pandas
# - https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
