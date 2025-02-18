#!/usr/bin/env python
# coding: utf-8

# # <font color='#d35400'> Lab 4 | Simulation Experiments </font>
# Welcome to Lab 4! In this lab, we work with the implications of sampling from a probability distribution. We start off this lab by plotting a single normal distribution. We then sample the distribution and look for any relationships. After writing code, we look at the effects of sampling which relate to the potential for Type I and Type II errors. Lastly, we use bootstrapping to estimate a confidence interval for the median of a dataset.
# 
# <p align="center">
#   <img src="dog_donut_painting.jpg" alt="Alt Text", width="300" />
# </p>

# ### <font color='#FF8C00'> About the Dataset </font>
# We start off this Jupyter Notebook by first examining the dataset from `dataset.csv`. We read this `.csv` file into a pandas data frame, upon which we then preview the dataset as a data frame by examining the first 5 rows.

# In[1]:


# importing the numpy library
import numpy as np

# random seed for reproducibility
np.random.seed(42)

# importing the pandas library
import pandas as pd

# importing the ipython library
from IPython.display import display

# reading the .csv file as a data frame
dataset_df = pd.read_csv("dataset.csv")

# checking for any null values
null_values = dataset_df.isnull().sum()
display("Number of Null Values: ", null_values)

# viewing the dataset
display(dataset_df.head(5))

# obtaining summary statistics
display(dataset_df.info())

# obtaining the description 
display(dataset_df.describe())

# displaying the median values
display(dataset_df.median())


# As we can see in this dataset, the three features collectively `variable_1`, `variable_2` and `variable_3`, are all of the data type `float64`.  There are 500 observations collectively. When we looked for any missing values, we happened to find none. Here are the summary statistics presented as follows:
# 
# <div align="center">
# 
# 
# | x                   | variable_1    | variable_2    | variable_3    |  
# |:-------------------:|:-------------:|:-------------:|:-------------:|  
# | mean                | -0.002162     | 0.822616      | 0.942165      |  
# | median              | 0.032894      | 0.797405      | 0.938862      |  
# | standard deviation  | 0.996963      | 1.006288      | 1.004571      |  
# | maximum             | 3.057066      | 4.150901      | 3.623870      |  
# | minimum             | -2.989982     | -2.428388     | -1.731287     |  
# | lower quartile      | -0.666835     | 0.176737      | 0.303725      | 
# | upper quartile      | 0.649753      | 1.498369      | 1.551920      |  
# 
# </div>
# 
# When we look at the summary statistics above, we noticed that the mean and median for each of the features are very similar. As a result, it would be difficult to determine whether there is positive skew or negative skew. We can use the `scipy` library to confirm this.

# In[2]:


# finding the skewness of each feature
skew_values = dataset_df.skew()
display(skew_values)


# Looking at the skew values, we can confirm that it is unlikely that each of the variables have any skewness in them. For the `variable_1` feature, since the skewness is less than 0, then the tail is on the left side, which means there is left-skew, which is a negative skew. 

# ## <font color = '#FF8C00'> Section 1 </font> | Plotting a Normal Distribution
# In this section, we are dealing with creating and plotting a normal distribution. Here are the following steps we take to achieve this:
# - [x] Import scipy and instantiate a normal distribution with µ = 0 and σ = 1
# - [x] Plot the pdf of this distribution (you may find .arange or .linspace useful).
# - [x] Plot the cdf of this distribution (you may ﬁnd .arange or .linspace useful).
# - [x] Using the inverse cdf and a unifrom distribution in the range of [0, 1), sample the distribution 1000 times. Plot the histogram of these 1000 observations (hint: lookup he probability point function).

# ### <font color = '#FF8C00'> Creating & Plotting a Probability Density Function </font>
# We start off by creating the probability density function. We initialize a normal distribution object using `norm()` and passing in the mean and standard deviation values. Next, we use `.linspace()` to create a array of equally spaced numbers from the `mean - 5*standard_deviation` to `mean + 5*standard_deviation`, which are plotted using 2000 generated values. 

# In[28]:


# importing the stats library
from scipy.stats import norm

# initializing the variables
mean = 0
standard_deviation = 1

# instantiating a normal distribution with Mean 1 and SD 1 
normal_distribution = norm(loc=mean, scale=standard_deviation)

# generating the values required for a distribution (2000 is number of points)
x_values = np.linspace(mean - 5*standard_deviation, mean + 5*standard_deviation, 2000);

# calulcating the probability density function
pdf_values = normal_distribution.pdf(x_values)

# calculating the mean of the sampling values
mean_value = np.mean(x_values)
print("Mean is: ", mean_value)

# calculating the standard deviation of the sampling values
std_value = np.mean(x_values)
print("STD is: ", std_value)


# We then move on to plotting the probability density function. Using the `matplotlib` library, we then plot the X values that we have generated against the density values that were generated after applying the `.pdf` function.

# In[4]:


# importing matplotlib
import matplotlib.pyplot as plt

# plotting the normal distribution
plt.plot(x_values, pdf_values, color="red")

# adding the title and labels
plt.title("PDF of a Normal Distrubtion Curve")
plt.xlabel("Values of X")
plt.ylabel("Density")

# saving the figure
plt.savefig(r"C:\GitHub\DataScience\wk_04\plots\PDFNormalDistribution.png")

# displaying the plot
plt.show()


# ### <font color = '#FF8C00'> Creating & Plotting  Cumulative Distribution Function </font>
# Next, we move on to creating the cumulative distribution function. Using the normalized distribution object, which has a passed in mean and standard deviation parameters, and the array of equally spaced numbers using `.linspace()`, we then use `.cdf()` to calculate the cumulative distribution function.

# In[5]:


# calculating the cumulative distirbution function
cdf_values = normal_distribution.cdf(x_values)


# We then move to plotting the cumulative distribution function. Using the `matplotlib` library, we then plot the X values that we have generated against the cumulative probabilities that were generated after applying the `.cdf` function. 

# In[6]:


# importing matplotlib
import matplotlib.pyplot as plt

# plotting the cumulative distribution
plt.plot(x_values, cdf_values, color="red")

# adding the title and labels
plt.title("CDF of a Normal Distrubtion Curve")
plt.xlabel("Values of X")
plt.ylabel("Cumulative Probability")

# saving the figure
plt.savefig(r"C:\GitHub\DataScience\wk_04\plots\CDFNormalDistribution.png")

# displaying the plot
plt.show()


# ### <font color = '#FF8C00'> The Inverse Cumulative Distribution Function </font>
# Next, we use the inverse cumulative distirbution function in the range of [0,1) and a sample of 1000 times. We then move on to plot a histogram of 1000 of these observations.  In order to find the inverse cumulative distribution, we use the `probability point function`, which returns the exact point where the probability of everything to the left is equal to the cumulative probability.

# In[7]:


# importing the uniform library from scipy
from scipy.stats import uniform

# generating random unform values between 0 and 1
# np.random.rand() is the uniform distirbution
random_uniform_values = np.random.rand(1000)

# calculating the probability point function values using mean 1 and SD 1
ppf_values = normal_distribution.ppf(random_uniform_values)


# Lastly, we then move on to plotting the probability point function as a histogram. Using the `matplotlib` library, we then plot the probability value on the x-axis and the cumulative probability on the y-axis after applying the `.ppf`.

# In[8]:


# plotting a histogram
plt.hist(ppf_values)

# setting up the titles and x and y headings
plt.title("Histogram of a Probability Point Function")
plt.xlabel("Probability")
plt.ylabel("Freqeuncy")

# saving the figure
plt.savefig(r"C:\GitHub\DataScience\wk_04\plots\PPFHistogram.png")

# displaying the plot
plt.show()


# ### <font color='#FF8C00'> Sources Used For Section One </font>
# - https://discovery.cs.illinois.edu/learn/Polling-Confidence-Intervals-and-Hypothesis-Testing/Python-Functions-for-Random-Distributions/

# ## <font color = '#FF8C00'> Section 2 </font> | Method for Sampling
# In this section, we create a function that takes in 2 parameters which in turn returns a list of length n elements. We use this function generate data and plot histograms. For each of these histograms, we plot a normal probability density function over the histogram. Here are the following steps to take:
# - [x] Using the results from problem 1, create a function that accepts 2 parameters - a scipy class distribution object and an integer n. Here, n is the number of samples to take from the distribution. This method should return a list of length n elements, where each element is the value of a single sample 
# - [x] Use your function to generate the following data and make histograms for 10, 20, 40, 80 and 160 samples
# - [x] For each one of these histograms, directly plot the normal pdf over the histogram (Hint: seaborns .histplot with stat=”probability” may be useful).
# 

# ### <font color = '#FF8C00'> Creating the Function </font>
# Here, we create a function that accepts 2 parameters, a `scipy` class distribution object and a integer `n`. Here, `n` is the number of samples to take from the distribution. This method should return a list of length n elements, where each element is the value of a single sample. 

# In[9]:


# a function that accepts two parameters
def sample_values(distribution, n):

    # generating random unform values between 0 and 1
    # np.random.rand() is the uniform distirbution
    random_uniform_values = np.random.rand(n)

    # calculating the probability point function values using mean 1 and SD 1
    ppf_values = distribution.ppf(random_uniform_values)

    # returning the ppf values
    return ppf_values


# ### <font color = '#FF8C00'> Plotting the Histograms </font>
# We use the function we created in the cell earlier to generate data and make histograms for each. Here are the details as follows:
# - Normal distribution - 10 samples
# - Normal distribution - 20 samples
# - Normal distribution - 40 samples
# - Normal distribution - 80 samples
# - Normal distribution - 160 samples

# In[10]:


# the mean, standard deviation
mean = 0
standard_deviation = 1

# instantiating the normal distribution
normal_dist = norm(loc=mean, scale=standard_deviation)

# list of sample sizes
sample_sizes = [10, 20, 40, 80, 160]

# for looping through the sample sizes
for size in sample_sizes:

    # a list of generated values
    generated_values = sample_values(normal_dist, size)

    # generating a histogram
    plt.hist(generated_values)

    # setting the title, x label and y label
    plt.title(f'Normal Distribution Histogram of {size} Samples')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # defining the file name
    file_name = f'NDH of {size} Samples'

    # saving the figure
    plt.savefig(rf"C:\GitHub\DataScience\wk_04\plots\{file_name}.png")
    
    # displaying the figures
    plt.show()


# ### <font color = '#FF8C00'> Normal PDF over Histogram </font>
# Lastly, for each one of the histograms, we directly plot the normal probability density function over each of the generated histograms. We achieve this by using the `seaborn` library.

# In[11]:


# importing the seaborn library
import seaborn as sns

# importing the scipy libraru
import scipy

# for looping through the sample sizes
for size in sample_sizes:
    generated_values = sample_values(normal_dist, size)

    # histogram plot using seaborn
    ax =  sns.histplot(generated_values, kde=False, stat='density', label='samples')

    # extracting end points for x-axis
    x0, x1 = ax.get_xlim()
    x_pdf = np.linspace(x0, x1, 100)
    y_pdf = scipy.stats.norm.pdf(x_pdf)

    # plotting the normal probability density function
    ax.plot(x_pdf, y_pdf, 'r', lw=2, label='pdf')

    # displaying the legend
    ax.legend()

    # displaying the title and x and y labels
    plt.title(f'PDF over Histogram of {size} Samples')
    plt.xlabel("Value")

    # defining the file name
    file_name = f'PDFHistogram {size} Samples'

    # saving the figure
    plt.savefig(rf"C:\GitHub\DataScience\wk_04\plots\{file_name}.png")

    # displaying the plot
    plt.show()


# ### <font color='#FF8C00'> Sources Used For Section Two </font>
# - https://stackoverflow.com/questions/30453097/getting-the-parameter-names-of-scipy-stats-distributions
# - https://stackoverflow.com/questions/52908925/how-to-add-a-standard-normal-pdf-over-a-seaborn-histogram

# ## <font color = '#FF8C00'> Section 3 </font> | Type 1 Errors
# The section involves generating two sample groups from normal distributions, calculating the effect size as the absolute difference in sample means, and repeating this process for k iterations. After generating 1000 iterations, a histogram of the effect sizes is plotted, and the sample groups corresponding to the largest observed effect size are visualized. Here are the following steps to take:
# - [x] Using the function from the previous problem, draw two sample groups using the same distribution and the same n. E.g. the ﬁrst group should be 25 samples drawn from a normal distribution with µ = 0 and σ = 1, the second group should be 25 diﬀerent samples drawn from a normal distribution with µ = 0 and σ = 1.
# - [x] Calculate the estimated eﬀect size between these two groups. (the absolute value of the diﬀerence between group 1 sample mean and group 2 sample mean).
# - [x] Create a function that accepts two distribution objects, two values of n for number of samples to draw from the distributions, and an integer k which is the number of times to loop the sample generations. This method should draw from each distribution n times and repeat this process for k iterations. The method should return three lists
# that are k elements long. The elements in the ﬁrst list are the estimated eﬀect sizes at iteration k between the drawn samples of the two distributions. Each element in the second list is a list of the drawn samples from distribution 1 at iteration k. Each element in the third list is a list of the drawn samples from distribution 2 at iteration k.
# - [x] Use the deﬁned method to generate two sample groups, each with 25 observations, for 1000 iterations. Plot a histogram of the eﬀect sizes. 
# - [x] Find the index of the largest observed eﬀect size. Use the index to plot the histograms of the two associated sample groups.

# ### <font color = '#FF8C00'> Drawing Two Sample Groups </font>
# We start off by using the previous function to draw two sample groups using the same distribution and the same n. For example:
# - the first group should be 25 samples drawn from a normal distribution with mean = 0 and SD = 1
# - the second group should be 25 different samples drawn from a normal distribution with mean = 0 and SD = 1

# In[12]:


# importing the random library
import random

# maintaining a constant n
n_size = 25

# instantiating the normal distribution
normal_dist_one = norm(loc=0, scale=1)

# generating random unform values between 0 and 1
# np.random.rand() is the uniform distirbution
random_uniform_values_one = np.random.rand(n_size)

# calculating the probability point function values using mean 1 and SD 1
ppf_values_one = normal_dist_one.ppf(random_uniform_values_one)

# instantiating the normal distribution
normal_dist_two = norm(loc=0, scale=1)

# generating random unform values between 0 and 1
# np.random.rand() is the uniform distirbution
random_uniform_values_two = np.random.rand(n_size)

# calculating the probability point function values using mean 1 and SD 1
ppf_values_two = normal_dist_two.ppf(random_uniform_values_two)


# ### <font color = '#FF8C00'> Estimated Effect Size </font>
# Next, we calculate the estimated effect size between the two groups. We calculate this by finding the absolute value of the difference between group 1 sample mean and group 2 sample mean.

# In[13]:


# finding the mean of group one sample
mean_one = sum(ppf_values_one) / len(ppf_values_one)
print("Mean of Random Sample One: ", mean_one)

# finding the mean grop two sample
mean_two = sum(ppf_values_two) / len(ppf_values_two)
print("Mean of Random Sample Two: ", mean_two)

# finding the absolute value of the difference
difference = mean_two - mean_one
absolute_difference = abs(difference)
print("Estimated Effect Size: ", absolute_difference)


# ### <font color = '#FF8C00'> Creating a Function </font>
# We now create a function that accepts two distribution objects, two values of n for number of samples to draw from the distributions, and a integer k which is the number of times to loop the sample generations.
# 
# This method should draw from each distribution n times and repeat this process for k iterations. As a result, this method should return three lists that are k elements long. 
# 
# - The elements in the first list are the estimated effect sizes at iteration k between the drawn samples of the two distributions
# - Each element in the second list is a list of the drawn samples from distribution 1 at iteration k
# - Each element in the third list is a list of the drawn samples from distribtuion 2 at iteration k

# In[14]:


# creating a function designed to take in 4 parameters
def two_distribution(distribution_one, distribution_two, n_one, n_two, k):
    """
    distribution one - a distribution object
    distribution two - a distribution object
    n_one - a number of samples
    n_two - a number of samples
    k - number of times to loop
    """

    # intitializing all the required lists
    effect_size = []
    sample_one_array = []
    sample_two_array = []

    # for looping through the values of k
    for _ in range(k):

        # generating random unform values between 0 and 1
        # np.random.rand() is the uniform distirbution
        random_uniform_values_one = np.random.rand(n_one)

        # calculating the probability point function values using mean 1 and SD 1
        ppf_values_one = distribution_one.ppf(random_uniform_values_one)

        # appending the ppf values one to the sample_one_array
        sample_one_array.append(ppf_values_one)

        # generating random unform values between 0 and 1
        # np.random.rand() is the uniform distirbution
        random_uniform_values_two = np.random.rand(n_two)

        # calculating the probability point function values using mean 1 and SD 1
        ppf_values_two = distribution_two.ppf(random_uniform_values_two)

        # appending the ppf values one to the sample_one_array
        sample_two_array.append(ppf_values_two)

        # calculating the effect size
        mean_one = np.mean(sample_one_array)
        mean_two = np.mean(sample_two_array)
        absolute_value = abs(mean_two - mean_one)
        effect_size.append(absolute_value)

    
    # returning the three lists
    return effect_size, sample_one_array, sample_two_array

    


# ### <font color = '#FF8C00'> Generating Two Sample Groups </font>
# We now use our defined method to generate two sample groups, each with 25 observations for 1000 iterations. We then move to plot a histogram of the effect sizes.

# In[15]:


# declaring the new mean and standard deviation variable
mean_one = 0
standard_deviation_one = 1

mean_two = 0
standard_deviation_two = 1

# caling the normal distribution objects
normal_dist_one = norm(loc=mean_one, scale=standard_deviation_one)
normal_dist_two = norm(loc=mean_two, scale=standard_deviation_two)

# storing the outputs of the functions
effect_size, sample_one_array, sample_two_array = two_distribution(normal_dist_one, normal_dist_two, 25, 25, 1000)

# plotting a histogram of the effect size
plt.hist(effect_size)

# adding the title, x and y axis label
plt.xlabel("Effect Size")
plt.ylabel("Iterations")
plt.title("Effect Size vs Iterations")

# saving the figure
plt.savefig(rf"C:\GitHub\DataScience\wk_04\plots\EffectSizeIterations.png")


# displaying the plot
plt.show()


# ### <font color = '#FF8C00'> Histogram of Sample Groups </font>
# Next, we find the index of the largest observed effect size. We use the index to plot the histograms of the of the two associated sample groups.

# In[16]:


# finding largest observed effect size
max_number = np.max(effect_size)

# finding the index number
index_number = effect_size.index(max_number)

# printing out the maximum number
print("Maximum Number is: ", max_number)

# variable for storing the index number
index = 0

# finding the index of the effect size
for i in range(0, len(effect_size)):
    print("Effect Size is: ", effect_size[i],"and index is ", i)
    if effect_size[i] == max_number:
        index += i

# printing out the index number
print("Index Number: ", index)


# In[17]:


# plotting the histogram of both samples using the index number
plt.hist(sample_one_array[index_number], label="Sample One")
plt.hist(sample_two_array[index_number], label="Sample Two")

# putting in the title, x label and y label
plt.title("Histogram of Largest Effect Size of Two Samples")
plt.xlabel("Effect Size")
plt.ylabel("Frequency")
plt.legend()

# saving the figure
plt.savefig(rf"C:\GitHub\DataScience\wk_04\plots\HistogramSample.png")

# displaying the plots
plt.show()


# ### <font color='#FF8C00'> Sources Used For Section Three </font>
# - https://www.geeksforgeeks.org/randomly-select-n-elements-from-list-in-python/
# - https://www.geeksforgeeks.org/remove-item-from-list-in-python/
# - https://www.physiotutors.com/wiki/effect-size/

# ## <font color = '#FF8C00'> Section 4 </font> | Type 2 Errors
# In this section, we work with Type II errors. We generate two normal distributions, then run 1000 iterations of sampling 25 values from each. We compute the effect sizes, plot their histogram, and identify the minimum effect size index. Finally, visualize the corresponding sample groups using histograms. Here are the following steps to take:
# - [x] Instantiate another normal distribution object with parameters µ = 1 and σ = 1
# - [x] Using the function developed in problem 3.3., repeat the experiment in part 3.4. using two diﬀerent distributions. This time the ﬁrst sample group should be 1000 iterations of 25 samples drawn from a normal distribution with µ = 0 and σ = 1 and the second group should be 1000 iterations of 25 diﬀerent samples drawn from a normal distribution with µ = 1 and σ = 1.  
# - [x] Plot the histogram of the estimated eﬀect sizes. Find the index of the minimum eﬀect size and plot the two sample groups associated with this index using a histogram.

# ### <font color = '#FF8C00'> Instantiating Normal Distributions </font>
# We start off by instantiating the normal distributions. The first one should have the mean set to 0 and standard deviation set to 1. The second one should have the mean set to 1 and standard deviation set to 1. Next, we run 1000 iterations of sampling 25 values each.

# In[ ]:


# setting up the mean and standard deviation
mean_one = 0
standard_deviation_one = 1

mean_two = 1
standard_deviation_two = 1

# caling the normal distribution objects
normal_dist_one = norm(loc=mean_one, scale=standard_deviation_one)
normal_dist_two = norm(loc=mean_two, scale=standard_deviation_two)

# storing the outputs of the functions
effect_size, sample_one_array_min, sample_two_array_min = two_distribution(normal_dist_one, normal_dist_two, 25, 25, 1000)


# ### <font color = '#FF8C00'> Plotting the Histogram </font>
# Next, we plot the histogram of the estimated effect sizes. We find the index of the minimum effect size and plot the two sample groups associated with this index using a histogram.

# In[19]:


# plotting the estimated effect sizes as a histogram
plt.hist(effect_size)

# naming the title and x and y axes
plt.title("Histogram of Effect Sizes")
plt.xlabel("Effect Size")
plt.ylabel("Frequency")

# saving the figure
plt.savefig(rf"C:\GitHub\DataScience\wk_04\plots\HistogramIterations.png")

# displaying the plot
plt.show()


# In[20]:


# finding smallest observed effect size
min_number = np.min(effect_size)

# finding the index number
index_number_small = effect_size.index(min_number)

# printing out the maximum number
print("Minimum Number is: ", min_number)

# variable for storing the index number
index = 0

# finding the index of the effect size
for i in range(0, len(effect_size)):
    print("Effect Size is: ", effect_size[i],"and index is ", i)
    if effect_size[i] == min_number:
        index += i

# printing out the index number
print("Index Number: ", index)


# In[21]:


# plotting the histogram of both samples using the index number
plt.hist(sample_one_array_min[index_number_small], label="Sample One")
plt.hist(sample_two_array_min[index_number_small], label="Sample Two")

# putting in the title, x label and y label
plt.title("Histogram of Smallest Effect Size of Two Samples")
plt.xlabel("Effect Size")
plt.ylabel("Frequency")
plt.legend()

# saving the figure
plt.savefig(rf"C:\GitHub\DataScience\wk_04\plots\HistogramSample.png")

# displaying the plots
plt.show()


# ### <font color='#FF8C00'> Sources Used For Section Four </font>
#  - Office Hours with Dr. Bukowy

# ## <font color = '#FF8C00'> Section 5 </font> | Bootstrapping
# In thi section, we implement a simple bootstrapping algorithm from scratch to estimate the confidence interval around the median of a data set. A confidence interval is a measure of how much the median is expected to vary. Here are the following deliverables:
# - [x]  Load in the data set from the assignment on canvas called some data.csv. There should be three variables in this data set. We will produce bootstrapped estimates of the conﬁdence interval around the median for each of the three variables.
# - [x]  For each of the three variables perform the following. For 1000 iterations: create a new bootstrap data set of 500 obsverations by sampling with replacement from the 500 observations generated in the data set; for each bootstrap data set compute the median and record it; sort the recorded values you have estimated for the median; take the 25th and 975th values as the bounds of your conﬁdence interval.

# ### <font color = '#FF8C00'> Loading the Dataset </font>
# We begin by loading the dataset form `data.csv`. We display this dataset using `pandas`.

# In[22]:


# viewing the dataset of the first 5 rows
dataset_df.head(5)


# ### <font color = '#FF8C00'> Bootstrapping </font>
# We have three variables that consists of 500 values. For each of these variables, we will use a for loop of 1,000 iterations to randomly sample the 500 values. We then compute the median of the resampled data and store the median.

# In[23]:


# converting variables to lists
variable_one = dataset_df['variable_1'].values.tolist()
variable_two = dataset_df['variable_2'].values.tolist()
variable_three = dataset_df['variable_3'].values.tolist()

# number of medians list
median_one = []
median_two = []
median_three = []

# for looping using iterations for variable_one
iterations = 1000
for i in range(iterations):
    sample = np.random.choice(variable_one)
    median_one.append(np.median(sample))

# for looping using iterations for variable_two 
for i in range(iterations):
    sample = np.random.choice(variable_two)
    median_two.append(np.median(sample))
        
# for looping using iterations for variable_three
for i in range(iterations):
    sample = np.random.choice(variable_three)
    median_three.append(np.median(sample))

# printing out the median lists
print("Median One: ", median_one, "\n")
print("Median Two: ", median_two, "\n")
print("Median Three: ", median_three, "\n")


# ### <font color = '#FF8C00'> Computing Confidence Interval </font>
# After 1,000 iterations, we will have 1,000 median values for each variable. We sort these median values in ascending order. Next, we find the 25th value and the 975th value. These values tell us the range within which the true median is likely to fall 95% of the time.

# In[24]:


# sorting the median values in ascending order
median_sort_one = np.sort(median_one)
median_sort_two = np.sort(median_two)
median_sort_three = np.sort(median_three)

# finding the confidence interval of median_one
lower_bound_one = median_sort_one[25]
upper_bound_one = median_sort_one[975]

# finding the confidence interval of median_two
lower_bound_two = median_sort_two[25]
upper_bound_two = median_sort_two[975]

# finding the confidence interval of median_three
lower_bound_three = median_sort_three[25]
upper_bound_three = median_sort_three[975]

# printing out the lower bounds for one, two, three
print("Lower Bound One: ", lower_bound_one, "\n")
print("Lower Bound Two: ", lower_bound_two, "\n")
print("Lower Bound Three: ", lower_bound_three, "\n")

# printing out the upper bounds for one, two, three
print("Upper Bound One: ", upper_bound_one, "\n")
print("Upper Bound Two: ", upper_bound_two, "\n")
print("Upper Bound Three: ", upper_bound_three, "\n")

