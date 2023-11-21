
# PROJECT 4

Glassdoor Jobs (2017-18)

Introduction


Data from Glassdoor.com for 2 years (2017-18) has ~750 job postings in the US.  Using the salary listed for each job, various models are implemented to predict salaries.
Two types of models, viz., .Linear Regression and Logistic Classification models are used for salary prediction.

## PART I

### Visual Understanding of Data

Glassdoor data was obtained from Kaggle.com:

https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor

The file name:  eda_data.csv 
In Jupyter Notebook, a Python file ‘glassdoor_Data’ was created to ‘clean’ the dataset.  
The cleaned data has 742 records (job postings), and 14 features, viz.,

1.	'job_state',
2.	'job_region',
3.	'Sector',
4.	'Company_Age',
5.	'Avg_Company_Size',
6.	'Job_Title',
7.	'Avg_Salary',
8.	'Max_Salary',
9.	'Usr_Rating',
10.	'Python_y_n',
11.	'R_y_n',
12.	'Spark_y_n',
13.	'AWS_y_n',
14.	'Excel_y_n'

A SQLite database is created and a table Glassdoor_clean is created, in the database, with 16 features.  The SQLite file is accessed in Jupyter notebook for analysis using Python.

Re-formatted JSON files are also created for analysis using JavaScript.


#### DATA ANALYSIS

An overview of job prospects around the country (USA) in graphs  (Graph Descriptions):

Box-and whiskers plots of all continuous numeric features in the dataset.  
Each box includes 50% of the data within a feature.
Observations:
•	Most companies are between the ages of 10 years and 60 years.  Ages of some newly founded companies are not listed.
•	Most average annual salaries range between $75k and $130k.
•	Most maximum average salaries range between $90k to $160k.
•	Most employment sizes of companies range between 500 to 7500.
•	Most Glassdoor user ratings of a company, defined in the interval of 0 to 5, is between 3.3 and 4.

These ranges when viewed, give a general understanding of the overall data, without other categorical specifications, such as ‘state’ and ‘region’ where the jobs are located.


Violin plot displaying the types of jobs by title and corresponding salary range.  
All titles except those of ‘Manager’ and ‘Director’ have outliers toward higher ‘Average Salary’, indicating the availability of jobs with salaries higher than the norm.

The title ‘Research Scientist’ displays salaries, both lower and higher than the normal range.


Bar chart of average annual salary by sectors.  
24 sectors are listed in the dataset.  The bars are in descending order of salaries.  The 4 most paying jobs are in ‘Media’, ‘Accounting & Legal’, Information Technology and ‘Biotech & Pharmaceuticals’, all of which have more than $100k as average salaries.  ‘Construction, Repair & Maintenance’ is the lowest paying of all the listed sectors, with the average annual salary of less than $30k.


Bar chart showing the number of jobs available in each sector.
‘Information Technology’ tops the list followed by ‘Biotech & Pharmaceuticals’.  Both of these sectors are in the top 4, both in terms of job numbers and salary.  ‘Agriculture & Forestry’ has the least number of jobs.


Bubble chart to assess the number of jobs and the average salary for a sector at a glance.  
The height of the bubble from the x-axis shows the salary and the size of the bubble represents the number of jobs posted for a particular sector.
The chart clearly shows that ‘Information Technology’ (IT) and ‘Biotech & Pharmaceuticals’ (Biotech) are two of the highest and largest bubbles.


Bar chart that shows the number of available jobs by types and sectors.  
This shows that the number of job types like ‘Data Scientist’, ‘Data Analyst’ and ‘Data Engineer’ are available in most sectors, the greatest numbers being in ‘IT’ (gray).  ‘Research Scientist’ is most needed in ‘Biotech’ (violet) sector.  Title ‘Machine Learning Engineer’ (MLE) is available in sectors, ‘Education’, ‘IT’, ‘Aerospace & Defense’ and ‘Finance’.


Scatter plot Shows the Glassdoor user ratings data versus the age of the company.  
The pyramid shape suggests a normal curve for the data.  The mean / median is as user-rating of 3.7.  Well established companies older than 100 years received user rating close to the median value of 3.7.  This could be due to a large number of people employed by a company over the long period of existence of the company resulting in a large number of reviews and ratings leading to the average rating close to 3.7.
Younger companies less than 100 years old may not have a large set of review/rating data, hence the wide range of ratings ranging from 1.9 to 5, for companies less than 25 years old.

The size of the bubbles is proportionate to average salaries offered by companies.  A general trend of lower ratings for lower salaries and higher ratings for higher salaries can be deduced.



Pie chart showing 5 geographical regions of the USA (courtesy National Geographic) represented by the slices.  The size of a slice is equivalent to the percentage of average salary for the region out of the total amount  around the country.

The following is a list of regions and corresponding state codes of states that belong to each region:

•	'West':  WA, MT, OR, ID, WY, CA, NV, UT, CO, AK, HI,
•	'Southwest':  AZ, NM, TX, OK,
•	'Midwest':  ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH,
•	'Southeast':  KY, WV, VA, AR, TN, NC, SC, GA, AL, MS, LA, FL,
•	'Northeast':  PA, NY, VT, ME, NH, MA, RI, CT, NJ, DE, MD, DC

The chart shows that the earning capacity in ‘West’ exceeds that of the rest of the regions. Each region, however, may include states with very high paying jobs along with states with low paying jobs.


Bar plot showing the average salary vs. company size, i.e., the number of employees.  
It can be observed from the chart that there is no correlation between the average salary and the number of employees in a company.  
The average salary for the companies with number of employees ‘not listed’, offer the highest pay.  Some such companies are relatively new and are ≲ 10 years old.  Some such companies are:  Persivia, Kronos Bio, ALIN, Monte Rosa and Muso.



From all the above data visualization charts it can be inferred that the highest paying jobs as well as the largest number of jobs are in the sector:  Information Technology.  All ‘IT’ jobs require the knowledge of one or more of the following Programming Languages or Application Packages listed in the Glassdoor dataset:
•	Python,
•	R,
•	Spark,
•	Amazon Web Services (AWS)
•	Excel
Therefore, in the following plots, data is explored to analyze the earning potential with programming skills.

Histogram of job-count versus salary for each programming skill.  
All plots except one for ‘R’, resemble a ‘normal’ curve skewed right.  This implies that the average value of salary moves to the right and salary range is extended to the right.
The jobs requiring such skills offer much more than the median value of salary listed here:
•	Python: $107k
•	R: $70k
•	Spark: $108k
•	AWS: $107k
•	Excel: $92k

These median values are also the mode values with the largest number of jobs.


Bar plot illustrates the mean values of salaries for jobs requiring one or more of the 5 programming skills.  
Knowledge of Python, Spark and AWS are among the highest paying skills.


Scatter plot illustrates a combination of employee satisfaction and average salary for jobs with each skill.  
The blue dots in each scatter plot represent jobs with a particular skill for the plot.  In the plots for Python, Spark and AWS, the number of blue dots increases for salaries higher than the average value as well as the median ‘usr_rating’.



## PART II

### Linear Regression Models


OLS Linear Regression Model

To check that the underlying assumptions of no collinearity, Correlation Matrix was charted.  Some features which showed values  > 0.7 were dropped.  OLS model with the entire dataset showed R2 value of 42% and Durban-Watson value of 2.07.
Data was split into testing and training, as 0.25: 0.75.



Error Comparison Using OLS Linear Regression Model

	With Training Data	With Testing Data
MAE	21.7430			24.0752
MSE	736.4403		895.6746
RMSE	27.1374			29.9278
R2	41.39%			39.81%

*	Mean Absolute Error (MAE)
*	Mean Squared Error (MSE)
*	Root Mean Squared Error (RMSE)
*	R2 -Coefficient of Determination (R2-Square)


Homoscedasticity Test:

The graph shows uneven distribution, biased toward higher salary predictions.  This could be due to many outliers, that are greter than the 97.5th percentile of the data.

Quantile-Quantile Plot:

QQ-plot shows some data points on the 45 degree line, and most points above or below line, but close to the line.

Since, the R2 value is below 50%, Linear Regression Modeling with Decision Tree Regressor is also considered.


Tree Regression Model

Error Comparison

	With Training Data	With Testing Data
MAE	9.3584			18.8957
MSE	250.6310		739.4727
RMSE	15.8313			27.1932
R2	80.05%			50.31%

*	Mean Absolute Error (MAE)
*	Mean Squared Error (MSE)
*	Root Mean Squared Error (RMSE)
*	R2 -Coefficient of Determination (R2-Square)


Homoscedasticity Test:
 
The graph shows even distribution above and below the dotted line, but shows bias toward higher salary predictions.  This could be due to many outliers, that are greater than the 97.5th percentile of the data.

Quantile-Quantile-Plot

QQ-plot shows most points either on, or above or below 45 line.  QQ-plot for OLS regression shows more waviness compared to the tree regressor model.


CONCLUSION

•	R2 with Decision Tree Regressor is slightly > 50%.  
	Therefore, this model is a better predictor of salary than OLS model is with the 	present data set.



## PART III

### Logistic Regression Models

Target: Annual Median Salary ≥  $ 107 k ?

Model Evaluation Comparison

#### Decision Tree Classification Model

	With Training Data	With Testing Data
TN =	257			96
TP =	214			73
FP =	72			14
FN =	13			3

Acc:	0.85			0.91
 


#### Random Forest Classification Model

	With Training Data	With Testing Data
TN =	316			88
TP =	214			60
FP =	13			22
FN =	13			16

Acc:	0.95			0.80



#### ‘lbfgs’ Logistic Regression Model

	With Training Data	With Testing Data
TN =	275			80
TP =	154			45
FP =	54			30
FN =	73			31

Acc:	0.77			0.67




CONCLUSION

•	Decision Tree Model shows accuracy value to be high for both testing and training data (>90%)
•	Random Forest Model shows accuracy values of 95% for training and 80% for testing data.
•	‘lbfgs’ predicts 77% accuracy for training data and 67% for testing data.

Considering high testing accuracy of 80%, and low numbers of FN, Random Forest model is the best predictor of salary classification for this Glassdoor data.
























