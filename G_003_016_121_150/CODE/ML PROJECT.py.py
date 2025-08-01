print("TASK 1")
# importing libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm 
from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer, KNNImputer
from scipy.stats import zscore
from scipy import stats


#loading dataset into python
abc=pd.read_csv("C:\\Users\\bhara\\Downloads\\Gurgaon_RealEstate.csv")
#to identify all features and datatypes
print(abc.info())

#to check if there is any duplicate row 
DUPLICATEE_ROWSS=abc[abc.duplicated()]
#to print duplicate rows.
print("Duplicate Rows:",DUPLICATEE_ROWSS)
#to remove duplicate rows
abc=abc.drop_duplicates()

#to explore property_type column(flat or house)
plot.figure(figsize=(12,8))
sns.countplot(x='property_type',data=abc)
plot.title('property type distrubution')
plot.xlabel('property_type')
plot.ylabel('count')
plot.show()

# to explore society column
society_counts = abc['society'].value_counts()
# Define a threshold for the minimum number of flats or houses per society
threshold = 6
valid_societies = society_counts[society_counts >= threshold].index
abc = abc[abc['society'].isin(valid_societies)] 

# to explore price column
print("Missing values in price column:", abc['price'].isnull().sum())
print("Descriptive statistics for price column:\n", abc['price'].describe())
plot.figure(figsize=(12,8))
sns.histplot(abc['price'], bins=20, kde=True)
plot.title('Price Distribution')
plot.xlabel('Price')
plot.ylabel('Frequency')
plot.show()
plot.figure(figsize=(12,8))
sns.boxplot(x=abc['price'])
plot.title('Boxplot of Price')
plot.xlabel('Price')
plot.show()
print("Skewness of price column:", abc['price'].skew())
print("Kurtosis of price column:", abc['price'].kurt())
# to explore price_per_sqft

missing_values = abc['price_per_sqft'].isnull().sum()
print("Number of missing values:", missing_values)

print("Descriptive statistics of price_per_sqft column:")
print(abc['price_per_sqft'].describe())
#hostogram for price_per_sqft
plot.figure(figsize=(12,8))
sns.histplot(abc['price_per_sqft'], bins=20, kde=True, color='blue', edgecolor='black')
plot.xlabel('Price per Square Foot')
plot.ylabel('Frequency')
plot.title('Histogram of Price per Square Foot')
plot.show()
#boxplot for outliers of prive_per_sqft
plot.figure(figsize=(12,8))
sns.boxplot(x=abc['price_per_sqft'])
plot.xlabel('Price per Square Foot')
plot.title('Box Plot of Price per Square Foot')
plot.show()
#to check skewness and kurtosis
skewness = abc['price_per_sqft'].skew()
kurtosis = abc['price_per_sqft'].kurtosis()
print("Skewness:",skewness)
print("Kurtosis:",kurtosis)

# to explore area column
missing_values = abc['area'].isnull().sum()
print("Number of missing values:", missing_values)

print("Descriptive statistics of area column:")
print(abc['area'].describe())
#hostogram for area
plot.figure(figsize=(12,8))
sns.histplot(abc['area'], bins=20, kde=True, color='blue', edgecolor='black')
plot.xlabel('area')
plot.ylabel('Frequency')
plot.title('Histogram of area')
plot.show()
#boxplot for outliers of area
plot.figure(figsize=(12,8))
sns.boxplot(x=abc['area'])
plot.xlabel('area')
plot.title('Box Plot of area')
plot.show()
#to check skewness and kurtosis
skewness = abc['area'].skew()
kurtosis = abc['area'].kurtosis()
print("Skewness:",skewness)
print("Kurtosis:",kurtosis)
# to explore bedroom
missing_values = abc['bedRoom'].isnull().sum()
print("Number of missing values:", missing_values)

print("Descriptive statistics of bedRoom column:")
print(abc['bedRoom'].describe())
#hostogram for bedroom
plot.figure(figsize=(12,8))
sns.histplot(abc['bedRoom'], bins=20, kde=True, color='blue', edgecolor='black')
plot.xlabel('bedRoom')
plot.ylabel('Frequency')
plot.title('Histogram of bedRoom')
plot.show()
#boxplot for outliers of bedroom
plot.figure(figsize=(12,8))
sns.boxplot(x=abc['bedRoom'])
plot.xlabel('bedRoom')
plot.title('Box Plot of bedRoom')
plot.show()
#to check skewness and kurtosis
skewness = abc['bedRoom'].skew()
kurtosis = abc['bedRoom'].kurtosis()
print("Skewness:",skewness)
print("Kurtosis:",kurtosis)
# to explore bathroom
missing_values = abc['bathroom'].isnull().sum()
print("Number of missing values:", missing_values)

print("Descriptive statistics of bathroom column:")
print(abc['bathroom'].describe())
#hostogram for bathroom
plot.figure(figsize=(12,8))
sns.histplot(abc['bathroom'], bins=20, kde=True, color='blue', edgecolor='black')
plot.xlabel('bathroom')
plot.ylabel('Frequency')
plot.title('Histogram of bathroom')
plot.show()
#boxplot for outliers of bathroom
plot.figure(figsize=(12,8))
sns.boxplot(x=abc['bathroom'])
plot.xlabel('bathroom')
plot.title('Box Plot of bathroom')
plot.show()
#to check skewness and kurtosis
skewness = abc['bathroom'].skew()
kurtosis = abc['bathroom'].kurtosis()
print("Skewness:",skewness)
print("Kurtosis:",kurtosis)

# multivaratie analysis of all columns vs target column
# Perform multivariate analysis
# Property type vs price
plot.figure(figsize=(12,8))
sns.boxplot(x='property_type', y='price', data=abc)
plot.title('Property Type vs Price')
plot.xlabel('Property Type')
plot.ylabel('Price')
plot.xticks(rotation=45)
plot.show()

# Scatter plot between price and area
plot.figure(figsize=(12,8))
sns.scatterplot(x='area', y='price', data=abc)
plot.title('Price vs Area')
plot.xlabel('Area')
plot.ylabel('Price')
plot.show()

print("TASK 2")
#MISSING VALUE HANDLING

missval = abc.isnull().sum()
missdist = (missval / len(abc)) * 100
print("Missing Values Distribution:")
print(missval)
print(missdist) 


# FILLING NULL VALUES WITH MEAN OF A COLUMN 
abcd=abc.fillna(value=abc['carpet_area'].mean())
abcd=abc.fillna(value=abc['built_up_area'].mean())
abcd=abc.fillna(value=abc['super_built_up_area'].mean())
abcd=abc.fillna(value=abc['price'].mean())
abcd=abc.fillna(value=abc['price_per_sqft'].mean())


#ONEHOT ENCODING
# from sklearn.preprocessing import OneHotEncoder

# abcd.tail()
# abcd.dtypes
# abcd["society"].unique()
# abcd["facing"].unique()
# ohe = OneHotEncoder()
# ohe.fit_transform(abcd[["society","facing"]]).toarray()
# featurearr = ohe.fit_transform(abcd[["society","facing"]]).toarray()
# featlabels = ohe.get_feature_names_out(["society","facing"])
# np.array(featlabels).ravel() 
# featlabels = np.array(featlabels).ravel()
# print(featlabels)
# pd.dataframe(featurearr, columns = featlabels)
# features = pd.dataframe(featurearr, columns = featlabels)
# print(features)
# pd.concat([abc, features], axis=1)
# abcnew = pd.concat([abc, features], axis=1)

# handling missing values for categorical column 
#abcd=abcnew.fillna(value=abcnew['society'].mean())
#abcd=abcnew.fillna(value=abcnew['facing'].mean())
#abcd

abcd["society"].replace(np.NaN, abcd["society"].mode()[0], inplace=True)
abcd["facing"].replace(np.NaN, abcd["facing"].mode()[0], inplace=True)
print("\n MISSING VALUES AFTER HANDLING")
print(abcd.isnull().sum())


# Display initial data with missing values
print("Initial Data with Missing Values:")
print(abc.head(10))

# Visualize missing data
sns.heatmap(abc.isnull(), cbar=False, cmap="viridis")
plot.title('Heatmap of Missing Values')
plot.show()


# Strategy 1: Deletion
abc_deletion = abc.dropna()
print("\nData after Deletion:")
print(abc_deletion.head(10))

print("TASK 3")
print("OUTLIERS DETECTION")

# Select only numerical columns
numerical_abc = abc.select_dtypes(include=[np.number])
# Display descriptive statistics
print(numerical_abc.describe())
# Z-score method
z_score = np.abs(stats.zscore(numerical_abc))
outlier_z = (z_score > 3).any(axis=1)
print(f"Outliers detected by Z-score:\n{numerical_abc[outlier_z]}")
# IQR method
Q1 = numerical_abc.quantile(0.25)
Q3 = numerical_abc.quantile(0.75)
IQR = Q3 - Q1
outlier_iqr = ((numerical_abc < (Q1 - 1.5 * IQR)) | (numerical_abc > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"Outliers detected by IQR:\n{numerical_abc[outlier_iqr]}")
# Visualization of distributions and outliers
fig, ax = plot.subplots(len(numerical_abc.columns), 2, figsize=(16, 4 * len(numerical_abc.columns)))
for i, feature in enumerate(numerical_abc.columns):
 sns.histplot(numerical_abc[feature], kde=True, ax=ax[i, 0])
 ax[i, 0].set_title(f'Histogram of {feature}')
 sns.boxplot(x=numerical_abc[feature], ax=ax[i, 1])
 ax[i, 1].set_title(f'Box plot of {feature}')
plot.tight_layout()
plot.show()

#OUTLIERS DETECTION FOR NORMALLY DISTRUBUTED COLUMN THROUGH ZSCORE
#%matplotlib inline 
#matplotlib.rcParams['figure.figsize'] = (12,8)
#bedroom

#plot.hist(abcd.bedRoom, bins=20, rwidth=0.8)
#plot.xlabel('bedRoom')
#plot.ylabel('count')
#plot.show()

#rngg = np.arange(abcd.bedRoom.min(), abcd.bedRoom.max(), 0.1)
#plot.plot(rngg, norm.pdf(rngg, abcd.bedRoom.mean(), abcd.bedRoom.std()))  
#upperlim = abcd.bedRoom.mean() + 3*abcd.bedRoom.std()
#lowerlim = abcd.bedRoom.mean() - 3*abcd.bedRoom.std()
#abcd[(abcd.bedRoom>upperlim)|(abcd.bedRoom<lowerlim)]
#abcde=abcd[(abcd.bedRoom>upperlim)|(abcd.bedRoom<lowerlim)]
#abcd['zscore'] = (abcd.bedRoom - abcd.bedRoom.mean())/abcd.bedRoom.std()
#bathroom
#plot.hist(abcd.bathroom, bins=20, rwidth=0.8)
#plot.xlabel('bathroom')
#plot.ylabel('count')
#plot.show()
            
#rngg = np.arange(abcd.bathroom.min(), abcd.bathroom.max(), 0.1)
#plot.plot(rngg, norm.pdf(rngg, abcd.bathroom.mean(), abcd.bathroom.std()))  
#upperlim = abcd.bathroom.mean() + 3*abcd.bathroom.std()
#lowerlim = abcd.bathroom.mean() - 3*abcd.bathroom.std()
#abcd[(abcd.bathroom>upperlim)|(abcd.bathroom<lowerlim)]
#abcde=abcd[(abcd.bathroom>upperlim)|(abcd.bathroom<lowerlim)]
#abcd['zscore'] = (abcd.bathroom - abcd.bathroom.mean())/abcd.bathroom.std()
           
# #age possession
# plot.hist(abcd.agePossession, bins=20, rwidth=0.8)
# plot.xlabel('agePossession')
# plot.ylabel('count')
# plot.show()
           
# rngg = np.arange(abcd.agePossession.min(), abcd.agePossession.max(), 0.1)
# plot.plot(rngg, norm.pdf(rngg, abcd.agePossession.mean(), abcd.agePossession.std()))  
# upperlim = abcd.agePossession.mean() + 3*abcd.agePossession.std()
# lowerlim = abcd.agePossession.mean() - 3*abcd.bedRoom.std()
# abcd[(abcd.agePossession>upperlim)|(abcd.agePossession<lowerlim)]
# abcde=abcd[(abcd.agePossession>upperlim)|(abcd.agePossession<lowerlim)]
# abcd['zscore'] = (abcd.agePossession - abcd.agePossession.mean())/abcd.agePossession.std()
print("TASK 4")
print("OUTLIERS HANDLING")
# Select only numerical columns
numerical_abc = abc.select_dtypes(include=[np.number])
# Display initial descriptive statistics
print("Initial Descriptive Statistics:")
print(numerical_abc.describe())
# Z-score method to identify outliers
z_score = np.abs(stats.zscore(numerical_abc))
outlier_z = (z_score > 3).any(axis=1)
print(f"\nOutliers detected by Z-score:\n{numerical_abc[outlier_z]}")
# IQR method to identify outliers
Q1 = numerical_abc.quantile(0.25)
Q3 = numerical_abc.quantile(0.75)
IQR = Q3 - Q1
outlier_iqr = ((numerical_abc < (Q1 - 1.5 * IQR)) | (numerical_abc > (Q3 + 1.5 * IQR))).any(axis=1)
print(f"\nOutliers detected by IQR:\n{numerical_abc[outlier_iqr]}")

#STRATERGY 1

winsorized_abc = numerical_abc.apply(lambda x: winsorize(x, limits=[0.05, 0.05]))

# Strategy 2: Trimming
trimmed_abc = numerical_abc[~outlier_z]
# Strategy 3: Log Transformation (example for one feature, can be applied to all if needed)
transformed_abc = numerical_abc.apply(lambda x: np.log1p(x))

# Visualize the distributions and outliers before and after handling
fig, axs = plot.subplots(4, len(numerical_abc.columns), figsize=(20, 16))
for i, feature in enumerate(numerical_abc.columns):
 sns.histplot(numerical_abc[feature], kde=True, ax=axs[0, i])
 axs[0, i].set_title(f'Original {feature}')
 
 sns.histplot(winsorized_abc[feature], kde=True, ax=axs[1, i])
 axs[1, i].set_title(f'Winsorized {feature}')
 
 sns.histplot(trimmed_abc[feature], kde=True, ax=axs[2, i])
 axs[2, i].set_title(f'Trimmed {feature}')
 
 sns.histplot(transformed_abc[feature], kde=True, ax=axs[3, i])
 axs[3, i].set_title(f'Log Transformed {feature}')
plot.tight_layout()
plot.show()

 # Function to calculate metrics for evaluating impact
def calculatemetrics(data, title):
 metrics = pd.DataFrame({
 'Mean': data.mean(),
 'Std Dev': data.std(),
 'Skewness': data.skew(),
 'Kurtosis': data.apply(lambda x: stats.kurtosis(x))
 })
 print(f'\n{title}')
 print(metrics)
# Evaluate the impact of outlier handling
print("\nMetrics Before Handling Outliers:")
calculatemetrics(numerical_abc, 'Original Data Metrics')
print("\nMetrics After Winsorization:")
calculatemetrics(winsorized_abc, 'Winsorized Data Metrics')
print("\nMetrics After Trimming:")
calculatemetrics(trimmed_abc, 'Trimmed Data Metrics')
print("\nMetrics After Log Transformation:")
calculatemetrics(transformed_abc, 'Log Transformed Data Metrics')