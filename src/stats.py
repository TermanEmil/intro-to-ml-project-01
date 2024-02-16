
from main import *


# Find attribute names for raw data
X, attributeNames = importData()
attributeNames = attributeNames[:-3]
attributeNames.append('species')
# Import raw data instead of 1-out-of-K encoded
X = importRawData()


# Calculate statistics
mean_X = X.mean(axis=0)
std_X = X.std(axis=0,ddof=1)
median_X = np.median(X,axis=0)
range_X = X.max(axis=0) - X.min(axis=0)

# Display summary statistics as a table with row and column names
stats = pd.DataFrame(np.column_stack([mean_X,median_X,std_X,range_X]),
                     index=attributeNames,
                     columns=["Mean","Median","Standard deviation","Range"])
print(stats)


# Pairwise correlation - which ones are compared where??
pd.DataFrame(np.corrcoef(X,rowvar=False),       # rowvar false since the variables are the columns
             index=attributeNames,columns=attributeNames)
