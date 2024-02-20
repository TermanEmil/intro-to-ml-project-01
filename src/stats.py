
from main import *
from statistics import multimode


# Find attribute names for raw data
X, attributeNames = importData()
attributeNames = attributeNames[:-3]
# Import raw species data instead of 1-out-of-K encoded
X = importRawData()

# Calculate statistics for continuous attributes
mean_X = np.round(X[:,:-1].mean(axis=0),2)
std_X = np.round(X[:,:-1].std(axis=0,ddof=1),3)
median_X = np.round(np.median(X[:,:-1],axis=0),2)

# Display summary statistics as a table with row and column names
stats = pd.DataFrame(np.column_stack([mean_X,median_X,std_X]),
                     index=attributeNames,
                     columns=["Mean","Median","Standard deviation"])
print(stats)

# Determine mode of species
print(multimode(X[:,-1]))