import pandas

# Read the data
data = pandas.read_csv('creditcard.csv')

# create a sample of the data
sample = data.sample(frac=0.1, random_state=1)

# export the sample to a csv file
sample.to_csv('sample.csv', index=False)
