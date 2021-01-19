import pandas as pd
from sqlalchemy import create_engine

# load categories dataset
messages = pd.read_csv("messages.csv")
categories = pd.read_csv("categories.csv")
categories.head()

# merge datasets
df = messages.merge(categories, on='id')

# create a dataframe of the 36 individual category columns
categories = categories["categories"].str.split(";", expand=True)

# select the first row of the categories dataframe
row = categories[:1]
category_names_split=row.T[0].str.split("-", expand=True)
category_colnames = category_names_split.iloc[:, [0]]
category_colnames_list = category_colnames.values.tolist()
category_colnames_list_flat = [item for sublist in category_colnames_list for item in sublist]

# rename the columns of `categories`
categories.columns = category_colnames_list_flat

for column in categories:
    #First finding the position of "-""" in the text
    categories['pos'] = categories[column].str.find('-')
    #print(categories['pos'])
    #Using position to slice using a lambda function
    categories[column] = categories.apply(lambda x: x[column][x['pos']+1:x['pos']+2],axis=1)
    #print(".")
    # set each value to be the last character of the string

    # convert column from string to numeric
    categories[column] =categories[column].astype(int)
categories = categories.drop('pos',1)

# drop the original categories column from `df`
df = df.drop('categories',1)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df.reset_index(drop=True),categories.reset_index(drop=True)], axis=1)

# drop duplicates
df = df.drop_duplicates()

engine = create_engine('sqlite:///DR.db')
df.to_sql('DR', engine, index=False)
