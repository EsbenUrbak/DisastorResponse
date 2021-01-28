# import libraries
import pandas as pd
from sqlalchemy import create_engine

#loads the data from the csv paths and return the merged data in a panda
def load_data(messages_filepath, categories_filepath):
    """
    loads data from csv files, merged them and return a pandas containing the data

    parameters
    ----------
        messages_filepath: str
            the file path to the messages csv
        categories_filepath: str
            the file path to the categories csv

    Returns
    -------
        df: pandas
            panda containing the data
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on="id")
    return df

def clean_data(df):
    """
    cleans up the data in the pandas dataframe: converts the table data into integers (0 or 1) and renaming the columns

    parameters
    ----------
        df: pandas
            the pandas with the data to be cleaned

    Returns
    -------
        df: pandas
            panda containing the clean data
    """

    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat = ";", expand=True)
    # select the first row of the categories dataframe
    row = categories[:1]
    #cleaning up the text to so it can be used as the column names
    category_names_split=row.T[0].str.split("-", expand=True)
    category_colnames = category_names_split.iloc[:, [0]]
    category_colnames_list = category_colnames.values.tolist()
    category_colnames_list_flat = [item for sublist in category_colnames_list for item in sublist]
    categories.columns = category_colnames_list_flat
    #Convert category values to just numbers 0 or 1 (example related-1 => 1)
    for column in categories:
        #getting last charactor in each string
        categories[column] = categories[column].str[-1]
        #converting to integer type
        categories[column] =categories[column].astype(int)
    #Replace categories column in df with new category columns.
    df = df.drop('categories',axis = 1)
    df = pd.concat([df,categories], axis=1)
    #some numbers in "related" has the value 2. so converting these to 1
    df = df.drop_duplicates()
    df['related'].replace([2], [1], inplace=True)
    return df


def save_data(df, database_filename):
    """
    This saves the data to a sqlite database

    parameters
    ----------
        df: pandas
            pandas containing the data to be saved
        database_filename: str
            file path to where one wants the data saved

    Returns
    -------
    """
    engine = create_engine('sqlite:///data/'+database_filename+".db")
    df.to_sql(database_filename, engine, index=False, if_exists='replace')


def main():
    """
    This runs the ETL pipeline

    parameters
    ----------

    Returns
    -------

    """
    messages_filepath = "data\messages.csv"
    categories_filepath = "data\categories.csv"
    database_filepath = "DRDB"  #Dissastor Recovery Data Base

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')



if __name__ == '__main__':
    main()
