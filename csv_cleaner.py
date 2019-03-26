import csv
import os
import pandas
import numpy as np
import sys


# need to iterate over all the box score files 

for filename in os.listdir("datasets/"):
    if ".csv" in filename:
        # found a csv file to clean
        full_filename = "datasets/" + filename
        
        df = pandas.read_csv(full_filename)

        # print(df[df['WL'].isnull()])

        rows_to_delete = []

        for index, row in df.iterrows():
            has_no_wl_val = pandas.isnull(df.at[index, "WL"])

            if has_no_wl_val:
                rows_to_delete.append(index)




        # now have a list of rows to drop from the dataframe
        for rownum in rows_to_delete:
            row = df.drop(df.index[rownum])
        #overwrite the current file with the new clean dataset
        print(full_filename)
        df.to_csv(full_filename, index=None, header=True)