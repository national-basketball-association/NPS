import csv
import os
import pandas
import numpy as np
import sys


filepath = sys.argv[1] + '/' if len(sys.argv) == 2 else ''

# need to iterate over all the box score files
def clean():
    for filename in os.listdir(filepath + "datasets/"):
        if ".csv" in filename:
            # found a csv file to clean
            full_filename = filepath + "datasets/" + filename

            df = pandas.read_csv(full_filename)

            # print(df[df['WL'].isnull()])

            rows_to_delete = []

            for index, row in df.iterrows():
                has_no_wl_val = pandas.isnull(df.at[index, "WL"])

                if has_no_wl_val:
                    rows_to_delete.append(index)




            # now have a list of rows to drop from the dataframe
            modified_dataframe = df.drop(df.index[rows_to_delete])
            #overwrite the current file with the new clean dataset
            modified_dataframe.to_csv(full_filename, index=None, header=True)
