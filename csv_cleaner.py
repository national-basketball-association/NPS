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

        print(df[df['WL'].isnull()])


        for index, row in df.iterrows():
            wl_val = df.at[index, "WL"]

            if len(wl_val) == 0:
                print(row)

        print()
