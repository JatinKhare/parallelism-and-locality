import pandas as pd
import glob
import os

#path = r'C:\DRO\DCL_rawdata_files' # use your path
all_files = glob.glob(os.path.join("*.csv"))

li = []

for filename in all_files:
				df = pd.read_csv(filename, index_col=None, header=0)
				li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
frame.to_csv("final.csv")
