import os
import csv
import sys
import pandas as pd
string1 = "ref-cycles";
string2 = "dcache-loads";
string3 = "dcache-load-misses";
string4 = "l2_rqsts.references";
string5 = "l2_rqsts.miss";
# opening a text file
file1 = open(sys.argv[1], "r")
# setting flag and index to 0
flag = 0
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data1.append(string1)
data2.append(string2)
data3.append(string3)
data4.append(string4)
data5.append(string5)
if os.path.exists(sys.argv[2] + "_data_file.csv"):
				os.remove(sys.argv[2] + "_data_file.csv")
else:
		print("DNE")
file2 = open(sys.argv[2] + '_data_file.csv', 'x')
# Loop through the file line by line
for line in file1:
				if string1 in line:
								line_t = line.rsplit()
								print(line_t[0])
								data1.append(line_t[0])
				if string2 in line:
								line_t = line.rsplit()
								print(line_t[0])
								data2.append(line_t[0])
				if string3 in line:
								line_t = line.rsplit()
								print(line_t[0])
								data3.append(line_t[0])
				if string4 in line:
								line_t = line.rsplit()
								print(line_t[0])
								data4.append(line_t[0])
				if string5 in line:
								line_t = line.rsplit()
								print(line_t[0])
								data5.append(line_t[0])
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df4 = pd.DataFrame(data4)
df5 = pd.DataFrame(data5)
df1 = df1.T
df2 = df2.T
df3 = df3.T
df4 = df4.T
df5 = df5.T
df= df1.append(df2).append(df3).append(df4).append(df5)
df = df.T
df.to_csv("please.csv")
file1.close()
writer = csv.writer(file2)
writer.writerow(data2)
