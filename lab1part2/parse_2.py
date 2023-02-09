import os
import csv
import sys
import pandas as pd
string1 = "LLC-load-misses";
string2 = "LLC-loads";
string3 = "cache-references";
string4 = "cache-misses";
string6 = "matmul";
# opening a text file
file1 = open(sys.argv[1], "r")
# setting flag and index to 0
flag = 0
data1 = []
data2 = []
data3 = []
data4 = []
data_0 = []
data_1 = []
data_2 = []
data_3= []
data1.append(string1)
data2.append(string2)
data3.append(string3)
data4.append(string4)
data_0.append("N")
data_1.append("B1")
data_2.append("B2")
data_3.append("B3")
if os.path.exists(sys.argv[2] + "_data_file.csv"):
				os.remove(sys.argv[2] + "_data_file.csv")
else:
		print("DNE")
file2 = open(sys.argv[2] + '_data_file.csv', 'x')
# Loop through the file line by line
for line in file1:
				if string1 in line:
								line_t = line.rsplit()
								#print(line_t[4])
								data1.append(line_t[0])
				if string2 in line:
								line_t = line.rsplit()
								data2.append(line_t[0])
				if string3 in line:
								line_t = line.rsplit()
								data3.append(line_t[0])
				if string4 in line:
								line_t = line.rsplit()
								data4.append(line_t[0])
				if string6 in line:
								line_t = line.rsplit()
								#print(line_t[-3])
								data_0.append(line_t[-6])
								data_1.append(line_t[-5])
								data_2.append(line_t[-4])
								data_3.append(line_t[-3].replace('\'', ''))
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)
df4 = pd.DataFrame(data4)
df_0 = pd.DataFrame(data_0)
df_1 = pd.DataFrame(data_1)
df_2 = pd.DataFrame(data_2)
df_3 = pd.DataFrame(data_3)
df1 = df1.T
df2 = df2.T
df3 = df3.T
df4 = df4.T
df_0 = df_0.T
df_1 = df_1.T
df_2 = df_2.T
df_3 = df_3.T
df = df_0.append(df_1)
df = df.append(df_2)
df = df.append(df_3)
df = df.append(df1)
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)
df = df.T
df.to_csv(sys.argv[2] + ".csv")
file1.close()
writer = csv.writer(file2)
writer.writerow(data2)
