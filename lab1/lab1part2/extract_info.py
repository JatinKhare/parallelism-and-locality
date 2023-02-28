import os
import csv
import sys

string1 = sys.argv[2];
string2 = sys.argv[4];
# opening a text file
file1 = open(sys.argv[1], "r")
# setting flag and index to 0
flag = 0
data = []
if os.path.exists(sys.argv[3] + "_data_file.csv"):
				os.remove(sys.argv[3] + "_data_file.csv")
else:
		print("The file does not exist")
file2 = open(sys.argv[3] + '_data_file.csv', 'x')
# Loop through the file line by line
for line in file1:
				if string1 in line:
								line = line.rsplit()
								print(line[0])
								data.append(line[0])
				if string2 in line:
								line = line.rsplit()
								print(line[0])
								data.append(line[0])
file1.close()
writer = csv.writer(file2)
writer.writerow(data)
