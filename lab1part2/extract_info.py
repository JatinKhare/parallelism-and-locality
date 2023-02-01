import os
import csv
import sys

string1 = sys.argv[2];
#string1 = "ref-cycles:u"
# opening a text file
file1 = open(sys.argv[1], "r")
# setting flag and index to 0
flag = 0
index = 0
data = []
#os.remove(sys.argv[2] + ".csv")
file2 = open(sys.argv[3] + '_data_file.csv', 'x')
# Loop through the file line by line
for line in file1:
				index += 1
				#checking string is present in line or not
				if string1 in line:
								#print('String', string1, 'Found In Line', index)
								line = line.rsplit()
								print(line[0])
								data.append(line[0])
#closing text file	
file1.close()
#print(fma_perf)
writer = csv.writer(file2)
writer.writerow(data)
