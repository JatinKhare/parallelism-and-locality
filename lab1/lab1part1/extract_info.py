import csv
#inport sys
#string1 = sys.argv[1];
string1 = " cycles:u"
# opening a text file
file1 = open("temp", "r")
# setting flag and index to 0
flag = 0
index = 0
temp = 0
fma_perf = []
file2 = open('cycles_32.csv', 'x')
# Loop through the file line by line
for line in file1:
	index += 1
	#checking string is present in line or not
	if string1 in line:
		flag = 1
		temp++;
#checking condition for string found or not
	#print('String', string1, 'Found In Line', index)
	if(temp==2)
		line = line.rstrip()
	print(line)
	split_line = line.split("cycles:u")
	print(len(split_line))
	#print(split_line[0])
	fma_perf.append(split_line[0])
#closing text file
file1.close()
#print(fma_perf)
writer = csv.writer(file2)
writer.writerow(fma_perf)
