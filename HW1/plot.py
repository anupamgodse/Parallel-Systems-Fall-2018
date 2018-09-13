"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt


opacity = 1

pair1 = []
pair2 = []
pair3 = []
pair4 = []
sdpair1 = []
sdpair2 = []
sdpair3 = []
sdpair4 = []


f = open("output.txt", "r");
i = 0;

xlabels = []

excludeLines = [0];
for lineNo, line in enumerate(f):
	if(lineNo in excludeLines):
		continue;
	
	temp = line.split();
	print(temp);
		
	if((lineNo+3) % 4 == 0):
		xlabels.append(temp[0]);
		#print("temp1 =" + temp[2]);
		pair1.append(float(temp[2]));
		sdpair1.append(float(temp[3]));
	elif((lineNo+3) % 4 == 1):
		pair2.append(float(temp[2]));
		sdpair2.append(float(temp[3]));
	elif((lineNo+3) % 4 == 2):
		pair3.append(float(temp[2]));
		sdpair3.append(float(temp[3]));
	else:
		pair4.append(float(temp[2]));
		sdpair4.append(float(temp[3]));
	

N = 17

ind = np.arange(N)  # the x locations for the groups
width = 0.17      # the width of the bars

fig, ax = plt.subplots()
#print(pair1)
rects1 = ax.bar(ind, pair1, width, edgecolor='red', color = 'white', yerr=sdpair1, 
		alpha = opacity, ecolor = 'red', capsize = 5, linewidth = 1)

rects2 = ax.bar(ind + width, pair2, width, edgecolor='violet', color = 'white', 
		yerr=sdpair2, alpha = opacity, ecolor='violet', capsize = 5, linewidth = 1)

rects3 = ax.bar(ind + 2 * width, pair3, width, edgecolor='blue', color = 'white', 
		yerr=sdpair3, alpha = opacity, ecolor='blue', capsize = 5, linewidth = 1)

rects4 = ax.bar(ind + 3 * width, pair4, width, edgecolor='green', color = 'white', 
		yerr=sdpair4, alpha = opacity, ecolor='green', capsize = 5, linewidth = 1)


# add some text for labels, title and axes ticks
ax.set_xlabel('Message Size(B)')
ax.set_ylabel('Average Round Trip Time(s)')
ax.set_title('Average round trip time for varying sizes')
ax.set_xticks(ind + 3 * width / 2)
ax.set_xticklabels(xlabels)

plt.xticks(fontsize=8)

#plt.rcParams.update({'font.size': 22})

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0],), ('pair1', 'pair2', 'pair3', 'pair4'));

plt.savefig('output.png')

plt.show()


