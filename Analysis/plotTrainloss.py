import numpy as np  
import matplotlib
matplotlib.use('Agg')  
from matplotlib.pyplot import plot,savefig
 
import matplotlib.pyplot as plt


def readfile(inputfile):
	f=open(inputfile,'r')
	lines=f.readlines()
	f.close()

	losses=np.zeros(len(lines))
	losses=map(float, lines)
	#for ix, line in enumerate(lines):
	#	losses[ix]=float(line)

	return losses


if __name__ == "__main__":
	inputfile1="trainloss_with.txt"
	inputfile2="trainloss_without.txt"
	loss1 = readfile(inputfile1)
        loss2 = readfile(inputfile2)

	lens=min(len(loss1),len(loss2))
	x=[y*600 for y in range(lens)]

	plt.figure()
	y_loss1=loss1[0:lens]
	y_loss2=loss2[0:lens]
	plot(x,y_loss1,'b', label="with_english_attenmaps", linewidth=2)  
	plot(x,y_loss2,'r', label="without_english_attenmaps",linewidth=2)
	plt.legend(loc='upper left')
	plt.title('train_loss', fontsize = 16)
	savefig('train_loss.jpg')
