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
	inputfile1="trainloss_train_without_engAttenmaps.txt"
	inputfile2="trainloss_train_with_v1_1dividedby196.txt"
        inputfile3="trainloss_train_with_v1_1.txt"
	loss1 = readfile(inputfile1)
        loss2 = readfile(inputfile2)
	loss3 = readfile(inputfile3)

	lens=min(len(loss1),len(loss2),len(loss3))
	x=[y*600 for y in range(lens)]

	plt.figure()
	y_loss1=loss1[0:lens]
	y_loss2=loss2[0:lens]
	y_loss3=loss3[0:lens]
	plot(x,y_loss1,'r', label=inputfile1, linewidth=2)  
	plot(x,y_loss2,'b', label=inputfile2, linewidth=2)
	plot(x,y_loss3,'g', label=inputfile3, linewidth=2)
	plt.legend(loc='lower right')
	plt.title('train_loss', fontsize = 16)
	savefig('train_loss.jpg')
