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
	inputfile4="trainloss_train_with_v2_onlyA.txt"
        inputfile5="trainloss_train_with_v2_addone.txt"

	loss1 = readfile(inputfile1)
        loss2 = readfile(inputfile2)
	loss3 = readfile(inputfile3)
	loss4 = readfile(inputfile4)
	loss5 = readfile(inputfile5)
	lens=min(len(loss1),len(loss2),len(loss3),len(loss4),len(loss5))
	x=[y*600 for y in range(lens)]

	plt.figure()
	y_loss1=loss1[0:lens]
	y_loss2=loss2[0:lens]
	y_loss3=loss3[0:lens]
	y_loss4=loss4[0:lens]
	y_loss5=loss5[0:lens]
	plot(x,y_loss1,'r', label=inputfile1, linewidth=2)  
	plot(x,y_loss2,'b', label=inputfile2, linewidth=2)
	plot(x,y_loss3,'g', label=inputfile3, linewidth=2)
        plot(x,y_loss3,'c', label=inputfile4, linewidth=2)
        plot(x,y_loss3,'m', label=inputfile5, linewidth=2)
	plt.legend(loc='lower right')
	plt.title('train_loss', fontsize = 16)
	savefig('train_loss.jpg')
