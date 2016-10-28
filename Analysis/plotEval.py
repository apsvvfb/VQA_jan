import numpy as np  
import matplotlib
matplotlib.use('Agg')  
from matplotlib.pyplot import plot,savefig
 
import matplotlib.pyplot as plt


def readfile(inputfile):
	f=open(inputfile,'r')
	lines=f.readlines()
	f.close()

	iters=np.zeros(len(lines))
	losses=np.zeros(len(lines))
	accs=np.zeros(len(lines))

	for ix, line in enumerate(lines):
		iters[ix]=int(line.split(' ')[0])
		losses[ix]=float(line.split(' ')[1])
		accs[ix]=float(line.split(' ')[2])

	return iters,losses,accs


if __name__ == "__main__":
	inputfile1="eval_with.res"
	inputfile2="eval_without.res"
	iter1, loss1, acc1 = readfile(inputfile1)
        iter2, loss2, acc2 = readfile(inputfile2)

	lens=min(len(iter1),len(iter2))
	x=iter1[0:lens]

	plt.figure()
	y_loss1=loss1[0:lens]
	y_loss2=loss2[0:lens]
	plot(x,y_loss1,'b', label="with_english_attenmaps", linewidth=2)  
	plot(x,y_loss2,'r', label="without_english_attenmaps",linewidth=2)
	plt.legend(loc='upper left')
	plt.title('eval_loss', fontsize = 16)
	savefig('eval_loss.jpg')
	
	plt.figure()
        y_acc1=acc1[0:lens]
        y_acc2=acc2[0:lens]
        plot(x,y_acc1,'b', label="with_english_attenmaps", linewidth=2)
        plot(x,y_acc2,'r', label="without_english_attenmaps",linewidth=2)
        plt.legend(loc='upper left')
        plt.title('eval_accuracy', fontsize = 16)
        savefig('eval_accuracy.jpg')

