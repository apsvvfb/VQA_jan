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
	inputfile1="eval_train_without_engAttenmaps_order.txt"
	inputfile4="eval_train_with_v2_onlyA_order.txt"
	inputfile5="eval_train_with_v2_addone_order.txt"
	iter1, loss1, acc1 = readfile(inputfile1)
        iter4, loss4, acc4 = readfile(inputfile4)
        iter5, loss5, acc5 = readfile(inputfile5)

	lens=min(len(iter1),len(iter5),len(iter4))

	x=iter1[0:lens]

	plt.figure()
	y_loss1=loss1[0:lens]
	y_loss4=loss5[0:lens]
	y_loss5=loss5[0:lens]

	plot(x,y_loss1,'r', label=inputfile1, linewidth=2)  
        plot(x,y_loss4,'c', label=inputfile4,linewidth=2)
        plot(x,y_loss5,'m', label=inputfile5,linewidth=2)

	plt.legend(loc='lower right')
	plt.title('eval_loss_order', fontsize = 16)
	savefig('eval_loss_order.jpg')
	
	plt.figure()
        y_acc1=acc1[0:lens]
        y_acc4=acc4[0:lens]
        y_acc5=acc5[0:lens]
        plot(x,y_acc1,'ro', label=inputfile1, linewidth=2)
        plot(x,y_acc4,'c*', label=inputfile4,linewidth=2)
        plot(x,y_acc5,'m', label=inputfile5,linewidth=2)

        plt.legend(loc='lower right')
        plt.title('eval_accuracy_order', fontsize = 16)
        savefig('eval_accuracy_order.jpg')

