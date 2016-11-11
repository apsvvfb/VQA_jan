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

        y_acc1=acc1[0:lens]
        y_acc4=acc4[0:lens]
        y_acc5=acc5[0:lens]
      
	
	for i in range(len(x)):
		print "iter: %d" %(x[i])
		print "without:%f, onlyA: %f, diff:%f" %(y_acc1[i],y_acc4[i],y_acc4[i]-y_acc1[i])
                print "without:%f, addone:%f, diff:%f\n" %(y_acc1[i],y_acc5[i],y_acc5[i]-y_acc1[i])
	mean1=sum(y_acc1)/len(x)
	mean4=sum(y_acc4)/len(x)
	mean5=sum(y_acc5)/len(x)
	print "iter:from %d to %d. save results every 6000 iters. -> %d " %(x[0],x[lens-1],lens)
	print "mean: %f, %f, %f" %(mean1,mean4,mean5)
	print "mean-diff: %f-%f=%f, %f-%f=%f" %(mean4,mean1,mean4-mean1,mean5,mean1,mean5-mean1)

 
