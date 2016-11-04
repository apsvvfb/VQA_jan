#!/bin/bash

for inputfile in `ls train_with*.txt`;do
#inputfile="train_with_engAttenmaps.txt"
#inputfile="train_without_engAttenmaps.txt"

outfile1="trainloss_$inputfile"
outfile2="eval_$inputfile"
if [ -r $outfile1 ];then rm $outfile1;fi
if [ -r $outfile2 ];then rm $outfile2;fi

interval=12
num=0
del="accuracy"
while read line 
do
	num=$[num+1]
	if [ $[(num-1)%12] -eq 0 ]; then
		iternum=`echo $line | cut -d':' -f1 | cut -d' ' -f2`
		train_loss=`echo $line | cut -d' ' -f4 | cut -d',' -f1`
		echo $train_loss >> $outfile1
		continue
	elif [ $[(num-2)%12] -eq 0 ]; then
		val_loss=`echo $line | cut -d' ' -f3`
		val_loss=${val_loss%${del}}
		val_acc=`echo $line | cut -d' ' -f4`
		echo $iternum $val_loss $val_acc >> $outfile2
	elif [ $[(num-3)%12] -eq 0 ]; then
		continue
	else
		train_loss=`echo $line | cut -d' ' -f4 | cut -d',' -f1`	
		echo $train_loss >> $outfile1
	fi
done < $inputfile

done


