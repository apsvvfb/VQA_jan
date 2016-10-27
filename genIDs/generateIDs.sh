#!/bin/bash
trainImg=/home/c-nrong/VQA/VG_100K/
testImg=/home/c-nrong/VQA/VG_100K_2/
allID=allIDs.txt
trainID=trainID.txt
testID=testID.txt

#python genALLids.py $allIDs
#sort $allIDs > ${allIDs}.temp

function generate {
  outfile=$1
  imgpath=$2
  allID=$3

  ls -1 $imgpath | cut -d'.' -f1 | sort > ${outfile}.temp
  comm ${allID} ${outfile}.temp -1 -2 > ${outfile}.temp2
  while read id
  do
	grayorrgb=`identify "${imgpath}/${id}.jpg" | cut -d' ' -f6`
	if [ $grayorrgb == "sRGB" ];then
		echo $id >> ${outfile}.temp3
	fi
  done < ${outfile}.temp2

  sort -n ${outfile}.temp3 > $outfile

  rm ${outfile}.temp ${outfile}.temp2 ${outfile}.temp3
}

generate $trainID  $trainImg ${allIDs}.temp
generate $testID   $testImg  ${allIDs}.temp

#rm ${allIDs}.temp

