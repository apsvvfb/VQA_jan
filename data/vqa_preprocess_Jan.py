import sys
import json 
import numpy as np
filename="/home/c-nrong/VQA/draw/Json/question_answers_jan.json"
trainIDs="../genIDs/trainID.txt"
testIDs="../genIDs/testID.txt"

jfile = json.load(open(filename, 'r'))
imdir='%s/%s.jpg'
outtrain=[]
outtest=[]

train="VG_100K"
test="VG_100K_2"

f=open(trainIDs,'r')
trainline=f.readlines()
f.close()
trainlines=map(str.strip, trainline)

f=open(testIDs,'r')
testline=f.readlines()
f.close()
testlines=map(str.strip, testline)

maxNum=8
trainNum={}
testNum={}
for i in range(len(jfile)):
  print("%d:%d" % (len(jfile),i))
  image_id = str(jfile[i]['id'])
  qas=jfile[i]['qas']
  for qid in range(len(qas)):
    ans = qas[qid]['answer'] 
    question = qas[qid]['question']
    question_id = qas[qid]['qa_id']
    if image_id in trainlines:
  	trainNum[image_id]=trainNum.get(image_id,0)+1
	if trainNum[image_id] <= maxNum:
	    image_path = imdir%(train, image_id) 
	    outtrain.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})
    elif image_id in testlines:
        testNum[image_id]=testNum.get(image_id,0)+1
	if testNum[image_id] <= maxNum:
	    image_path = imdir%(test, image_id) 
	    outtest.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'ans': ans})
    else:
	print("Error: %s is neither train nor test id." % image_id )
	#sys.exit()
print 'Training sample %d, Testing sample %d...' %(len(outtrain), len(outtest)) 
 
json.dump(outtrain, open('vqa_raw_train.json', 'w')) 
json.dump(outtest, open('vqa_raw_test.json', 'w')) 

