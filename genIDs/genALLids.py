import json 
import sys
import numpy as np
outfile=sys.argv[1]
filename="/home/c-nrong/VQA/draw/Json/question_answers_jan.json"
train = json.load(open(filename, 'r'))
#print(len(train))
ids=np.zeros(len(train))
for i in range(len(train)):
    imgID=train[i]["id"]
    ids[i]=imgID
finids=np.unique(ids)
np.savetxt(outfile, finids, delimiter='\n',fmt='%d')




