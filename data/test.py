import sys
import json 
import numpy as np

filename="vqa_raw_train.json"
jfile = json.load(open(filename, 'r'))
print(len(jfile))
allid=np.zeros(len(jfile))
for i in range(len(jfile)):
    image = jfile[i]['img_path']
    imgid =image.split('/')[1].split('.')[0]
    allid[i]=imgid
finids=np.unique(allid)
np.savetxt('trainid', finids, delimiter=',',fmt='%d')
'''
filename="/home/c-nrong/VQA/draw/Json/question_answers_genome.json"
jfile = json.load(open(filename, 'r'))
allid=np.zeros(len(jfile))
for i in range(len(jfile)):
    imgid = jfile[i]['id']
    allid[i]=imgid
finids=np.unique(allid)
np.savetxt('endid', finids, delimiter=',',fmt='%d')
'''
