import argparse
import json
import sys
import numpy as np
import glob
import h5py
import os

def getIdx(imgidxs,itoimg):
  idxs = {}
  for i, imgidx in enumerate(imgidxs): 
     img=itoimg.get(str(int(imgidx)))
     img = img.encode('punycode') #unicode -> ascii
     imgid=img.split('/')[1].split('.')[0]
     if imgid not in idxs: 
         idxs[imgid] = [] 
     idxs[imgid].append(i)
  return idxs

def genHdf5(names,outfile,attenmaps,idxs):
  #attenImgs  = np.ones((len(names),196))/196	#v1_all_1dividedby196
  attenImgs  = np.ones((len(names),196))	#v1_all_1
  attenprobs = np.zeros((len(names),196))
  for ix, imgname in enumerate(names):
     imgid=os.path.basename(imgname).split('.')[0]
     ques_list=idxs.get(imgid,0)
     if ques_list != 0 :
	ques_len = len(ques_list)
        attentemp = np.zeros((ques_len,196))
        for j in range(ques_len):
            atten = attenmaps[ques_list[j]]
            atten=(atten-min(atten))/(max(atten)-min(atten))
            attentemp[j]=atten
        attenprobs[ix]=np.sum(attentemp, axis=0)/ques_len    
     else:
	attenprobs[ix]=np.ones((1,196))		#v1_all_1
	#attenprobs[ix]=np.ones((1,196))/196 	#v1_all_1dividedby196
  #final=np.hstack((attenImgs,attenprobs)) #v1_*
  final=attenprobs   #v2_onlyA
  #final=attenprobs+1 #v2_addone
  with h5py.File(outfile, 'w') as hf:
     hf.create_dataset('areaprobs', data=final)
  
def main(params):
    jfile = json.load(open(params['input_json'], 'r'))
    names_train=jfile["unique_img_train"]
    names_test =jfile["uniuqe_img_test"]

    jfile2 = json.load(open(params['input_json_atten']))
    itoimg_test=jfile2["ix_to_img_test"] #unicode

    with h5py.File(params['input_h5'],'r') as hf:
        data1 = hf.get('attenmaps')
        attenmaps = np.array(data1)
	data2 = hf.get('imgidx')
	imgidx = np.array(data2)
	
    Idx = getIdx(imgidx,itoimg_test)

    genHdf5(names_train,params['output_train'],attenmaps,Idx)
    genHdf5(names_test, params['output_test'], attenmaps,Idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='../data/vqa_data_prepro.json', help='input json file contains imgnames for train and test')
    parser.add_argument('--input_json_atten', default='/home/c-nrong/VQA/HieCoAttenVQA2/data/vqa_data_prepro.json', help='json for attenmaps. ix_to_img_test')
    parser.add_argument('--output_train', default='../data/vqa_data_area_train.h5', help='Attention hdf5 file for train')
    parser.add_argument('--output_test', default='../data/vqa_data_area_test.h5', help='Attention hdf5 file for test')
    parser.add_argument('--input_h5', default='/home/c-nrong/VQA/HieCoAttenVQA2/AttenmapsAndImgidx.h5', help='AttenMaps and imgidx') 

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)
