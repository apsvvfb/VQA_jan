import json
import sys
import numpy as np
import glob
import h5py
import os

def genHdf5(names,outfile,pathTxt):
  attenImgs  = np.ones((len(names),196))/196
  attenprobs = np.zeros((len(names),196))
  for ix, imgname in enumerate(names):
     tid=os.path.basename(imgname).split('.')[0]

     files=glob.glob("%s/%s_*.txt" %(pathTxt,tid))
     attens=np.zeros((len(files),196))
     
     for fid,filename in enumerate(files):
           f=open(filename,'r')
           nums=f.read()
           f.close()
           atten=np.zeros(196)
           for i in range(0,196):
               atten[i]=nums.split(' ')[i]
           atten=(atten-min(atten))/(max(atten)-min(atten))
           attens[fid]=atten
     if len(files) ~= 0:
     	   attenprobs[ix]=np.sum(attens, axis=0)/len(files)
     else:
	   attenprobs[ix]=np.ones((1,196))/196
  final=np.hstack((attenImgs,attenprobs))
  with h5py.File(outfile, 'w') as hf:
     hf.create_dataset('areaprobs', data=final)

def main(params):
    jfile = json.load(open(params['input_json'], 'r'))
    
    pathTxt=params['pathAttenMaps']

    trainnames=jfile["unique_img_train"]
    testnames=jfile["uniuqe_img_test"]

    genHdf5(trainnames,params['out_train'],pathTxt)
    genHdf5(testnames, params['out_test'],pathTxt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default='../data/vqa_data_prepro.json', help='input json file contains imgnames for train and test')
    parser.add_argument('--output_train', default='../data/vqa_data_area_train.h5', help='Attention hdf5 file for train')
    parser.add_argument('--output_test', default='../data/vqa_data_area_test.h5', help='Attention hdf5 file for test')
    parser.add_argument('--pathAttenMaps',default='/home/c-nrong/VQA/draw/AttenMaps/', help='AttenMaps of English questions for each images) 

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)
