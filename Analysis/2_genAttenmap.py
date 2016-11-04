import sys
import json 
import numpy as np
file_json="../data/vqa_data_prepro.json"
file_h5="../data/vqa_data_prepro.h5"
'''
file_atten="../data/vqa_data_area_test.h5"
#without attenmpas
file_pred1="../data/eval_prediction.h5"  
#with attenmaps
file_pred2="../data/eval_prediction.h5"  
'''
jfile = json.load(open(file_json, 'r'))
unique_img=jfile["uniuqe_img_test"] 
print(unique_img[0])
sys.exit()

with h5py.File(file_h5,'r') as hf:
	data1 = hf.get('img_pos_test')
	img_pos  = np.array(data1)
'''
with h5py.File(file_pred1,'r') as hf:
        data1 = hf.get('predictions')
        preds1  = np.array(data1)
with h5py.File(file_pred2,'r') as hf:
        data1 = hf.get('predictions')
        preds2  = np.array(data1)
'''
_iter=1   #0<=iteri<len(pred)
pred1=preds1[_iter]    # 1 0 0 1 ...
pred2=preds2[_iter]    # 1 1 0 0 ...
for i in range(pred1):
	idx=img_pos[i]
	idx=idx-1
	attenmap=attenmaps[idx]
	imgname=unique_img[idx]
	if pred1[i]==1 and pred2[i]==0: #attenmap fails
		print 1
	elif pred1[i]==0 and pred2[i]==1: #attenmap works
		print 2	




