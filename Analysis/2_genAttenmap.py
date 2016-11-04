import os
import sys
import json 
import numpy as np
from scipy import misc
from PIL import Image

def draw(imgpath,atten,outpath):
  #1.generate Attention Map
  img=np.ones((448,448))
  for i in range(0,196):
	atten[i]=nums.split(' ')[i]
  atten=(atten-min(atten))/(max(atten)-min(atten))
  atten=np.floor(atten*255)
  #atten=atten.reshape(14,14)
  #print max(atten),min(atten)
  for i in range(0,14):
	r=i*32
	for j in range(0,14):
	   c=j*32
	   mul=atten[i*14+j]
	   for ri in range(r,r+31):
		for cj in range(c,c+31):
		    img[ri][cj]=img[ri][cj]*mul
  misc.toimage(img, cmin=0.0, cmax=255.0).save('atten.png')
  #sys.exit()
  #2.reshape the original image
  img = Image.open(imgpath) # image extension *.png,*.jpg
  new_width  = 448
  new_height = 448
  img = img.resize((new_width, new_height), Image.ANTIALIAS)
  img.save('test448.jpg')
  #3.overlay two images
  background = Image.open("test448.jpg")
  overlay = Image.open("atten.png")

  background = background.convert("RGBA")
  overlay = overlay.convert("RGBA")

  new_img = Image.blend(background, overlay, 0.7)
  outfile="%s/%s.png" % (outpath,os.path.basename(filename).split('.')[0])
  new_img.save(outfile,"PNG")

  os.remove("test448.jpg")

if __name__ == "__main__":

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
  #print(unique_img[0])

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
	imgpath="/home/c-nrong/VQA/VG_100K_2/"+imgname
	outpath="drawres/"
	if pred1[i]==1 and pred2[i]==0: #attenmap fails
		draw(imgpath,attenmap,outpath)
	elif pred1[i]==0 and pred2[i]==1: #attenmap works
		print 2	




