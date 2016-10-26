import sys
import json

filename="/home/c-nrong/VQA/HieCoAttenVQA2/data/vqa_data_prepro.json"
data=json.load(open(filename))
i=data["ix_to_img_test"]
print(i.get("91139"))
print(i.get(91139))
for w,n in i.iteritems():
	#print w
	#print n
	sys.exit()
