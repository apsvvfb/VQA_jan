local cjson = require 'cjson'
filename="/home/c-nrong/VQA/HieCoAttenVQA2/data/vqa_data_prepro.json"
local file = io.open(filename, 'r') 
local text = file:read() 
file:close() 
local json_file = cjson.decode(text) 
word = json_file.ix_to_img_test
print(word)
