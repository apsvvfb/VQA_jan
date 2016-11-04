require 'os'
require 'nn'
num_b=3 --batchsize
num_n=196
num_v=512
local img=torch.Tensor(num_b,num_n,num_v)

local cnn = nn.Sequential()
                      :add(nn.View(num_v):setNumInputDims(2))
                      :add(nn.Linear(num_v, num_v))
                      :add(nn.View(-1, num_n, num_v))
                      :add(nn.Tanh())
                      :add(nn.Dropout(0.5))

local img_feat = cnn:forward(img)

--print(img)
p1=nn.View(num_v):setNumInputDims(2)(img)
--print(p1)
p2=nn.Linear(num_v, num_v)(p1)
--print(p2)
p3=nn.View(-1, num_n, num_v)(p2)
--print(p3)
p4=nn.Tanh()(p3)
--print(p4)
p5=nn.Dropout(0.5)(p4)
--print(p5)



local img_atten=torch.Tensor(num_b,num_n) -- batchsize x 196

local img_atten_dim = nn.View(1,-1):setNumInputDims(1)(img_atten) 
print(#img_atten_dim)
print(#img_feat) 
local img_atten_feat = nn.MM(false, false)({img_atten_dim, img_feat}) 
print(#img_atten_feat)
local img_atten_feat = nn.View(input_size_img):setNumInputDims(2)(img_atten_feat) 
print(#img_atten_feat)
