require 'os'
require 'nn'

    imgfeat = torch.Tensor(3,2,5)  -- batchsize x  196             x 512
    probs = torch.Tensor(3,4,1)    -- batchsize x [196 x (1 + 1)]
    local input_size = 4
    local probs0 = nn.Reshape(input_size,true)(probs)
--print(probs0)
    local probs1 = nn.Linear(input_size, 2)(probs0)
--print(probs1)
    local probs2 = nn.View(1,-1):setNumInputDims(1)(probs1)
print(probs2)
    local probs3 = nn.Transpose({2, 3})(probs2)
print(probs3)
    local probs4 = nn.Replicate(5,2,2)(probs3)
print(probs4)
    local probs5=  nn.Reshape(2,5,true)(probs4) --nn.Reshape(batchsize,196,512)(probs4)
print(probs5)
    local new_imgfeat = nn.CMulTable()({probs5,imgfeat})
--print(#new_imgfeat)
