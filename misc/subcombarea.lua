require 'nngraph'
require 'nn'
local subcombarea = {}

function subcombarea.subcombmainfunc()
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())

    local imgfeat = inputs[1]   -- batchsize x  196             x 512
    local probs = inputs[2]     -- batchsize x [196 x (1 + 1)]  x 1

    local input_size = 196*(1+1)
    --local probs0 = nn.Reshape(input_size,true)(probs)
    --local probs1 = nn.Linear(input_size, 196)(probs0)
    --local probs2 = nn.View(1,-1):setNumInputDims(1)(probs1)
    --local probs3 = nn.Transpose({2, 3})(probs2)
    ----local probs4 = nn.Replicate(512,2,2)(probs3): contiguous()
    ----local probs5 = nn.View(-1,196,512)(probs4)
    local probs4 = nn.Replicate(512,2,2)(probs) --(probs3)
    local probs5=  nn.Reshape(196,512,true)(probs4) --nn.Reshape(batchsize,196,512)(probs4)
    local new_imgfeat = nn.CMulTable()({probs5,imgfeat})

    table.insert(outputs, new_imgfeat)
    
    return nn.gModule(inputs,outputs)
end

return subcombarea

