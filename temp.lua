require 'os'
file = io.open("tesaat.txt", "w")
var="abc"
file:write('wrote json checkpoint to ' .. var .. '.json', "\n")
file:write(string.format('iter %s',var), "\n")
file:write("hello", "\n")
file:write("hello", "\n")
file:write("hello", "\n")
io.close(file)
os.exit()



require 'nn'

    imgfeat = torch.Tensor(3,196,512)  -- batchsize x  196             x 512
print(#imgfeat)
    probs = torch.Tensor(3,196*2,1)      -- batchsize x [196 x (1 + 1)]
    local input_size = 196*(1+1)
    local probs0 = nn.Reshape(input_size,true)(probs)
--print(#probs0)
    local probs1 = nn.Linear(input_size, 196)(probs0)
--print(#probs1)
    local probs2 = nn.View(1,-1):setNumInputDims(1)(probs1)
    local probs3 = nn.Transpose({2, 3})(probs2)
    local probs4 = nn.Replicate(512,2,2)(probs3):contiguous()
--print(#probs4)
    local probs5 = nn.View(-1,196,512)(probs4)
    local probs5=  nn.Reshape(196,512,true)(probs4) --nn.Reshape(batchsize,196,512)(probs4)
--print(#probs5)
    local new_imgfeat = nn.CMulTable()({probs5,imgfeat})
print(#new_imgfeat)
