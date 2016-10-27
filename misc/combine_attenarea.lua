require 'nn'
local subcombarea = require 'misc.subcombarea'
local layer, parent = torch.class('nn.combine_attenarea','nn.Module')

function layer:__init()
    parent.__init(self)
    self.subcomb = subcombarea.subcombmainfunc()
end

function layer:getModulesList()
    return {self.subcomb}
end

function layer:parameters()
    local p1,g1 = self.subcomb:parameters()
    local params = {}
    for k,v in pairs(p1) do table.insert(params,v) end
    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params,v) end

    return params, grad_params
end

function layer:training()
    self.subcomb:training()
end

function layer:evaluate()
    self.subcomb:evaluate()
end

function layer:updateOutput(input)
    local imgfeat = input[1]
    local probs = input[2]
    local new_img_feat= self.subcomb:forward({imgfeat,probs})

    return {new_img_feat}
end

function layer:updateGradInput(input,gradOutput)
    local imgfeat = input[1]
    local probs = input[2]

    local d_imgfeat, d_probs = unpack(self.subcomb:backward({imgfeat, probs}, gradOutput[1])) --???
   
    self.gradInput = {d_imgfeat, d_probs}
    return self.gradInput
end





