require 'nn'
local subcombarea = require 'misc.subcombarea'

local layer, parent = torch.class('nn.combine_attenarea','nn.Module')4
function layer:_init()
    parent._init(self)
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
    self.imgfeat = input[1]
    self.probs = input[2]

    self.new_img_feat= unpack(self.subcomb:forward({self.imgfeat,self.probs})

    return {self.new_img_feat}
end

function layer:updateGradInput(input,gradOutput)
    self.imgfeat = input[1]
    self.probs = input[2]

    local d_probs = unpack(self.subcomb:backforward({self.imgfeat,self.probs},gradOutput)) --???
   
    self.gradInput = d_probs 
    return self.gradInput






