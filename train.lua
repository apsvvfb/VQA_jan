------------------------------------------------------------------------------
--  Hierarchical Question-Image Co-Attention for Visual Question Answering
--  J. Lu, J. Yang, D. Batra, and D. Parikh
--  https://arxiv.org/abs/1606.00061, 2016
--  if you have any question about the code, please contact jiasenlu@vt.edu
-----------------------------------------------------------------------------

require 'nn'
require 'torch'
require 'optim'
require 'misc.DataLoaderDisk'
require 'misc.combine_attenarea'
require 'misc.word_level'
require 'misc.phrase_level'
require 'misc.ques_level'
require 'misc.recursive_atten'
require 'misc.optim_updates'
local utils = require 'misc.utils'
require 'xlua'
require 'os'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_train_h5','data/vqa_data_img_vgg_train.h5','path to the h5file containing the image feature')
cmd:option('-input_img_test_h5','data/vqa_data_img_vgg_test.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data/vqa_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/vqa_data_prepro.json','path to the json file containing additional info and vocab')

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-co_atten_type', 'Alternating', 'co_attention type. Parallel or Alternating, alternating trains more faster than parallel.')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')


cmd:option('-hidden_size',512,'the hidden layer size of the model.')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-batch_size',20,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-output_size', 1000, 'number of output answers')
cmd:option('-rnn_layers',2,'number of the rnn layer')


-- Optimization
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 300, 'every how many epoch thereafter to drop LR by 0.1?')
cmd:option('-optim_alpha',0.99,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.995,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', 210000, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-iterPerEpoch', 1200)

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every',6000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'save/train_vgg', 'folder to save checkpoints into (empty = this folder)')

-- Visualization
cmd:option('-losses_log_every', 600, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-id', '0', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-gpuid', 6, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

-- c-nrong: combine areas
cmd:option('-use_english_attenmaps', 1, 'use attenmaps generated from English questions or not. 1 means yes, while 0 means no')
cmd:option('-input_area_train_h5','data/vqa_data_area_train.h5',' hdf5 file which stores the attenmaps for train images')
cmd:option('-input_area_test_h5', 'data/vqa_data_area_test.h5','a hdf5 file which stores the attenmaps for test images')
cmd:option('-output_eval_h5','data/eval_prediction.h5','a hdf5 file which stores the result for each sample in evaluation set. max_iters/save_checkpoint_every * totalnum_evalset') 
cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
  require 'cudnn' 
  end
  --cutorch.manualSeed(opt.seed)
  --cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader = DataLoader{h5_img_file_train = opt.input_img_train_h5, h5_img_file_test = opt.input_img_test_h5, h5_ques_file = opt.input_ques_h5, json_file = opt.input_json, feature_type = opt.feature_type, h5_area_file_train=opt.input_area_train_h5, h5_area_file_test=opt.input_area_test_h5, ifengatten=opt.use_english_attenmaps}
------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
print('Building the model...')
-- intialize language model
local loaded_checkpoint
local lmOpt
if string.len(opt.start_from) > 0 then
  local start_path = path.join(opt.checkpoint_path .. '_' .. opt.co_atten_type ,  opt.start_from)
  loaded_checkpoint = torch.load(start_path)
  lmOpt = loaded_checkpoint.lmOpt
else
  lmOpt = {}
  lmOpt.vocab_size = loader:getVocabSize()
  lmOpt.hidden_size = opt.hidden_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = opt.rnn_layers
  lmOpt.dropout = 0.5
  lmOpt.seq_length = loader:getSeqLength()
  lmOpt.batch_size = opt.batch_size
  lmOpt.output_size = opt.rnn_size
  lmOpt.atten_type = opt.co_atten_type
  lmOpt.feature_type = opt.feature_type
end
if opt.use_english_attenmaps == 1 then
  protos.combarea = nn.combine_attenarea()
end
protos.word = nn.word_level(lmOpt)
protos.phrase = nn.phrase_level(lmOpt)
protos.ques = nn.ques_level(lmOpt)

protos.atten = nn.recursive_atten()
protos.crit = nn.CrossEntropyCriterion()
-- ship everything to GPU, maybe

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

if opt.use_english_attenmaps == 1 then
  cparams, grad_cparams = protos.combarea:getParameters()
end
local wparams, grad_wparams = protos.word:getParameters()
local pparams, grad_pparams = protos.phrase:getParameters()
local qparams, grad_qparams = protos.ques:getParameters()
local aparams, grad_aparams = protos.atten:getParameters()


if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  if opt.use_english_attenmaps == 1 then
    cparams:copy(loaded_checkpoint.cparams)
  end
  wparams:copy(loaded_checkpoint.wparams)
  pparams:copy(loaded_checkpoint.pparams)
  qparams:copy(loaded_checkpoint.qparams)
  aparams:copy(loaded_checkpoint.aparams)
end

if opt.use_english_attenmaps == 1 then
  print('total number of parameters in combine_attenarea: ', cparams:nElement())
  assert(cparams:nElement() == grad_cparams:nElement())
end

print('total number of parameters in word_level: ', wparams:nElement())
assert(wparams:nElement() == grad_wparams:nElement())

print('total number of parameters in phrase_level: ', pparams:nElement())
assert(pparams:nElement() == grad_pparams:nElement())

print('total number of parameters in ques_level: ', qparams:nElement())
assert(qparams:nElement() == grad_qparams:nElement())
protos.ques:shareClones()

print('total number of parameters in recursive_attention: ', aparams:nElement())
assert(aparams:nElement() == grad_aparams:nElement())

collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split)
  if opt.use_english_attenmaps == 1 then
    protos.combarea:evaluate()
  end
  protos.word:evaluate()
  protos.phrase:evaluate()
  protos.ques:evaluate()
  protos.atten:evaluate()
  loader:resetIterator(split)

  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local right_sum = 0
  local predictions = {}
  local total_num = loader:getDataNum(split)

  local ni=0
  --local predres=torch.zeros(total_num)
  local predres=torch.zeros(317584) -- total_num + batchsize(=20)
  while true do
    local data = loader:getBatch{batch_size = opt.batch_size, split = split}
    -- ship the data to cuda
    if opt.gpuid >= 0 then
      data.answer = data.answer:cuda()
      data.images = data.images:cuda()
      if opt.use_english_attenmaps == 1 then
        data.attenprobs = data.attenprobs:cuda()
      end
      data.questions = data.questions:cuda()
      data.ques_len = data.ques_len:cuda()
    end
  n = n + data.images:size(1)
  xlua.progress(n, total_num)
  if opt.use_english_attenmaps == 1 then 
    new_data_images = unpack(protos.combarea:forward({data.images,data.attenprobs}))
    word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({data.questions, new_data_images}))
  else
    word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({data.questions, data.images}))
  end

  local conv_feat, p_ques, p_img = unpack(protos.phrase:forward({word_feat, data.ques_len, img_feat, mask}))

  local q_ques, q_img = unpack(protos.ques:forward({conv_feat, data.ques_len, img_feat, mask}))

  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img}
  local out_feat = protos.atten:forward(feature_ensemble)

  -- forward the language model criterion
  local loss = protos.crit:forward(out_feat, data.answer)

    local tmp,pred=torch.max(out_feat,2)

    for i = 1, pred:size()[1] do
      ni=ni+1
      if pred[i][1] == data.answer[i] then
        right_sum = right_sum + 1
	predres[ni]=1
      end
    end

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
    if n >= total_num then break end
  end

  return loss_sum/loss_evals, right_sum / total_num, predres
end


-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun()
  if opt.use_english_attenmaps == 1 then
    protos.combarea:training()
    grad_cparams:zero()  
  end

  protos.word:training()
  grad_wparams:zero()  

  protos.phrase:training()
  grad_pparams:zero()

  protos.ques:training()
  grad_qparams:zero()

  protos.atten:training()
  grad_aparams:zero()

  ----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local data = loader:getBatch{batch_size = opt.batch_size, split = 0}
  if opt.gpuid >= 0 then
    data.answer = data.answer:cuda()
    data.questions = data.questions:cuda()
    data.ques_len = data.ques_len:cuda()
    data.images = data.images:cuda()
    if opt.use_english_attenmaps == 1 then 
	data.attenprobs = data.attenprobs:cuda()
    end
  end

  if opt.use_english_attenmaps == 1 then
    new_data_images = unpack(protos.combarea:forward({data.images,data.attenprobs}))
    word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({data.questions, new_data_images}))
  else
    word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({data.questions, data.images}))
  end

  local conv_feat, p_ques, p_img = unpack(protos.phrase:forward({word_feat, data.ques_len, img_feat, mask}))

  local q_ques, q_img = unpack(protos.ques:forward({conv_feat, data.ques_len, img_feat, mask}))

  local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img}
  local out_feat = protos.atten:forward(feature_ensemble)
  
  -- forward the language model criterion
  local loss = protos.crit:forward(out_feat, data.answer)
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(out_feat, data.answer)
  
  local d_w_ques, d_w_img, d_p_ques, d_p_img, d_q_ques, d_q_img = unpack(protos.atten:backward(feature_ensemble, dlogprobs))

  local d_ques_feat, d_ques_img = unpack(protos.ques:backward({conv_feat, data.ques_len, img_feat}, {d_q_ques, d_q_img}))
    
  --local d_ques1 = protos.bl1:backward({ques_feat_0, data.ques_len}, d_ques2)
  local d_conv_feat, d_conv_img = unpack(protos.phrase:backward({word_feat, data.ques_len, img_feat}, {d_ques_feat, d_p_ques, d_p_img}))

  if opt.use_english_attenmaps == 1 then  
    d_new_image = protos.word:backward({data.questions, new_data_images}, {d_conv_feat, d_w_ques, d_w_img, d_conv_img, d_ques_img})
    d_prob = protos.combarea:backward({data.images,data.attenprobs}, {d_new_image})
  else
    dummy = protos.word:backward({data.questions,data.images}, {d_conv_feat, d_w_ques, d_w_img, d_conv_img, d_ques_img})
  end
  -----------------------------------------------------------------------------
  -- and lets get out!
  local stats = {}
  stats.dt = dt
  local losses = {}
  losses.total_loss = loss
  return losses, stats
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------

local loss0
local c_optim_state = {}
local w_optim_state = {}
local p_optim_state = {}
local q_optim_state = {}
local a_optim_state = {}
local loss_history = {}
local accuracy_history = {}
local learning_rate_history = {}
local best_val_loss = 10000
local ave_loss = 0
local timer = torch.Timer()
local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
local learning_rate = opt.learning_rate
-- create the path to save the model.
paths.mkdir(opt.checkpoint_path .. '_' .. opt.co_atten_type)

outfilena = io.open("train_with_engAttenmaps.txt", "w")
evali=0
evalnum=317564+opt.batch_size
outpred=torch.Tensor(torch.ceil(opt.max_iters/opt.save_checkpoint_every),evalnum)

while true do
  -- eval loss/gradient
  local losses, stats = lossFun()
  ave_loss = ave_loss + losses.total_loss
  -- decay the learning rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  if iter % opt.losses_log_every == 0 then
    ave_loss = ave_loss / opt.losses_log_every
    loss_history[iter] = losses.total_loss 
    accuracy_history[iter] = ave_loss
    learning_rate_history[iter] = learning_rate

    print(string.format('iter %d: %f, %f, %f, %f', iter, losses.total_loss, ave_loss, learning_rate, timer:time().real))
    outfilena:write(string.format('iter %d: %f, %f, %f, %f', iter, losses.total_loss, ave_loss, learning_rate, timer:time().real), "\n")
    ave_loss = 0
  end

  -- save checkpoint once in a while (or on final iteration)
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
      local val_loss, val_accu, preds = eval_split(2)
      evali=evali+1
      outpred[evali]=preds

      print('validation loss: ', val_loss, 'accuracy ', val_accu)
      outfilena:write('validation loss: ', val_loss, 'accuracy ', val_accu, "\n")

      local checkpoint_path = path.join(opt.checkpoint_path .. '_' .. opt.co_atten_type, 'model_id' .. opt.id .. '_iter'.. iter)
      torch.save(checkpoint_path..'.t7', {cparams=cparams,wparams=wparams, pparams = pparams, qparams=qparams, aparams=aparams, lmOpt=lmOpt}) 

      local checkpoint = {}
      checkpoint.opt = opt
      checkpoint.iter = iter
      checkpoint.loss_history = loss_history
      checkpoint.accuracy_history = accuracy_history
      checkpoint.learning_rate_history = learning_rate_history

      local checkpoint_path = path.join(opt.checkpoint_path .. '_' .. opt.co_atten_type, 'checkpoint' .. '.json')

      utils.write_json(checkpoint_path, checkpoint)
      print('wrote json checkpoint to ' .. checkpoint_path .. '.json')
      outfilena:write('wrote json checkpoint to ' .. checkpoint_path .. '.json', "\n")
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    if opt.use_english_attenmaps == 1 then
      rmsprop(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
    end
    rmsprop(wparams, grad_wparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, w_optim_state)
    rmsprop(pparams, grad_pparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, p_optim_state)
    rmsprop(qparams, grad_qparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, q_optim_state)
    rmsprop(aparams, grad_aparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, a_optim_state)
  else
    error('bad option opt.optim')
  end

  iter = iter + 1
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
end

io.close(outfilena)
local eval_h5_file = hdf5.open(opt.output_eval_h5, 'w') 
eval_h5_file:write('/predictions', outpred) 
eval_h5_file:close() 

