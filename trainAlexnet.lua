require 'sys'
require 'cunn'
require 'cudnn'
require 'optim'

local nets = {}
nets[#nets+1] = require 'alexnet'

local libs = {}
libs[#libs+1] = {cudnn.SpatialConvolution, cudnn.SpatialMaxPooling, cudnn.ReLU, 'BDHW', 'cudnn'}

print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

function makeInput(config, size)
   local layout = config[4]
   local osize
   if layout == 'BDHW' then
      osize = size
   elseif layout == 'DHWB' then
      osize = {size[2],size[3],size[4],size[1]}
   elseif layout == 'BHWD' then
      osize = {size[1], size[3], size[4], size[2]}
   end
   return torch.randn(torch.LongStorage(osize))
end

function makeLabel(label, size)
	for i =  1, size[1] do
		label[i] = torch.DoubleTensor({torch.random(3)})
	end
	return label
end

collectgarbage()

local model, model_name, size = nets[1](libs[1])
model = model:cuda()
local inputs = makeInput(libs[1],size):cuda()
local labels = torch.DoubleTensor(size[1])
labels = makeLabel(labels, size):cuda()
local lib_name = libs[1][5]
print('ModelType: ' .. model_name, 'Kernels: ' .. lib_name,
      'Input shape: ' .. inputs:size(1) .. 'x' .. inputs:size(2) ..
      'x' .. inputs:size(3) .. 'x' .. inputs:size(4))

sys.tic()

local parameters, gradParameters = model:getParameters()

local optimState = {
    learningRate = 0.02
    --learningRateDecay = 0.0,
    --momentum = 0,
    --dampening = 0.0,
    --weightDecay = 0
}


--loss = nn.ClassNLLCriterion():cuda()
loss = nn.CrossEntropyCriterion():cuda()

for epoch = 1, 10 do
   feval = function(parameters)
      model:zeroGradParameters()
      local outputs = model:forward(inputs)
      local err = loss:forward(outputs, labels)
      local gradOutputs = loss:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      print(epoch..' '..err)
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)
end

sys.toc()
