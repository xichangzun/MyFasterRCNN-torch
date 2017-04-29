require 'nn'
require 'nngraph'
require 'BatchIterator'
require 'utilities'
require 'objective'
require 'optim'
require 'image'
require 'Detector'

--read options
cmd = torch.CmdLine()
cmd:option('-snapshot', 800, 'snapshot interval')
cmd:option('-plot', 200, 'plot training progress interval')
cmd:option('-lr', 1E-3, 'learn rate')
opt  = cmd:parse(arg)

--torch.setdefaulttensortype('torch.CudaTensor')

Convolution = nn.SpatialConvolution
Avg = nn.SpatialAveragePooling
ReLU = nn.ReLU
Max = nn.SpatialMaxPooling
SBatchNorm = nn.Identity

function plot_training_progress(prefix, stats)
  local fn = prefix .. '_progress.png'
  gnuplot.pngfigure(fn)
  gnuplot.title('Traning progress over time')
  
  local xs = torch.range(1, #stats.pcls)
  
  gnuplot.plot(
    { 'pcls', xs, torch.Tensor(stats.pcls), '-' },
    { 'preg', xs, torch.Tensor(stats.preg), '-' },
    { 'dcls', xs, torch.Tensor(stats.dcls), '-' },
    { 'dreg', xs, torch.Tensor(stats.dreg), '-' }
  )
 
  gnuplot.axis({ 0, #stats.pcls, 0, 10 })
  gnuplot.xlabel('iteration')
  gnuplot.ylabel('loss')
  
  gnuplot.plotflush()
end

function shortcut(nInputPlane, nOutputPlane, stride)
    if nInputPlane ~= nOutputPlane then
    	-- Strided, zero-padded identity shortcut
        return nn.Sequential()
        	:add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(1)
            	:add(nn.Identity())
            	:add(nn.Sequential()
					:add(nn.SpatialConvolution(nInputPlane, nOutputPlane-nInputPlane, 1, 1, 1, 1))
					:add(nn.MulConstant(0))))
    else
        return nn.SpatialAveragePooling(1, 1, stride, stride)
    end
end

function basicblock(n, stride)
	local nInputPlane = iChannels
	iChannels = n

	local s = nn.Sequential()
	s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
	s:add(SBatchNorm(n))
	s:add(ReLU(true))
	s:add(Convolution(n,n,3,3,1,1,1,1))
	s:add(SBatchNorm(n))

	return nn.Sequential()
		:add(nn.ConcatTable()
		:add(s)
		:add(shortcut(nInputPlane, n, stride)))
		:add(nn.CAddTable(true))
		:add(ReLU(true))
end

function AnchorNetwork(nInputPlane, n, kernelWidth)
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(nInputPlane, n, kernelWidth,kernelWidth, 1,1))
    net:add(nn.PReLU())
    net:add(nn.SpatialConvolution(n, 3 * (2 + 4), 1, 1))  -- aspect ratios { 1:1, 2:1, 1:2 } x { class, left, top, width, height }
    return net
end

function load_model(model, network_filename)

	local weights, gradient = combine_and_flatten_parameters(model.pnet, model.cnet)
	local training_stats
  if network_filename and #network_filename > 0 then
	local stored = load_obj(network_filename)
	training_stats = stored.stats
	weights:copy(stored.weights)
  end

  return weights, gradient, training_stats
end

layers = {
	{ filters= 128, kW= 2 },
    { filters= 256, kW= 2 },
    { filters= 512, kW= 2 }
}

anchor = {
	{kW = 3,n = 256,input = 3 },
	{kW = 3,n = 256,input = 4 },
	{kW = 5,n = 256,input = 4 },
	{kW = 7,n = 256,input = 4 }
}

pc_models = {}
input = nn.Identity()()
prev = input
p_outputs = {}
iChannels = 64

m1 = nn.Sequential()
m1:add(Convolution(3,64,3,3,1,1,1,1))
m1:add(SBatchNorm(64))
m1:add(ReLU(true))
prev = m1(prev)
table.insert(pc_models,prev)

for i,v in ipairs(layers) do
	prev = basicblock(v.filters,v.kW)(prev)
	table.insert(pc_models,prev)
end

for i,v in ipairs(anchor) do
	local net  = AnchorNetwork(layers[v.input-1].filters,v.n,v.kW)
	table.insert(p_outputs,net(pc_models[v.input]))
end
table.insert(p_outputs,pc_models[#pc_models])

_pnet = nn.gModule({input},p_outputs)

-- print(_pnet:forward(torch.Tensor(2,3,224,224)))

cnet = nn.Sequential()
cnet:add(nn.Linear(6*6*512,1024))
cnet:add(nn.BatchNormalization(1024))
cnet:add(ReLU(true))
cnet:add(nn.Dropout(0.5))
cnet:add(nn.Linear(1024,512))
cnet:add(nn.BatchNormalization(512))
cnet:add(nn.Dropout(0.5))
input = nn.Identity()()
node = cnet(input)
cout = nn.Linear(512,21)(node)
rout = nn.Linear(512,4)(node)
_cnet = nn.gModule({input},{rout,cout})

VOC2007_cfg = {
	class_count = 20,  -- excluding background class
	target_smaller_side = 480,
	scales = { 48, 96, 192, 384 },
	max_pixel_size = 1000,
	normalization = {},
	augmentation = { vflip = 0, hflip = 0.25, random_scaling = 0, aspect_jitter = 0 },
	color_space = 'rgb',
	roi_pooling = { kw = 6, kh = 6 },
	examples_base_path = '',
	background_base_path = '',
	batch_size = 200,
	positive_threshold = 0.6, 
	negative_threshold = 0.25,
	best_match = true,
	nearby_aversion = true
}

model = {
	layers = layers,
	cfg = VOC2007_cfg,
	pnet = _pnet:cuda(),
	cnet = _cnet:cuda()
}

-- img = image.load('IM2007/2007_000027.jpg',3,'float')

dofile('index.lua')
training_data = {}
training_data.training_set = SampleIndex
training_data.validatoin_set = {}
weights, gradient, training_stats = load_model( model, network_filename)
if not training_stats then
	training_stats = { pcls={}, preg={}, dcls={}, dreg={} }
end

batch_iterator = BatchIterator.new(model, training_data,classSet)
eval_objective_grad = create_objective(model, weights, gradient, batch_iterator, training_stats)
local sgd_state = {
	learningRate = opt.lr, 
	weightDecay = 0.0005, 
	momentum = 0.9,
	dampening = 0.0,
	nesterov = true,
}
for i=1,800 do
	if i % 200 == 0 then
      opt.lr = opt.lr / 5
      adam_state.lr = opt.lr
    end
    local timer = torch.Timer()
    --local _, loss = optim.adam(eval_objective_grad, weights, adam_state)
    --local _, loss = optim.nag(eval_objective_grad, weights, nag_state)
    local _, loss = optim.sgd(eval_objective_grad, weights, sgd_state)
    
    local time = timer:time().real

    print(string.format('%d: loss: %f', i, loss[1]))
    
    if i%opt.plot == 0 then
      plot_training_progress('result'..i, training_stats)
    end
    
    if i%opt.snapshot == 0 then
      -- save snapshot
      save_model(string.format('%s_%06d.t7','result', i), weights, opt, training_stats)
    end
end

