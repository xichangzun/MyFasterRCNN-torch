require 'nn'
require 'nngraph'
require 'Localizer'
require 'utilities'
require 'Rect'
require 'image'

Convolution = nn.SpatialConvolution
Avg = nn.SpatialAveragePooling
ReLU = nn.ReLU
Max = nn.SpatialMaxPooling
SBatchNorm = nn.SpatialBatchNormalization


function shortcut(nInputPlane, nOutputPlane, stride)
    if nInputPlane ~= nOutputPlane then
    	-- Strided, zero-padded identity shortcut
        return nn.Sequential()
        	:add(nn.SpatialAveragePooling(1, 1, stride, stride))
            :add(nn.Concat(2)
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

layers = {
	{ filters= 128, dW= 2 },
    { filters= 256, dW= 2 },
    { filters= 384, dW= 2 }
}

anchor = {
	{kW = 3,n = 256,input = 3 },
	{kW = 3,n = 256,input = 4 },
	{kW = 5,n = 256,input = 4 },
	{kW = 7,n = 256,input = 4 }
}

-- iChannels = 64

-- model = nn.Sequential()
-- m1 = nn.Sequential()
-- m1:add(Convolution(3,64,3,3,1,1,1,1))
-- m1:add(SBatchNorm(64))
-- m1:add(ReLU(true))
-- model:add(m1)
-- for i,v in ipairs(layers) do
-- 	cm = basicblock(v.filters,v.dW)
-- 	model:add(cm)
-- end


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
	prev = basicblock(v.filters,v.dW)(prev)
	table.insert(pc_models,prev)
end

for i,v in ipairs(anchor) do
	local net  = AnchorNetwork(layers[v.input-1].filters,v.n,v.kW)
	table.insert(p_outputs,net(pc_models[v.input]))
end
pnet = nn.gModule({input},p_outputs)


localizer = Localizer.new(pnet.outnode.children[1])
print(localizer.layers)

green = torch.Tensor({0,1,0})
prefix = 'Users/xichangzun/Code/luacode/MyFasterRCNN'
for i = 1,50 do
	fn = string.format('AN2007/2007_00%04d.xml',i)
	_,f = pcall(io.open,fn,'r')
	if f then
		print('case :',i)
		img = image.load(string.format('IM2007/2007_00%04d.jpg',i), 3, 'float')
		content = f:read('*all')
		for name,box in string.gmatch(content,'<object>%s*<name>(.-)</name>.-<bndbox>(.-)</bndbox>.-</object>') do 
			local index = {}
			for v in string.gmatch(box, '%d*') do
				table.insert( index , tonumber(v) )
			end
			rect = Rect.new(table.unpack(index))
			srect = localizer:inputToFeatureRect(rect)
			print(srect)
			minx,miny,maxx,maxy = srect:unpack()
			rect = localizer:featureToInputRect(minx,miny,maxx,maxy)
			draw_rectangle(img,rect,green)	
		end
		image.saveJPG(string.format('test%d.jpg', i), img)
	end
end



