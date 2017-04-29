require 'Rect'
require 'utilities'

local Localizer = torch.class('Localizer')

function Localizer:__init(layerInfo)
	self.layers = {}
	table.insert(self.layers,{ filters = 64, kW = 3, kH = 3,dW = 1,dH = 1,padW = 1,padH = 1})
	for i,v in ipairs(layerInfo) do
		table.insert(self.layers,{filters = v.filters, kW=3, kH=3, dW=v.dW, dH=v.dW, padW = 1, padH= 1 })
		table.insert(self.layers,{filters = v.filters, kW=3, kH=3, dW=1, dH=1, padW = 1, padH= 1 })
	end

end


function Localizer:inputToFeatureRect(rect, layer_index)
  layer_index = layer_index or #self.layers
  for i=1,layer_index do
    local l = self.layers[i]
    if l.dW < l.kW then
      rect = rect:inflate((l.kW-l.dW), (l.kH-l.dH)) --minx,miny - (l.kw-l.dw),maxx,maxy + (l.kw-l.dw)
    end

    rect = rect:offset(l.padW, l.padH)
    
    -- reduce size, keep only filters that fit completely into the rect (valid convolution)
    rect.minX = rect.minX / l.dH
    rect.minY = rect.minY / l.dH
    if (rect.maxX-l.kW) % l.dW == 0 then
      rect.maxX = math.max((rect.maxX-l.kW)/l.dW + 1, rect.minX+1)
    else
      rect.maxX = math.max(math.ceil((rect.maxX-l.kW) / l.dW) + 1, rect.minX+1)
    end
    if (rect.maxY-l.kH) % l.dH == 0 then
      rect.maxY = math.max((rect.maxY-l.kH)/l.dW + 1, rect.minY+1)
    else
      rect.maxY = math.max(math.ceil((rect.maxY-l.kH) / l.dH) + 1, rect.minY+1)
    end

  end
  return rect:snapToInt()
end

function Localizer:featureToInputRect(minX, minY, maxX, maxY, layer_index)
  layer_index = layer_index or #self.layers
  for i=layer_index,1,-1 do
    local l = self.layers[i]
    minX = minX * l.dW - l.padW
    minY = minY * l.dH - l.padW
    maxX = maxX * l.dW - l.padH + l.kW - l.dW
    maxY = maxY * l.dH - l.padH + l.kH - l.dH
  end
  return Rect.new(minX, minY, maxX, maxY)
end

