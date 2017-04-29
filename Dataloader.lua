require 'utilities'
require 'Rect'
require 'image'
dofile('index.lua')
green = torch.Tensor({0,1,0})
prefix = 'Users/xichangzun/Code/luacode/MyFasterRCNN'
for i = 1,50 do
	fn = string.format('AN2007/2007_00%04d.xml',i)
	_,f = pcall(io.open,fn,'r')
	if f then
		print('case :',i)
		-- img = image.load(string.format('IM2007/2007_00%04d.jpg',i), 3, 'float')
		content = f:read('*all')
		for name,box in string.gmatch(content,'<object>%s*<name>(.-)</name>.-<bndbox>(.-)</bndbox>.-</object>') do 
			print(classSet[name])
			-- local index = {}
			-- for v in string.gmatch(box, '%d*') do
			-- 	table.insert( index , tonumber(v) )
			-- end
			-- rect = Rect.new(table.unpack(index))
			-- draw_rectangle(img,rect,green)	
		end
		-- image.saveJPG(string.format('test%d.jpg', i), img)
		f:close()
	end
end

