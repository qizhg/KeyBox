
require'gnuplot'
file = 'rldl2.t7'
local f = torch.load(file)
print(#f.paramx)
