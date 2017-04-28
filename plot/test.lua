
require'gnuplot'
file = 'rldl1.t7'
local f = torch.load(file)
print(#f.paramx)
print(f.opts)
