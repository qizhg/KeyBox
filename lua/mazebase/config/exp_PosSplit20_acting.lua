if not g_opts then g_opts = {} end
g_opts.multigames = {}
-------------------
--some shared RangeOpts
--current min, current max, min max, max max, increment
local mapH = torch.Tensor{4,4,5,10,1}
local mapW = torch.Tensor{4,4,5,10,1}
local blockspct = torch.Tensor{.0,.0, 0,.2,.01}
local waterpct = torch.Tensor{.0,.0, 0,.2,.01}

-------------------
--some shared StaticOpts
local sso = {}
-------------- costs:
sso.costs = {}
sso.costs.goal = -1
sso.costs.empty = 0.1
sso.costs.block = 1000
sso.costs.water = 0.3
sso.costs.corner = 0
sso.costs.step = 0.1
sso.costs.pushableblock = 1000
---------------------
sso.toggle_action = 0
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_corners = 0
sso.enable_boundary = 0
sso.max_attributes = g_opts.max_attributes or 6


----------Game Specific----------------
g_opts.n_keys = 2
g_opts.n_color_keys = 2
g_opts.n_boxes = 2
g_opts.n_color_boxes = 2
g_opts.status_boxes = 'all' --all | one

sso.costs.success_open = -2
g_opts.model = 'MLP_acting'
g_opts.nlayers = 2
g_opts.visibile_attr = {'type', 'color', 'status', 'id'}

g_opts.hidsz = 128

g_opts.MH = mapH[1]
g_opts.MW = mapW[1]

local f = torch.load('lua/mazebase/config/id2pos'..'_'..g_opts.MH..'_'..g_opts.MW..'_'..g_opts.n_keys..'_'..g_opts.n_boxes..'.t7')
g_opts.id2pos = f.pos
local training_percetage = 0.20
local training_testing = torch.rand(#g_opts.id2pos)
training_testing = torch.le(training_testing, training_percetage) --1: training, 0:testing
_, g_opts.training_testing_indices = torch.sort(training_testing, 1, true)
g_opts.num_training = torch.eq(training_testing,1):sum()
g_opts.num_testing = #g_opts.id2pos - g_opts.num_training
g_opts.training_testing = 1
g_log_test = {}




-- KeyBox:
local KeyBoxRangeOpts = {}
KeyBoxRangeOpts.mapH = mapH:clone()
KeyBoxRangeOpts.mapW = mapW:clone()
KeyBoxRangeOpts.blockspct = blockspct:clone()
KeyBoxRangeOpts.waterpct = waterpct:clone()

local KeyBoxStaticOpts = {}
for i,j in pairs(sso) do KeyBoxStaticOpts[i] = j end

KeyBoxOpts ={}
KeyBoxOpts.RangeOpts = KeyBoxRangeOpts
KeyBoxOpts.StaticOpts = KeyBoxStaticOpts

g_opts.multigames.KeyBox = KeyBoxOpts


return g_opts
