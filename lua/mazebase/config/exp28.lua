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
sso.n_keyboxpairs = 2
sso.boxstatus = 'all' --all | one
sso.n_colors = sso.n_keyboxpairs
sso.costs.success_open = -5
g_opts.visibile_attr_monitoring = {'type', 'color' , 'id'}
g_opts.visibile_attr = {'type', 'color', 'status'}
g_opts.model = 'MLP_A3C'
g_opts.nlayers = 2
g_opts.model_id = 1
g_opts.nhop = 1

local traing = {'RL', 'Gumbel'}
local nsymbols_monitoring = {4, 16, 32, 50}
local lr = {1e-3, 5e-4, 2.5e-4}
local hidsz = {50, 128}
local beta = {0.01, 0.1}
local Gumbel_temp = {1.0, 0.5, 1.5}

g_opts.traing = traing[torch.random(1,#traing)]
g_opts.nsymbols_monitoring = nsymbols_monitoring[torch.random(1,#nsymbols_monitoring)]
g_opts.lr = lr[torch.random(1,#lr)]
g_opts.hidsz = hidsz[torch.random(1,#hidsz)]
g_opts.beta = beta[torch.random(1,#beta)]
g_opts.Gumbel_temp = Gumbel_temp[torch.random(1,#Gumbel_temp)]



g_opts.MH = mapH[1]
g_opts.MW = mapW[1]
g_opts.n_keyboxpairs = sso.n_keyboxpairs
g_opts.n_colors = sso.n_colors
g_opts.boxstatus = sso.boxstatus





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
