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

sso.costs.success_open = -1
g_opts.model = 'MLP_acting'
g_opts.nlayers = 2
g_opts.visibile_attr = {'type', 'color', 'status', 'id'}


g_opts.hidsz = 128

g_opts.MH = mapH[1]
g_opts.MW = mapW[1]

g_opts.id2pos={}
local position_comb = combs(g_opts.n_keys+g_opts.n_boxes,g_opts.MH*g_opts.MW)
--local position_comb = combs(3,3)

local keybox_within_comb = combs(g_opts.n_keys, g_opts.n_keys+g_opts.n_boxes)
for k, v in ipairs(position_comb) do 
	for kk, vv in ipairs(keybox_within_comb) do 
		local cur_key = 1
		local cur_box = 1
		local pos={}
		for item = 1, g_opts.n_keys+g_opts.n_boxes do 
			if has_value (vv, item) then --key
				pos[cur_key] = v[vv[cur_key]]
				cur_key = cur_key +1
			else
				pos[cur_box + g_opts.n_keys] = v[item]
				cur_box = cur_box +1
			end
		end
		g_opts.id2pos[#g_opts.id2pos+1] = pos
	end
end
local training_percetage = 1.0
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
