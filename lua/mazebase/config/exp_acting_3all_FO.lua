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
g_opts.n_keys = 3
g_opts.n_color_keys = 3
g_opts.n_boxes = 3
g_opts.n_color_boxes = 3
g_opts.status_boxes = 'all' --all | one

sso.costs.success_open = -5
g_opts.model = 'MLP_acting'
g_opts.nlayers = 2
g_opts.visibile_attr = {'type', 'color', 'status', 'id'}

g_opts.hidsz = 128

g_opts.MH = mapH[1]
g_opts.MW = mapW[1]
g_opts.conv_sz = 2*g_opts.MH - 1

local function gen_matching_label(mathcing_string, key_color, box_colors)
	if key_color > g_opts.n_keys then
		g_opts.id2matchingstring[id] = mathcing_string
		g_opts.matchingstring2id[mathcing_string] = id
		id = id + 1
	else
		local cache = mathcing_string
		for i, box_color in pairs(box_colors) do 
			mathcing_string = mathcing_string..key_color..'-'..box_color..' '
			local box_colors_next = {table.unpack(box_colors)}
			table.remove(box_colors_next, i)
			gen_matching_label(mathcing_string, key_color+1, box_colors_next)
			mathcing_string = cache
		end
	end
end

g_opts.id2matchingstring={}
g_opts.matchingstring2id={}
local mathcing_string=''
local box_colors = {}
for i=1,g_opts.n_keys do
	table.insert(box_colors, i)
end
id = 1
gen_matching_label(mathcing_string, 1, box_colors)
id = nil



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
