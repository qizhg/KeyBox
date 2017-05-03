-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('nn')
require('nngraph')
paths.dofile('LinearNB.lua')
paths.dofile('LSTM.lua')

local function share(name, mod)
    if g_shareList[name] == nil then g_shareList[name] = {} end
    table.insert(g_shareList[name], mod)
end

local function nonlin()
    if g_opts.nonlin == 'tanh' then
        return nn.Tanh()
    elseif g_opts.nonlin == 'relu' then
        return nn.ReLU()
    elseif g_opts.nonlin == 'none' then
        return nn.Identity()
    else
        error('wrong nonlin')
    end
end

local function mlp_monitoring( input )
    local MH = g_opts.conv_sz
    local MW = g_opts.conv_sz
    local memsize = g_opts.memsize
    local na = g_opts.nactions
    local nwords = g_opts.nwords

    local in_dim = MH*MW*nwords + 1

    local hidstate
    if g_opts.nlayers > 1 then
        local a = nn.Sequential()
        local atab = nn.LookupTable(in_dim, g_opts.hidsz)
        g_modules.atab_monitoring = atab
        a:add(atab)
        a:add(nn.Sum(2))
        a:add(nn.Add(g_opts.hidsz)) -- bias
        a:add(nonlin())
        for l = 3, g_opts.nlayers do
            a:add(nn.Linear(g_opts.hidsz, g_opts.hidsz))
            a:add(nonlin())
        end
        hidstate = a(input)
    else
        error('wrong nlayers')
    end

    return hidstate
end


local function mlp_acting( input )
    local MH = g_opts.conv_sz
    local MW = g_opts.conv_sz
    local memsize = g_opts.memsize
    local na = g_opts.nactions
    local nwords = g_opts.nwords

    local in_dim = MH*MW*nwords + 1

    local hidstate
    if g_opts.nlayers > 1 then
        local a = nn.Sequential()
        local atab = nn.LookupTable(in_dim, g_opts.hidsz)
        g_modules.atab = atab
        a:add(atab)
        a:add(nn.Sum(2))
        a:add(nn.Add(g_opts.hidsz)) -- bias
        a:add(nonlin())
        for l = 3, g_opts.nlayers do
            a:add(nn.Linear(g_opts.hidsz, g_opts.hidsz))
            a:add(nonlin())
        end
        hidstate = a(input)
    else
        error('wrong nlayers')
    end

    return hidstate
end

function g_build_model()

	g_modules = {}

    local comm_in = nn.Identity()()
    local input_acting = nn.Identity()()
    local comm = nn.Sequential()
    comm:add(nn.LookupTable(2, g_opts.hidsz))
    comm:add(nn.Add(g_opts.hidsz)) -- bias
    comm:add(nonlin())
	local comm_in_embeding = comm(comm_in)
	local input2hid_acting = mlp_acting(input_acting)

    local hid_final_acting = nn.CAddTable()({comm_in_embeding, input2hid_acting})
    
    local hid_act_acting = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_final_acting))
    local action_acting = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act_acting)
    local action_prob_acting = nn.LogSoftMax()(action_acting)
    local hid_bl_acting = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_final_acting))
    local baseline_acting = nn.Linear(g_opts.hidsz, 1)(hid_bl_acting)

    local model = nn.gModule({comm_in, input_acting}, {action_prob_acting, baseline_acting})
    return model
end
