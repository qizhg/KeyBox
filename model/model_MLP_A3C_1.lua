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
        local atab = nn.LookupTable(in_dim, g_opts.nsymbols_monitoring)
        g_modules.atab_monitoring = atab
        a:add(atab)
        a:add(nn.Sum(2))
        a:add(nn.Add(g_opts.nsymbols_monitoring)) -- bias
        a:add(nonlin())
        --a:add(nn.BatchNormalization(g_opts.nsymbols_monitoring))

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
        
        hidstate = a(input)
    else
        error('wrong nlayers')
    end

    return hidstate
end

function g_build_model()

	g_modules = {}

    input_monitoring = nn.Identity()()
	local input2hid_monitoring = mlp_monitoring(input_monitoring)
	
	local out_monitoring
	if g_opts.traing == 'Continues2' then
        out_monitoring = input2hid_monitoring
    else
    	local action_monitoring = nn.Linear(g_opts.hidsz, g_opts.nsymbols_monitoring)(input2hid_monitoring)
        out_monitoring = nn.LogSoftMax()(action_monitoring)
    end

    input_acting = nn.Identity()()
    comm_in = nn.Identity()()
    g_modules['comm_in'] = comm_in.data.module
	local input2hid_acting = mlp_acting(input_acting)
	local hid_final_acting
	if g_opts.traing == 'Continues2' then
        hid_final_acting = nn.JoinTable(2)({comm_in, input2hid_acting})
        --hid_final_acting = nn.CAddTable()({comm_in, input2hid_acting})
    else
    	local comm_in2hid = nn.LinearNB(g_opts.nsymbols_monitoring, g_opts.hidsz)(comm_in)
    	hid_final_acting = nn.JoinTable(2)({comm_in2hid, input2hid_acting})
        --hid_final_acting = nn.CAddTable()({comm_in2hid, input2hid_acting})
    end
    local final_sz  = g_opts.hidsz +  g_opts.nsymbols_monitoring
	
	local hid_act_acting = nonlin()(nn.Linear(final_sz, g_opts.hidsz)(hid_final_acting))
    local action_acting = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act_acting)
    local action_prob_acting = nn.LogSoftMax()(action_acting)
    local hid_bl_acting = nonlin()(nn.Linear(final_sz, g_opts.hidsz)(hid_final_acting))
    local baseline_acting = nn.Linear(g_opts.hidsz, 1)(hid_bl_acting)

    local model = nn.gModule({input_monitoring, input_acting, comm_in}, 
    						 {out_monitoring, action_prob_acting, baseline_acting})
    return model
end
