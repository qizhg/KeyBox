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

local function mlp_encoder_monitoring( input )
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
        hidstate = a(input)
    else
        error('wrong nlayers')
    end

    return hidstate
end


local function mlp_encoder_acting(input)
    local MH = g_opts.conv_sz
    local MW = g_opts.conv_sz
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

local function comm_decoder_acting(input)
    local comm_decoder = nn.Sequential()
    comm_decoder:add(nn.Linear(g_opts.nsymbols_monitoring, g_opts.hidsz))
    local comm_in_embeding = comm_decoder(input)
    return comm_in_embeding
end

local function comm_decoder_monitoring(input)
    local comm_decoder = nn.Sequential()
    comm_decoder:add(nn.Linear(g_opts.nsymbols_monitoring, g_opts.hidsz))
    local comm_in_embeding = comm_decoder(input)
    return comm_in_embeding
end

function g_build_model()

	g_modules = {}

	comm_in = nn.Identity()()
    g_modules['comm_in'] = comm_in.data.module

	--monitoring LSTM
	local input_monitoring = nn.Identity()()
    local prev_hid_monitoring = nn.Identity()() 
    g_modules['prev_hid_monitoring'] = prev_hid_monitoring.data.module
    local prev_cell_monitoring = nn.Identity()()
    g_modules['prev_cell_monitoring'] = prev_cell_monitoring.data.module

    local input2hid_monitoring = mlp_encoder_monitoring(input_monitoring)
    local comm2hid_monitoring = comm_decoder_monitoring(comm_in)
    local lstm_input_monitoring = nn.JoinTable(2)({input2hid_monitoring, comm2hid_monitoring})
    local lstm_sz_monitoring = 2 * g_opts.hidsz
    local hid_monitoring, cell_monitoring = build_lstm(lstm_input_monitoring, prev_hid_monitoring, prev_cell_monitoring, g_opts.hidsz, lstm_sz_monitoring)

    --monitroing out	
	local out_monitoring = nn.Linear(g_opts.hidsz, g_opts.nsymbols_monitoring)(hid_monitoring)

    --acting LSTM
    local input_acting = nn.Identity()()
    local prev_hid_acting = nn.Identity()() 
    g_modules['prev_hid_acting'] = prev_hid_acting.data.module
    local prev_cell_acting = nn.Identity()()
    g_modules['prev_cell_acting'] = prev_cell_acting.data.module

    local input2hid_acting = mlp_encoder_acting(input_acting)
    local comm2hid_acting = comm_decoder_acting(comm_in)
    local lstm_input_acting = nn.JoinTable(2)({input2hid_acting, comm2hid_acting})
    local lstm_sz_acting = 2 * g_opts.hidsz
    local hid_acting, cell_acting = build_lstm(lstm_input_acting, prev_hid_acting, prev_cell_acting, g_opts.hidsz, lstm_sz_acting)
	
	--acting out
	local hid_act_acting = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_acting))
    local action_acting = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act_acting)
    local action_prob_acting = nn.LogSoftMax()(action_acting)
    local hid_bl_acting  = nonlin()( nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_acting))
    local baseline_acting = nn.Linear(g_opts.hidsz, 1)(hid_bl_acting)

    local model = nn.gModule({comm_in, input_monitoring, input_acting, prev_hid_monitoring, prev_cell_monitoring, prev_hid_acting, prev_cell_acting}, 
    						 {out_monitoring, action_prob_acting, baseline_acting, hid_monitoring, cell_monitoring , hid_acting, cell_acting})
    return model
end
