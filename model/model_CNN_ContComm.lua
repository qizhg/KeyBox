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
local function build_conv_monitoring(input)
    local out_dim
    local d = g_opts.MH

    local conv1 = nn.SpatialConvolution(g_opts.convdim, g_opts.convdim, 3, 3, 1, 1, 1, 1)(input)
    local nonl1 = nonlin()(conv1)

    nonl1 = nn.SpatialMaxPooling(2, 2, 2, 2)(nonl1)
    d = math.floor(d / 2)

    local conv2 = nn.SpatialConvolution(g_opts.convdim, g_opts.convdim, 3, 3, 1, 1, 1, 1)(nonl1)
    local nonl2 = nonlin()(conv2)
    --assert(d > 1 and d < 6)

    out_dim = d * d * g_opts.convdim
    local fc0 = nn.View(out_dim):setNumInputDims(3)(nonl2)
    local fc1 = nn.Linear(out_dim, g_opts.hidsz)(fc0)
    return nonlin()(fc1)
end

local function conv_monitoring(input_monitoring)
     -- process 2D spatial information
    local in_emb = nn.LookupTable(g_opts.nwords, g_opts.convdim)(input_monitoring)
    g_modules.LT_monitoring = in_emb.data.module
    local in_A = nn.View(-1, g_opts.max_attributes, g_opts.convdim):setNumInputDims(2)(in_emb)
    local in_bow = nn.Sum(3)(in_A)
    local in_bow2d = nn.View(g_opts.MH, g_opts.MW, g_opts.convdim):setNumInputDims(2)(in_bow)
    local in_conv = nn.Transpose({2,4})(in_bow2d)

    local conv_out = build_conv_monitoring(in_conv)
    return conv_out
end

local function build_conv_acting(input)
    local out_dim
    local d = g_opts.conv_sz

    local conv1 = nn.SpatialConvolution(g_opts.convdim, g_opts.convdim, 3, 3, 1, 1, 1, 1)(input)
    local nonl1 = nonlin()(conv1)

    nonl1 = nn.SpatialMaxPooling(2, 2, 2, 2)(nonl1)
    d = math.floor(d / 2)

    local conv2 = nn.SpatialConvolution(g_opts.convdim, g_opts.convdim, 3, 3, 1, 1, 1, 1)(nonl1)
    local nonl2 = nonlin()(conv2)
    --assert(d > 1 and d < 6)

    out_dim = d * d * g_opts.convdim
    local fc0 = nn.View(out_dim):setNumInputDims(3)(nonl2)
    local fc1 = nn.Linear(out_dim, g_opts.hidsz)(fc0)
    return nonlin()(fc1)
end

local function conv_acting(input_acting)
     -- process 2D spatial information
    local in_emb = nn.LookupTable(g_opts.nwords, g_opts.convdim)(input_acting)
    g_modules.LT = in_emb.data.module
    local in_A = nn.View(-1, g_opts.max_attributes, g_opts.convdim):setNumInputDims(2)(in_emb)
    local in_bow = nn.Sum(3)(in_A)
    local in_bow2d = nn.View(g_opts.conv_sz, g_opts.conv_sz, g_opts.convdim):setNumInputDims(2)(in_bow)
    local in_conv = nn.Transpose({2,4})(in_bow2d)

    local conv_out = build_conv_acting(in_conv)
    return conv_out
end



function g_build_model()
	g_modules = {}

    local input_monitoring = nn.Identity()()
    local input2hid_monitoring = conv_monitoring(input_monitoring)
    local comm_mean = nn.Linear(g_opts.hidsz, g_opts.nsymbols_monitoring)(input2hid_monitoring)
    g_modules['comm_mean'] = comm_mean.data.module
    local out_monitoring = comm_mean

    local input_acting = nn.Identity()()
    local comm_in = nn.Identity()()
    g_modules['comm_in'] = comm_in.data.module
    local input2hid_acting = conv_acting(input_acting)
    local comm_decoder = nn.Sequential()
    comm_decoder:add(nn.Linear(g_opts.nsymbols_monitoring, g_opts.hidsz))
    comm_decoder:add(nonlin())
    local comm_in_embeding = comm_decoder(comm_in)

    local hid_final_acting = nn.CAddTable()({comm_in_embeding, input2hid_acting})
    
    local hid_act_acting = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_final_acting))
    local action_acting = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act_acting)
    local action_prob_acting = nn.LogSoftMax()(action_acting)
    local hid_bl_acting = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_final_acting))
    local baseline_acting = nn.Linear(g_opts.hidsz, 1)(hid_bl_acting)

    local model = nn.gModule({input_monitoring, input_acting, comm_in}, 
                             {out_monitoring, action_prob_acting, baseline_acting})
    return model
end
