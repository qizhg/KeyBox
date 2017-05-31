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

local function factorial(n)
    local res = 1
    for i = 1, n do 
        res = res * i
    end
    return res
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
    local comm_encoder = nn.Sequential()
    local num_out = factorial(g_opts.n_colors)
    comm_encoder:add(nn.Linear(g_opts.hidsz, g_opts.hidsz))
    comm_encoder:add(nonlin())
    comm_encoder:add(nn.Linear(g_opts.hidsz, num_out))
    comm_encoder:add(nn.LogSoftMax())
    local out_monitoring = comm_encoder(input2hid_monitoring)

    local model = nn.gModule({input_monitoring}, 
    						 {out_monitoring})
    return model
end
