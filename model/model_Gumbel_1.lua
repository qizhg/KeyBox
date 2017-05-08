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

    local input = nn.Identity()()
    local input_embeding = nonlin()(nn.Linear(g_opts.nsymbols_monitoring, g_opts.nsymbols_monitoring)(input))
    local logp = nn.LogSoftMax()(input_embeding)
    local Gumbel_noise = nn.Identity()()

    local temp = 4.0
    local Gumbel_trick = nn.CAddTable()({Gumbel_noise, logp})
    local Gumbel_trick_temp = nn.MulConstant(1.0/temp)(Gumbel_trick)
    local Gumbel_SoftMax = nn.SoftMax()(Gumbel_trick_temp)

    local output = nn.Linear(g_opts.nsymbols_monitoring, g_opts.nsymbols_monitoring)(Gumbel_SoftMax)

    local model = nn.gModule({input, Gumbel_noise}, 
    						 {output, Gumbel_SoftMax})
    return model
end
