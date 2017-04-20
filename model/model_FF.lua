-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('nn')
require('nngraph')
paths.dofile('LinearNB.lua')

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

local function build_lookup_bow(input, context, hop)
    local A_LT = nn.LookupTable(g_opts.nwords, g_opts.hidsz)(context)
    share('A_LT', A_LT)
    g_modules['A_LT'] = A_LT
    local A_V = nn.View(-1, g_opts.max_attributes, g_opts.hidsz):setNumInputDims(2)(A_LT)
    local Ain = nn.Sum(3)(A_V)

    local B_LT = nn.LookupTable(g_opts.nwords, g_opts.hidsz)(context)
    share('B_LT', B_LT)
    g_modules['B_LT'] = B_LT
    local B_V = nn.View(-1, g_opts.max_attributes, g_opts.hidsz):setNumInputDims(2)(B_LT)
    local Bin = nn.Sum(3)(B_V)

    local hid3dim = nn.View(1, -1):setNumInputDims(1)(input)
    local MMaout = nn.MM(false, true)
    local Aout = MMaout({hid3dim, Ain})
    local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)
    local P = nn.SoftMax()(Aout2dim)
    g_modules[hop]['prob'] = P.data.module
    local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
    local MMbout = nn.MM(false, false)
    return MMbout({probs3dim, Bin})
end

local function build_memory(input, context)
    local hid = {}
    hid[0] = input

    for h = 1, g_opts.nhop do
        g_modules[h] = {}
        local Bout = build_lookup_bow(hid[h-1], context, h)
        local C = nn.LinearNB(g_opts.hidsz, g_opts.hidsz)(hid[h-1])
        share('proj', C)
        local D = nn.CAddTable()({C, Bout})
        hid[h] = nonlin()(D)
    end
    return hid
end



local function build_model_memnn()
    local input = nn.Identity()()
    local context = nn.Identity()()
    local hid = build_memory(input, context)
    return {input, context}, hid[#hid]
end

function g_build_model()
    local input, mem_out
    g_shareList = {}
    g_modules = {}
    
    input, mem_out = build_model_memnn()
    

    local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(mem_out))
    local action = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act)
    local action_prob = nn.LogSoftMax()(action)
    local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(mem_out))
    local baseline = nn.Linear(g_opts.hidsz, 1)(hid_bl)
    local model = nn.gModule(input, {action_prob, baseline})

    for _, l in pairs(g_shareList) do
        if #l > 1 then
            local m1 = l[1].data.module
            for j = 2,#l do
                local m2 = l[j].data.module
                m2:share(m1,'weight','bias','gradWeight','gradBias')
            end
        end
    end
    return model
end