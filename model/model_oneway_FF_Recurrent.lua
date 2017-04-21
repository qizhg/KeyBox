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

----acting----
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
        if g_modules[h] == nil then g_modules[h] = {} end
        local Bout = build_lookup_bow(hid[h-1], context, h)
        local C = nn.LinearNB(g_opts.hidsz, g_opts.hidsz)(hid[h-1])
        share('proj', C)
        local D = nn.CAddTable()({C, Bout})
        hid[h] = nonlin()(D)
    end
    return hid
end



local function build_model_memnn(input, context)
    local hid = build_memory(input, context)
    return hid[#hid]
end
----END acting----

----monitoring----
local function build_lookup_bow_monitoring(input, context, hop)
    local A_LT = nn.LookupTable(g_opts.nwords, g_opts.hidsz)(context)
    share('A_LT_monitoring', A_LT)
    g_modules['A_LT_monitoring'] = A_LT
    local A_V = nn.View(-1, g_opts.max_attributes, g_opts.hidsz):setNumInputDims(2)(A_LT)
    local Ain = nn.Sum(3)(A_V)

    local B_LT = nn.LookupTable(g_opts.nwords, g_opts.hidsz)(context)
    share('B_LT_monitoring', B_LT)
    g_modules['B_LT_monitoring'] = B_LT
    local B_V = nn.View(-1, g_opts.max_attributes, g_opts.hidsz):setNumInputDims(2)(B_LT)
    local Bin = nn.Sum(3)(B_V)

    local hid3dim = nn.View(1, -1):setNumInputDims(1)(input)
    local MMaout = nn.MM(false, true)
    local Aout = MMaout({hid3dim, Ain})
    local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)
    local P = nn.SoftMax()(Aout2dim)
    g_modules[hop]['prob_monitoring'] = P.data.module
    local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
    local MMbout = nn.MM(false, false)
    return MMbout({probs3dim, Bin})
end

local function build_memory_monitoring(input, context)
    local hid = {}
    hid[0] = input

    for h = 1, g_opts.nhop do
        if g_modules[h] == nil then g_modules[h] = {} end
        local Bout = build_lookup_bow_monitoring(hid[h-1], context, h)
        local C = nn.LinearNB(g_opts.hidsz, g_opts.hidsz)(hid[h-1])
        share('proj_monitoring', C)
        local D = nn.CAddTable()({C, Bout})
        hid[h] = nonlin()(D)
    end
    return hid
end
----END monitoring----


local function build_model_memnn_monitoring(input, context)
    local hid = build_memory_monitoring(input, context)
    return hid[#hid]
end

function g_build_model()
    g_shareList = {}
    g_modules = {}

    --monitoring
    local mem_state_monitoring = nn.Identity()()
    local context_monitoring = nn.Identity()()  --constant
    local mem_out_monitoring = build_model_memnn_monitoring(context_monitoring, mem_state_monitoring)

    local hid_act_monitoring = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(mem_out_monitoring))
    local action_monitoring = nn.Linear(g_opts.hidsz, g_opts.nsymbols_monitoring)(hid_act_monitoring)
    local action_prob_monitoring = nn.LogSoftMax()(action_monitoring)
    local hid_bl_monitoring = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(mem_out_monitoring))
    local baseline_monitoring = nn.Linear(g_opts.hidsz, 1)(hid_bl_monitoring)
    --END monitoring--

    
    --acting
    ----LSTM build context vector
    local prev_mem_out = nn.Identity()()
    g_modules['prev_mem_out'] = prev_mem_out.data.module
    local prev_hid = nn.Identity()() 
    g_modules['prev_hid'] = prev_hid.data.module
    local prev_cell = nn.Identity()()
    g_modules['prev_cell'] = prev_cell.data.module
    local comm_in = nn.Identity()() --(#batch) integears
    g_modules['comm_in'] = comm_in.data.module

    local symbol_LT = nn.LookupTable(g_opts.nsymbols_monitoring, g_opts.hidsz)(comm_in)
    g_modules['symbol_LT'] = symbol_LT
    local symbol_embedding = nn.Sum(2)(symbol_LT)
    local lstm_in = nn.JoinTable(2)({symbol_embedding, prev_mem_out})
    local hid, cell = build_lstm(lstm_in, prev_hid, prev_cell, g_opts.hidsz, 2 * g_opts.hidsz)

    local mem_state = nn.Identity()()
    local mem_out = build_model_memnn(hid, mem_state)
    

    local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(mem_out))
    local action = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act)
    local action_prob = nn.LogSoftMax()(action)
    local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(mem_out))
    local baseline = nn.Linear(g_opts.hidsz, 1)(hid_bl)
    --END acting--
    
    local model = nn.gModule(
        {context_monitoring, mem_state_monitoring, mem_state, prev_mem_out, comm_in, prev_hid, prev_cell}, 
        {action_prob, baseline, mem_out, hid, cell, action_prob_monitoring, baseline_monitoring}
        )

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