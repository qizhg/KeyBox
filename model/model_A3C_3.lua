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
    if g_opts.share_word_embedding == true then
        share('A_LT', A_LT)
        g_modules['A_LT'] = A_LT
    else
        share('A_LT_monitoring', A_LT)
        g_modules['A_LT_monitoring'] = A_LT
    end
    local A_V = nn.View(-1, g_opts.max_attributes, g_opts.hidsz):setNumInputDims(2)(A_LT)
    local Ain = nn.Sum(3)(A_V)

    local B_LT = nn.LookupTable(g_opts.nwords, g_opts.hidsz)(context)
    if g_opts.share_word_embedding == true then 
        share('B_LT', B_LT)
        g_modules['B_LT'] = B_LT
    else
        share('B_LT_monitoring', B_LT)
        g_modules['B_LT_monitoring'] = B_LT
    end

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
    ------memory
    local dummy_context_monitoring = nn.Identity()()
    local mem_state_monitoring = nn.Identity()()
    local mem_out_monitoring = build_model_memnn_monitoring(dummy_context_monitoring, mem_state_monitoring)

    ------comm
    local comm_in = nn.Identity()() --(#batch, #symbols)
    g_modules['comm_in'] = comm_in.data.module
    local comm_in_embedding_monitoring = nn.LinearNB(g_opts.nsymbols_monitoring, g_opts.hidsz)(comm_in)

    -----final
    local final_lstm_in_monitoring = nn.CAddTable()({mem_out_monitoring, comm_in_embedding_monitoring})
    local prev_hid_final_monitoring = nn.Identity()() 
    g_modules['prev_hid_final_monitoring'] = prev_hid_final_monitoring.data.module
    local prev_cell_final_monitoring = nn.Identity()()
    g_modules['prev_cell_final_monitoring'] = prev_cell_final_monitoring.data.module
    local hid_final_monitoring, cell_final_monitoring = build_lstm(final_lstm_in_monitoring, prev_hid_final_monitoring, prev_cell_final_monitoring, g_opts.hidsz, g_opts.hidsz)

    -----out
    local hid_act_monitoring = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_final_monitoring))
    local action_monitoring = nn.Linear(g_opts.hidsz, g_opts.nsymbols_monitoring)(hid_act_monitoring)
    local out_monitoring
    if g_opts.traing == 'Continues2' then
        out_monitoring = action_monitoring
    else
        out_monitoring = nn.LogSoftMax()(action_monitoring)
    end
    --END monitoring--

    --acting
    ------memory
    local dummy_context_acting = nn.Identity()()
    local mem_state_acting = nn.Identity()()
    local mem_out_acting = build_model_memnn(dummy_context_acting, mem_state_acting)

    ------comm
    local comm_in_embedding_acting = nn.LinearNB(g_opts.nsymbols_monitoring, g_opts.hidsz)(comm_in)

    -----final
    local final_lstm_in_acting = nn.JoinTable(2)({mem_out_acting, comm_in_embedding_acting})
    local prev_hid_final_acting = nn.Identity()() 
    g_modules['prev_hid_final_acting'] = prev_hid_final_acting.data.module
    local prev_cell_final_acting = nn.Identity()()
    g_modules['prev_cell_final_acting'] = prev_cell_final_acting.data.module
    local hid_final_acting, cell_final_acting = build_lstm(final_lstm_in_acting, prev_hid_final_acting, prev_cell_final_acting, g_opts.hidsz, 2*g_opts.hidsz)

    -----out
    local hid_act_acting = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_final_acting))
    local action_acting = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act_acting)
    local action_prob_acting = nn.LogSoftMax()(action_acting)
    local hid_bl_acting = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid_final_acting))
    local baseline_acting = nn.Linear(g_opts.hidsz, 1)(hid_bl_acting)
    --END acting--
    
    local model = nn.gModule(
        {comm_in, --1
        dummy_context_monitoring, --2 
        mem_state_monitoring, --3
        prev_hid_final_monitoring, --4 
        prev_cell_final_monitoring, --5
        dummy_context_acting, --6
        mem_state_acting, --7
        prev_hid_final_acting, --8
        prev_cell_final_acting}, --9
        
        {out_monitoring, --1
        action_prob_acting, --2
        baseline_acting, --3
        hid_final_monitoring, --4 
        cell_final_monitoring, --5
        hid_final_acting, --6
        cell_final_acting} --7
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