require('nn')
require('nngraph')
paths.dofile('module/LSTM.lua')
paths.dofile('module/GRU.lua')
paths.dofile('module/LinearNB.lua')
paths.dofile('module/Gumbel.lua')

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

local function share(name, mod)
    if ask_shareList[name] == nil then ask_shareList[name] = {} end
    table.insert(ask_shareList[name], mod)
end


local function build_lookup_bow(input, context, hop)
    local A_LT = nn.LookupTable(g_opts.nwords, g_opts.hidsz)(context)
    share('A_LT', A_LT)
    ask_modules['A_LT'] = A_LT
    local A_V = nn.View(-1, g_opts.max_attributes, g_opts.hidsz):setNumInputDims(2)(A_LT)
    local Ain = nn.Sum(3)(A_V)

    local B_LT = nn.LookupTable(g_opts.nwords, g_opts.hidsz)(context)
    share('B_LT', B_LT)
    ask_modules['B_LT'] = B_LT
    local B_V = nn.View(-1, g_opts.max_attributes, g_opts.hidsz):setNumInputDims(2)(B_LT)
    local Bin = nn.Sum(3)(B_V)

    local hid3dim = nn.View(1, -1):setNumInputDims(1)(input)
    local MMaout = nn.MM(false, true)
    local Aout = MMaout({hid3dim, Ain})
    local Aout2dim = nn.View(-1):setNumInputDims(2)(Aout)
    local P = nn.SoftMax()(Aout2dim)
    ask_modules[hop]['prob'] = P.data.module
    local probs3dim = nn.View(1, -1):setNumInputDims(1)(P)
    local MMbout = nn.MM(false, false)
    return MMbout({probs3dim, Bin})
end

local function build_memory(input, context)
    local hid = {}
    hid[0] = input

    for h = 1, g_opts.nhop do
        ask_modules[h] = {}
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
    local context = nn.Identity()() --memory slots
    local hid = build_memory(input, context)
    return {input, context}, hid[#hid]
end

function build_ask_model()

	local input_table, output_table, hid

    local input_table, hid = build_model_memnn()
    local hid_act = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid))
    local action = nn.Linear(g_opts.hidsz, g_opts.nactions)(hid_act)
    local action_prob = nn.LogSoftMax()(action)
    local hid_bl = nonlin()(nn.Linear(g_opts.hidsz, g_opts.hidsz)(hid))
    local baseline = nn.Linear(g_opts.hidsz, 1)(hid_bl)

    output_table = {action_prob, baseline}
    local model = nn.gModule(input_table, output_table)
    
    for _, l in pairs(ask_shareList) do
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