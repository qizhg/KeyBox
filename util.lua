-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function sample_multinomial(p)
    -- for some reason multinomial fails sometimes
    local s, sample = pcall(
        function() 
            return torch.multinomial(p, 1) 
        end) 
    if s == false then
        sample = torch.multinomial(torch.ones(p:size()),1)
    end
    return sample
end

function tensor_to_words(input, show_prob)
    for i = 1, input:size(1) do
        local line = i .. ':'
        for j = 1, input:size(2) do
            line = line .. '\t'  .. g_ivocab[input[i][j]]
        end
        if show_prob then
            for h = 1, g_opts.nhop do
                line = line .. '\t' .. string.format('%.2f', g_modules[h]['prob'].output[1][i])
            end
        end
        print(line)
    end
end

function combs(k, n)
    if k*n == 0 then 
        local ret = {{}}
        return ret
    elseif k >= n then 
        local ret={}
        ret[1] = {}
        for i = 1, n do
            ret[1][1+#ret[1]] = i
        end
        return ret
    else
        local ret_case1 = combs(k, n-1)
        local ret_case2 = combs(k-1, n-1)
        for i = 1, #ret_case2 do
            ret_case2[i][1+#ret_case2[i]] = n
            ret_case1[1+#ret_case1] = ret_case2[i]
        end
        return ret_case1
    end

end

function has_value (tab, val)
    for index, value in ipairs(tab) do
        if value == val then
            return true
        end
    end

    return false
end
function index2yx(index, W)
    --index = (y-1)*W + x
    x = index % W
    if x== 0 then x = W end
    y = (index-x)/W + 1
    return y, x

end

function proc_stat(stat)
	for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
                
            end
        end
        if stat.bl_count ~= nil and stat.bl_count > 0 then
            stat.bl_cost = stat.bl_cost / stat.bl_count
        else
            stat.bl_cost = 0
        end
    stat.epoch = #g_log + 1
end


function format_stat(stat)
    local a = {}
    for n in pairs(stat) do table.insert(a, n) end
    table.sort(a)
    local str = ''
    for i,n in ipairs(a) do
        if string.find(n,'count_') then
            str = str .. n .. ': ' .. string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    for i,n in ipairs(a) do
        if string.find(n,'reward_') then
            str = str .. n .. ': ' ..  string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    for i,n in ipairs(a) do
        if string.find(n,'success_') then
            str = str .. n .. ': ' ..  string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    str = str .. 'bl_cost: ' .. string.format("%2.4g",stat['bl_cost']) .. ' '
    str = str .. 'reward: ' .. string.format("%2.4g",stat['reward']) .. ' '
    str = str .. 'success: ' .. string.format("%2.4g",stat['success']) .. ' '
    str = str .. 'epoch: ' .. stat['epoch']
    return str
end
function print_tensor(a)
    local str = ''
    for s = 1, a:size(1) do str = str .. string.format("%2.4g",a[s]) .. ' '  end
    return str
end
function format_helpers(gname)
    local str = ''
    if not gname then
        for i,j in pairs(g_factory.helpers) do
            str = str .. i .. ' :: '
            str = str .. 'mapW: ' .. print_tensor(j.mapW) .. ' ||| '
            str = str .. 'mapH: ' .. print_tensor(j.mapH) .. ' ||| '
            str = str .. 'wpct: ' .. print_tensor(j.waterpct) .. ' ||| '
            str = str .. 'bpct: ' .. print_tensor(j.blockspct) .. ' ||| '
            str = str .. '\n'
        end
    else
        local j = g_factory.helpers[gname]
        str = str .. gname .. ' :: '
        str = str .. 'mapW: ' .. print_tensor(j.mapW) .. ' ||| '
        str = str .. 'mapH: ' .. print_tensor(j.mapH) .. ' ||| '
        str = str .. 'wpct: ' .. print_tensor(j.waterpct) .. ' ||| '
        str = str .. 'bpct: ' .. print_tensor(j.blockspct) .. ' ||| '
        str = str .. '\n'
    end
    return str
end

function g_load_model()
    if g_opts.load ~= '' then
        if paths.filep(g_opts.load) == false then
            print('WARNING: Failed to load from ' .. g_opts.load)
            return
        end
        local f = torch.load(g_opts.load)
        g_paramx:copy(f.paramx)
        g_log = f.log
        if f.log_test then g_log_test = f.log_test end
        g_plot_stat = {}
        for i = 1, #g_log do
            g_plot_stat[i] = {g_log[i].epoch, g_log[i].reward, g_log[i].success, g_log[i].bl_cost}
        end
        if f['optim_state'] then g_optim_state = f['optim_state'] end
        print('model loaded from ', g_opts.load)
    end
end

function g_save_model()
    if g_opts.save ~= '' then
        f = {opts=g_opts, paramx=g_paramx, log=g_log}
        if g_optim_state then f['optim_state'] = g_optim_state end
        if g_log_test then f['log_test'] = g_log_test end
        torch.save(g_opts.save, f)
        print('model saved to ', g_opts.save)
    end
end

function g_save_glogs()
    if g_opts.save ~= '' then
        f = {opts=g_opts, log=g_logs}
        torch.save(g_opts.save, f)
    end
end

--[[
local function gen_matching_label(mathcing_string, key_color, box_colors)
    if key_color > sso.n_keyboxpairs then
        g_opts.id2matchingstring[id] = mathcing_string
        g_opts.matchingstring2id[mathcing_string] = id
        id = id + 1
    else
        local cache = mathcing_string
        for i, box_color in pairs(box_colors) do 
            mathcing_string = mathcing_string..key_color..'-'..box_color..' '
            local box_colors_next = {table.unpack(box_colors)}
            table.remove(box_colors_next, i)
            gen_matching_label(mathcing_string, key_color+1, box_colors_next)
            mathcing_string = cache
        end
    end
end

g_opts.id2matchingstring={}
g_opts.matchingstring2id={}
local mathcing_string=''
local box_colors = {}
for i=1,sso.n_keyboxpairs do
    table.insert(box_colors, i)
end
id = 1
gen_matching_label(mathcing_string, 1, box_colors)
--]]
