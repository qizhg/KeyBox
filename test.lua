-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

package.path = package.path .. ';lua/?/init.lua'
g_mazebase = require('mazebase')

 function init_master()
    require('xlua')

    require'gnuplot'
    require'csvigo'
    paths.dofile('util.lua')
    paths.dofile('model.lua')
    paths.dofile('train.lua')
    torch.setdefaulttensortype('torch.FloatTensor')
end

 function init_worker()
    require('xlua')
    paths.dofile('util.lua')
    torch.setdefaulttensortype('torch.FloatTensor')
end

 function init_threads()
    print('starting ' .. g_opts.nworker .. ' workers')
     threads = require('threads')
    threads.Threads.serialization('threads.sharedserialize')
     workers = threads.Threads(g_opts.nworker, init_worker)
    workers:specific(true)
    for w = 1, g_opts.nworker do
        workers:addjob(w,
            function(opts_orig, vocab_orig)
                package.path = package.path .. ';lua/?/init.lua'
                g_mazebase = require('mazebase')
                g_opts = opts_orig
                g_vocab = vocab_orig

                paths.dofile('model.lua')
                paths.dofile('train.lua')
                g_init_model()
                g_mazebase.init_game()
            end,
            function() end,
            g_opts, g_vocab
        )
    end
    workers:synchronize()
    return workers
end

local exp = 'exp_2bb_1D_std00'
local f = torch.load(exp..'.t7')
g_opts = f.opts
g_opts.load = exp..'.t7'
g_mazebase.init_vocab()
g_mazebase.init_game()
init_master()
g_init_model()
g_load_model()

local test_file = 'test_batch/test_'..g_opts.model..'.lua'
paths.dofile(test_file)
g_opts.batch_size = 100

--------loc fixed-------------

g_opts.loc_keys = {}
g_opts.loc_keys[1] = {}
g_opts.loc_keys[1].y = 1
g_opts.loc_keys[1].x = 1
g_opts.loc_keys[2] = {}
g_opts.loc_keys[2].y = 1
g_opts.loc_keys[2].x = 4
g_opts.loc_boxes = {}
g_opts.loc_boxes[1] = {}
g_opts.loc_boxes[1].y = 4
g_opts.loc_boxes[1].x = 1
g_opts.loc_boxes[2] = {}
g_opts.loc_boxes[2].y = 4
g_opts.loc_boxes[2].x = 4

--------agent loc fixed-------------
--[[
g_opts.loc_agents = {}
g_opts.loc_agents[1].y = 
g_opts.loc_agents[1].x = 
--]]

--------id fixed-------------
--[[
g_opts.id_keys = torch.randperm(2)
g_opts.id_keys[1] = 1
g_opts.id_keys[2] = 2
g_opts.id_boxes = torch.randperm(2)
g_opts.id_boxes[1] = 1
g_opts.id_boxes[2] = 2
--]]

--------color fixed-------------
--[[
g_opts.color_keys = torch.randperm(2)
g_opts.color_keys[1] = 1
g_opts.color_keys[2] = 2
g_opts.color_boxes = torch.randperm(2)
g_opts.color_boxes[1] = 1
g_opts.color_boxes[2] = 2
--]]


stat = test_batch()
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
print(stat)
--[[
batch = batch_init(g_opts.batch_size)
input = {}
comm = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0)
active = batch_active(batch)
input[1] = batch_input_monitoring(batch, active, 1)
input[2] = batch_input(batch, active, 1)
input[3] = comm:clone()

out = g_model:forward(input)
comm = out[1]:clone()
csvigo.save({path = "comm"..g_opts.exp_id..".csv", data = torch.totable(comm)})
matching = batch_matching(batch)
csvigo.save({path = "matching"..g_opts.exp_id..".csv", data = torch.totable(matching:view(-1,1))})

batch = batch_init(g_opts.batch_size)
active = {}
     reward = {}
     input = {}
     action = {}
     symbol = {}
     Gumbel_noise ={}
     comm = {}
     comm_sz = g_opts.nsymbols_monitoring

    comm[0] = torch.Tensor(#batch * g_opts.nagents, comm_sz):fill(0)
    active[1] = batch_active(batch)
     oneshot_comm = batch_input_monitoring(batch, active[1], 1)

    -- play the games
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end

        input[t] = {}
        if g_opts.oneshot_comm == true then
            input[t][1] = oneshot_comm:clone()
        else
            input[t][1] = batch_input_monitoring(batch, active[t], t)
        end
        input[t][2] = batch_input(batch, active[t], t)
        input[t][3] = comm[t-1]:clone()

         out = g_model:forward(input[t])
        action[t] = sample_multinomial(torch.exp(out[2]))

        comm[t] = out[1]:clone()

        batch_act(batch, action[t]:view(-1), active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t],t == g_opts.max_steps)
    end
     success = batch_success(batch)
    print(success:sum())
--]]

--[[
local csv = csvigo.load({path = "./comm1.csv", verbose = false, mode = "raw"})
local input = torch.Tensor(csv)
local Gumbel_noise = torch.rand(input:size(1), input:size(2)):log():neg():log():neg()
local out = g_model:forward({input, Gumbel_noise})

csvigo.save({path = "symbol1"..".csv", data = torch.totable(out[2])})
--]]
