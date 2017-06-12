-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function test_batch()
    -- start a new episode
    if g_opts.training_testing then g_opts.training_testing = 0 end
    local batch_size_test = 2000
    local batch = batch_init(batch_size_test)
    local active = {}
    local reward = {}
    local input = {}
    local action = {}
    local symbol = {}
    local Gumbel_noise ={}
    local comm = {}
    local comm_sz = g_opts.nsymbols_monitoring
    --local matching = batch_matching(batch)
    --csvigo.save({path = "matching_"..g_opts.exp..".csv", data = torch.totable(matching:view(-1,1))})

    
    comm[0] = torch.Tensor(#batch * g_opts.nagents, comm_sz):fill(0)
    active[1] = batch_active(batch)
    local oneshot_monitoring = batch_input_mlp_monitoring(batch, active[1], 1)
    local oneshot_noise = torch.Tensor(#batch * g_opts.nagents, comm_sz):fill(0)
    if g_opts.noise_std and g_opts.noise_std>0 then
        --print(g_opts.noise_std)
        oneshot_noise:normal(0, g_opts.noise_std)
    end

    -- play the games
    local agent = batch[1].agent
    for t = 1, g_opts.max_steps do
        --print(agent.loc.y..', '..agent.loc.x)
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end

        input[t] = {}
        if g_opts.oneshot_comm == true then 
            input[t][1] = oneshot_monitoring:clone()
            input[t][4] = oneshot_noise:clone()
        else
            input[t][1] = batch_input_mlp_monitoring(batch, active[t], t)
            input[t][4] = torch.Tensor(#batch * g_opts.nagents, comm_sz):fill(0)
            if g_opts.noise_std and g_opts.noise_std>0 then 
                input[t][4]:normal(0, g_opts.noise_std)
            end
        end
        input[t][2] = batch_input_mlp(batch, active[t], t)
        input[t][3] = comm[t-1]:clone()

        local out = g_model:forward(input[t])
        if t==1 and g_opts.oneshot_comm == true then 
            local channel_in = g_modules['comm_mean'].output:clone()
            local channel_out = nn.Sigmoid():forward(channel_in)
            --print(channel_in[1])
            --csvigo.save({path = "channelIn_"..g_opts.exp..".csv", data = torch.totable(channel_in)})
            --csvigo.save({path = "channelOut_"..g_opts.exp..".csv", data = torch.totable(channel_out)})
        end

        action[t] = sample_multinomial(torch.exp(out[2]))

        comm[t] = out[1]:clone()
        
        batch_act(batch, action[t]:view(-1), active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t],t == g_opts.max_steps)
    end
    local success = batch_success(batch)

    local R = torch.Tensor(batch_size_test * g_opts.nagents):zero()
    local stat={}
    for t = g_opts.max_steps, 1, -1 do
        if active[t] ~= nil and active[t]:sum() > 0 then
            local out = g_model:forward(input[t])
            R:add(reward[t])
            R:cmul(active[t])
        end
    end

    R:resize(batch_size_test, g_opts.nagents)
    -- stat by game type
    for i, g in pairs(batch) do
        stat.reward = (stat.reward or 0) + R[i]:mean()
        stat.success = (stat.success or 0) + success[i]
        stat.count = (stat.count or 0) + 1
    end
    return stat

end