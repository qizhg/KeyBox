-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function train_batch()
    -- start a new episode
    local batch = batch_init(g_opts.batch_size)
    local active = {}
    local reward = {}
    local input = {}
    local action = {}
    local comm = {}
    local symbol = {}
    local Gumbel_noise ={}

    -- play the games
    comm[0] = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0) --1
    local prev_mem_out_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --3
    local prev_hid_context_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --4
    local prev_cell_context_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --5
    local prev_hid_comm_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --6
    local prev_cell_comm_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --7
    local prev_hid_final_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0)--8
    local prev_cell_final_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --9
    local prev_mem_out_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --11
    local prev_hid_context_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --12
    local prev_cell_context_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --13
    local prev_hid_comm_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --14
    local prev_cell_comm_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --15
    local prev_hid_final_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --16
    local prev_cell_final_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0)--17
    
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end

        local mem_state_monitoring = batch_input_monitoring(batch, active[t], t) --2
        local mem_state_acting = batch_input(batch, active[t], t) --10
        
        input[t] = {}
        input[t][1] = comm[t-1]:clone()
        input[t][2] = mem_state_monitoring:clone()
        input[t][3] = prev_mem_out_monitoring:clone()
        input[t][4] = prev_hid_context_monitoring:clone()
        input[t][5] = prev_cell_context_monitoring:clone()
        input[t][6] = prev_hid_comm_monitoring:clone()
        input[t][7] = prev_cell_comm_monitoring:clone()
        input[t][8] = prev_hid_final_monitoring:clone()
        input[t][9] = prev_cell_final_monitoring:clone()
        input[t][10] = mem_state_acting:clone()
        input[t][11] = prev_mem_out_acting:clone()
        input[t][12] = prev_hid_context_acting:clone()
        input[t][13] = prev_cell_context_acting:clone()
        input[t][14] = prev_hid_comm_acting:clone()
        input[t][15] = prev_cell_comm_acting:clone()
        input[t][16] = prev_hid_final_acting:clone()
        input[t][17] = prev_cell_final_acting:clone()

        local out = g_model:forward(input[t])

        action[t] = sample_multinomial(torch.exp(out[2])) 
        prev_mem_out_monitoring = out[4]:clone()--4
        prev_hid_context_monitoring = out[5]:clone() --5 
        prev_cell_context_monitoring = out[6]:clone()--6
        prev_hid_comm_monitoring = out[7]:clone()--7
        prev_cell_comm_monitoring = out[8]:clone()--8
        prev_hid_final_monitoring = out[9]:clone()--9
        prev_cell_final_monitoring = out[10]:clone()--10
        prev_mem_out_acting = out[11]:clone()--11
        prev_hid_context_acting = out[12]:clone()--12
        prev_cell_context_acting = out[13]:clone()--13
        prev_hid_comm_acting = out[14]:clone()--14
        prev_cell_comm_acting = out[15]:clone()--15
        prev_hid_final_acting = out[16]:clone()--16
        prev_cell_final_acting = out[17]:clone()--17

	    if g_opts.traing == 'RL' then
            symbol[t] = sample_multinomial(torch.exp(out[1]))
            comm[t] = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0)
            comm[t]:scatter(2, symbol[t], torch.ones(#batch * g_opts.nagents,1))
        elseif g_opts.traing == 'Gumbel' then
            Gumbel_noise[t] = torch.rand(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):log():neg():log():neg()
            comm[t] = g_Gumbel:forward({out[1],Gumbel_noise[t]}):clone()
        else
            error('training method wrong!!!')
        end
        
        batch_act(batch, action[t]:view(-1), active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t],t == g_opts.max_steps)
    end
    local success = batch_success(batch)

    -- increase difficulty if necessary
    if g_opts.curriculum == 1 then
        apply_curriculum(batch, success)
    end

    -- do back-propagation
    g_paramdx:zero()
    local grad_comm = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0)

    local grad_mem_out_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --4
    local grad_hid_context_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --5
    local grad_cell_context_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --6
    local grad_hid_comm_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --7
    local grad_cell_comm_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --8
    local grad_hid_final_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0)--9
    local grad_cell_final_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --10
    local grad_mem_out_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --11
    local grad_hid_context_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --12
    local grad_cell_context_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --13
    local grad_hid_comm_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --14
    local grad_cell_comm_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --15
    local grad_hid_final_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) --16
    local grad_cell_final_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0)--17

    local stat = {}
    local R = torch.Tensor(g_opts.batch_size * g_opts.nagents):zero()
    local baseline
    for t = g_opts.max_steps, 1, -1 do
        if active[t] ~= nil and active[t]:sum() > 0 then
            local out = g_model:forward(input[t])
            R:add(reward[t])
            R:cmul(active[t])

            --grad_baseline
            local baseline = out[3]
            baseline:cmul(active[t])
            stat.bl_cost = (stat.bl_cost or 0) + g_bl_loss:forward(baseline, R)
            stat.bl_count = (stat.bl_count or 0) + active[t]:sum()
            local grad_baseline = g_bl_loss:backward(baseline, R):mul(g_opts.alpha)

            --grad_action_monitoring
            local grad_action_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0)
            if g_opts.traing == 'RL' then
                grad_action_monitoring:scatter(2, symbol[t], baseline - R)
                local logp_action_monitoring = out[1]
                local entropy_action_monitoring = logp_action_monitoring:clone():add(1)
                entropy_action_monitoring:cmul(torch.exp(logp_action_monitoring))
                entropy_action_monitoring:mul(g_opts.beta)
                entropy_action_monitoring:cmul(active[t]:view(-1,1):expandAs(entropy_action_monitoring):clone())
                grad_action_monitoring:add(entropy_action_monitoring)
            elseif g_opts.traing == 'Gumbel' then
                g_Gumbel:forward({out[1],Gumbel_noise[t]})
                g_Gumbel:backward({out[1],Gumbel_noise[t]}, grad_comm)
                grad_action_monitoring = g_modules['Gumbel_logp'].gradInput:clone()
            end

            
            --grad_action_acting
            local grad_action_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.nactions):fill(0)
            grad_action_acting:scatter(2, action[t], baseline - R)
            local logp = out[2]
            local entropy_grad_action_acting = logp:clone():add(1)
            entropy_grad_action_acting:cmul(torch.exp(logp))
            entropy_grad_action_acting:mul(g_opts.beta)
            entropy_grad_action_acting:cmul(active[t]:view(-1,1):expandAs(entropy_grad_action_acting):clone())
            grad_action_acting:add(entropy_grad_action_acting)
            
            --normalize with div(#batch_size)
            grad_baseline:div(g_opts.batch_size)
            grad_action_monitoring:div(g_opts.batch_size)
            grad_action_acting:div(g_opts.batch_size)

            --backward with grad recurrent
            local grad_table = {}
            grad_table[1] = grad_action_monitoring
            grad_table[2] = grad_action_acting
            grad_table[3] = grad_baseline
            grad_table[4] = grad_mem_out_monitoring:clone()
            grad_table[5] = grad_hid_context_monitoring:clone()
            grad_table[6] = grad_cell_context_monitoring:clone()
            grad_table[7] = grad_hid_comm_monitoring:clone()
            grad_table[8] = grad_cell_comm_monitoring:clone()
            grad_table[9] = grad_hid_final_monitoring:clone()
            grad_table[10] = grad_cell_final_monitoring:clone()
            grad_table[11] = grad_mem_out_acting:clone()
            grad_table[12] = grad_hid_context_acting:clone()
            grad_table[13] = grad_cell_context_acting:clone()
            grad_table[14] = grad_hid_comm_acting:clone()
            grad_table[15] = grad_cell_comm_acting:clone()
            grad_table[16] = grad_hid_final_acting:clone()
            grad_table[17] = grad_cell_final_acting:clone()
            g_model:backward(input[t], grad_table)
            grad_mem_out_monitoring = g_modules['prev_mem_out_monitoring'].gradInput:clone()
            grad_hid_context_monitoring = g_modules['prev_hid_context_monitoring'].gradInput:clone()
            grad_cell_context_monitoring = g_modules['prev_cell_context_monitoring'].gradInput:clone()
            grad_hid_comm_monitoring = g_modules['prev_hid_comm_monitoring'].gradInput:clone()
            grad_cell_comm_monitoring = g_modules['prev_cell_comm_monitoring'].gradInput:clone()
            grad_hid_final_monitoring = g_modules['prev_hid_final_monitoring'].gradInput:clone()
            grad_cell_final_monitoring = g_modules['prev_cell_final_monitoring'].gradInput:clone()
            grad_mem_out_acting = g_modules['prev_mem_out_acting'].gradInput:clone()
            grad_hid_context_acting = g_modules['prev_hid_context_acting'].gradInput:clone()
            grad_cell_context_acting = g_modules['prev_cell_context_acting'].gradInput:clone()
            grad_hid_comm_acting = g_modules['prev_hid_comm_acting'].gradInput:clone()
            grad_cell_comm_acting = g_modules['prev_cell_comm_acting'].gradInput:clone()
            grad_hid_final_acting = g_modules['prev_hid_final_acting'].gradInput:clone()
            grad_cell_final_acting = g_modules['prev_cell_final_acting'].gradInput:clone()
            
        end
    end

    R:resize(g_opts.batch_size, g_opts.nagents)
    -- stat by game type
    for i, g in pairs(batch) do
        stat.reward = (stat.reward or 0) + R[i]:mean()
        stat.success = (stat.success or 0) + success[i]
        stat.count = (stat.count or 0) + 1


        local t = torch.type(batch[i])
        stat['reward_' .. t] = (stat['reward_' .. t] or 0) + R[i]:mean()
        stat['success_' .. t] = (stat['success_' .. t] or 0) + success[i]
        stat['count_' .. t] = (stat['count_' .. t] or 0) + 1
    end
    return stat
end

function apply_curriculum(batch,success)
    for i = 1, #batch do
        local gname = batch[i].__typename
        g_factory:collect_result(gname,success[i])
        local count = g_factory:count(gname)
        local total_count = g_factory:total_count(gname)
        local pct = g_factory:success_percent(gname)
        if not g_factory.helpers[gname].frozen then
            if total_count > g_opts.curriculum_total_count then
                print('freezing ' .. gname)
                g_factory:hardest(gname)
                g_factory:freeze(gname)
            else
                if count > g_opts.curriculum_min_count then
                    if pct > g_opts.curriculum_pct_high then
                        g_factory:harder(gname)
                        print('making ' .. gname .. ' harder')
                        print(format_helpers())
                    end
                    if pct < g_opts.curriculum_pct_low then
                        g_factory:easier(gname)
                        print('making ' .. gname .. ' easier')
                        print(format_helpers())
                    end
                    g_factory:reset_counters(gname)
                end
            end
        end
    end
end

function train_batch_thread(opts_orig, paramx_orig)
    g_opts = opts_orig
    g_paramx:copy(paramx_orig)
    local stat = train_batch()
    return g_paramdx, stat
end
