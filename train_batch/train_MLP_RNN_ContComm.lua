-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function train_batch()

    local batch = batch_init(g_opts.batch_size)
    local active = {}
    local reward = {}
    local input = {}
    local action = {}
    local comm = {}

    -- play the games
    comm[0] = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0) 
    local prev_hid_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0)
    local prev_cell_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) 
    local prev_hid_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) 
    local prev_cell_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0)

    active[1] = batch_active(batch)
    local oneshot_monitoring = batch_input_mlp_monitoring(batch, active[1], 1)
    
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end
        
        input[t] = {}
        input[t][1] = comm[t-1]:clone()
         if g_opts.oneshot_comm == true then 
            input[t][2] = oneshot_monitoring:clone()
        else
            input[t][2] = batch_input_mlp_monitoring(batch, active[t], t)
        end
        input[t][3] = batch_input_mlp(batch, active[t], t)
        input[t][4] = prev_hid_monitoring:clone()
        input[t][5] = prev_cell_monitoring:clone()
        input[t][6] = prev_hid_acting:clone()
        input[t][7] = prev_cell_acting:clone()

        local out = g_model:forward(input[t])

        comm[t] = out[1]:clone()
        action[t] = sample_multinomial(torch.exp(out[2])) 
        prev_hid_monitoring = out[4]:clone()
        prev_cell_monitoring = out[5]:clone()
        prev_hid_acting = out[6]:clone()
        prev_cell_acting = out[7]:clone()
        
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
    local grad_hid_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0)
    local grad_cell_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) 
    local grad_hid_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0) 
    local grad_cell_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):fill(0)

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

            --grad_out_monitoring
            local grad_out_monitoring = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0)
            grad_out_monitoring = grad_comm:clone()
            
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
            grad_action_acting:div(g_opts.batch_size)

            --backward with grad recurrent
            local grad_table = {}
            grad_table[1] = grad_out_monitoring
            grad_table[2] = grad_action_acting
            grad_table[3] = grad_baseline
            grad_table[4] = grad_hid_monitoring:clone()
            grad_table[5] = grad_cell_monitoring:clone()
            grad_table[6] = grad_hid_acting:clone()
            grad_table[7] = grad_cell_acting:clone()
            
            g_model:backward(input[t], grad_table)
            
            grad_comm = g_modules['comm_in'].gradInput:clone()
            grad_hid_monitoring = g_modules['prev_hid_monitoring'].gradInput:clone()
            grad_cell_monitoring = g_modules['prev_cell_monitoring'].gradInput:clone()
            grad_hid_acting = g_modules['prev_hid_acting'].gradInput:clone()
            grad_cell_acting = g_modules['prev_cell_acting'].gradInput:clone()
            
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
