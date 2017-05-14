-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function train_batch(num_batch)
    -- start a new episode
    local batch = batch_init(g_opts.batch_size)
    local active = {}
    local reward = {}
    local input_monitoring = {}
    local input_acting = {}
    local action = {}
    local symbol = {}
    local Gumbel_noise ={}
    local comm = {}
    local comm_sz = g_opts.nsymbols_monitoring
    local matching_label = batch_matching(batch)

    
    comm[0] = torch.Tensor(#batch * g_opts.nagents, comm_sz):fill(0)
    active[1] = batch_active(batch)
    local oneshot_comm = batch_input_monitoring(batch, active[1], 1)
    local oneshot_Gumbel = torch.rand(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):log():neg():log():neg()

    -- play the games
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end

        input_monitoring[t] = {}
        input_acting[t] = {}
        if g_opts.oneshot_comm == true then 
            input_monitoring[t][1] = oneshot_comm:clone()
            input_monitoring[t][2] = oneshot_Gumbel:clone()
        else
            input_monitoring[t][1] = batch_input_monitoring(batch, active[t], t)
            input_monitoring[t][2] = torch.rand(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):log():neg():log():neg()
        end
        input_acting[t][1] = batch_input(batch, active[t], t)
        input_acting[t][2] = comm[t-1]:clone()
        

        local out_monitoring = g_model_monitoring:forward(input_monitoring[t])
        local out_acting = g_model_acting:forward(input_acting[t])
        action[t] = sample_multinomial(torch.exp(out_acting[1]))
        
        --ST-Gumbel
        local temp, symbol = torch.max(out_monitoring, 2) 
        comm[t] = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0)
        comm[t]:scatter(2, symbol, 1)
        
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
    g_paramdx_monitoring:zero()
    g_paramdx_acting:zero()
    local grad_comm = torch.Tensor(#batch * g_opts.nagents, comm_sz):fill(0)

    local stat = {}
    local R = torch.Tensor(g_opts.batch_size * g_opts.nagents):zero()
    local baseline
    for t = g_opts.max_steps, 1, -1 do
        if active[t] ~= nil and active[t]:sum() > 0 then
            local out_monitoring = g_model_monitoring:forward(input_monitoring[t])
            local out_acting = g_model_acting:forward(input_acting[t])
            R:add(reward[t])
            R:cmul(active[t])

            --grad_baseline
            local baseline = out_acting[2]
            baseline:cmul(active[t])
            stat.bl_cost = (stat.bl_cost or 0) + g_bl_loss:forward(baseline, R)
            stat.bl_count = (stat.bl_count or 0) + active[t]:sum()
            local grad_baseline = g_bl_loss:backward(baseline, R):mul(g_opts.alpha)

            --grad_action_monitoring
            local grad_action_monitoring = torch.Tensor(#batch * g_opts.nagents, comm_sz):fill(0)
            grad_action_monitoring = grad_comm:clone()
            
            --grad_action_acting
            local grad_action_acting = torch.Tensor(#batch * g_opts.nagents, g_opts.nactions):fill(0)
            grad_action_acting:scatter(2, action[t], baseline - R)
            local logp = out_acting[1]
            local entropy_grad_action_acting = logp:clone():add(1)
            entropy_grad_action_acting:cmul(torch.exp(logp))
            entropy_grad_action_acting:mul(g_opts.beta)
            entropy_grad_action_acting:cmul(active[t]:view(-1,1):expandAs(entropy_grad_action_acting):clone())
            grad_action_acting:add(entropy_grad_action_acting)

            --grad_matching
            --local criterion = nn.ClassNLLCriterion()
            --local err = criterion:forward(out[4], matching_label)
            --local grad_matching = criterion:backward(out[4], matching_label)

            
            --normalize with div(#batch_size)
            grad_action_monitoring:div(g_opts.batch_size)
            grad_baseline:div(g_opts.batch_size)
            grad_action_acting:div(g_opts.batch_size)
            --grad_matching:div(g_opts.batch_size):zero()

            --backward with grad recurrent
            local grad_table_acting = {}
            grad_table_acting[1] = grad_action_acting
            grad_table_acting[2] = grad_baseline
            --grad_table[4] = grad_matching

            local grad_monitoring = grad_action_monitoring
            
            g_model_acting:backward(input_acting[t], grad_table_acting)
            g_model_monitoring:backward(input_monitoring[t], grad_monitoring)
            
            grad_comm = g_modules['comm_in'].gradInput:clone()
            
        end
    end

    R:resize(g_opts.batch_size, g_opts.nagents)
    -- stat by game type
    for i, g in pairs(batch) do
        stat.reward = (stat.reward or 0) + R[i]:mean()
        stat.success = (stat.success or 0) + success[i]
        stat.count = (stat.count or 0) + 1
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

function train_batch_thread(opts_orig, g_paramx_monitoring_orig, g_paramx_acting_orig, num_batch)
    g_opts = opts_orig
    g_paramx_monitoring:copy(g_paramx_monitoring_orig)
    g_paramx_acting:copy(g_paramx_acting_orig)
    local stat = train_batch(num_batch)
    return g_paramdx_monitoring, g_paramdx_acting, stat
end
