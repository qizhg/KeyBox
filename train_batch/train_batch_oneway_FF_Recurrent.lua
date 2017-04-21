-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function train_batch()
    -- start a new episode
    local batch = batch_init(g_opts.batch_size)
    local reward = {}
    local input = {}
    local action = {}
    local symbol = {}
    local active = {}

    -- play the games
    local prev_mem_out = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):type(g_opts.dtype):fill(0)
    local prev_hid = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):type(g_opts.dtype):fill(0)
    local prev_cell = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):type(g_opts.dtype):fill(0)
    symbol[0] = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):type(g_opts.dtype):fill(1)
    local dummy = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):type(g_opts.dtype):fill(0.1)
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch):type(g_opts.dtype)
        if active[t]:sum() == 0 then break end

        local mem_state = batch_input(batch, active[t], t):type(g_opts.dtype)
        local mem_state_monitoring = batch_input_monitoring(batch, active[t], t):type(g_opts.dtype)
        local context_monitoring = dummy:clone()
        local comm_in = symbol[t-1]
        input[t] = {context_monitoring, mem_state_monitoring, mem_state, prev_mem_out, comm_in, prev_hid, prev_cell}
        local out = g_model:forward(input[t])
        prev_mem_out = out[3]:clone()
        prev_hid = out[4]:clone()
        prev_cell = out[5]:clone()

	-- for some reason multinomial fails sometimes
        if not pcall(function() 
	      action[t] = torch.multinomial(torch.exp(out[1]), 1) 
		    end) 
    then
            action[t] = torch.multinomial(torch.ones(out[1]:size()),1)
        end
        
        if not pcall(function() 
          symbol[t] = torch.multinomial(torch.exp(out[6]), 1) 
            end) 
    then
            symbol[t] = torch.multinomial(torch.ones(out[6]:size()),1)
        end
        
        batch_act(batch, action[t]:view(-1), active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t],t == g_opts.max_steps):type(g_opts.dtype)
    end
    local success = batch_success(batch)

    -- increase difficulty if necessary
    if g_opts.curriculum == 1 then
        apply_curriculum(batch, success)
    end

    -- do back-propagation
    g_paramdx:zero()
    local grad_mem_out = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):type(g_opts.dtype):fill(0)
    local grad_hid = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):type(g_opts.dtype):fill(0)
    local grad_cell = torch.Tensor(#batch * g_opts.nagents, g_opts.hidsz):type(g_opts.dtype):fill(0)

    local stat = {}
    local R = torch.Tensor(g_opts.batch_size * g_opts.nagents):type(g_opts.dtype):zero()
    for t = g_opts.max_steps, 1, -1 do
        if active[t] ~= nil and active[t]:sum() > 0 then
            local out = g_model:forward(input[t])
            R:add(reward[t])
            R:cmul(active[t])
            
            --monitoring
            ----baseline
            local baseline_monitoring = out[7]
            baseline_monitoring:cmul(active[t])
            stat.bl_cost = (stat.bl_cost or 0) + g_bl_loss:forward(baseline_monitoring, R)
            stat.bl_count = (stat.bl_count or 0) + active[t]:sum()
            local grad_bl_monitoring = g_bl_loss:backward(baseline_monitoring, R):mul(g_opts.alpha)
            ----symbol action
            local grad_symbol = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.nsymbols_monitoring):type(g_opts.dtype):zero()
            baseline_monitoring:add(-1, R)
            grad_symbol:scatter(2, symbol[t], baseline_monitoring)
            ------ entropy regularization
            local logp_symbol = out[6]
            local entropy_grad_symbol = logp_symbol:clone():add(1)
            entropy_grad_symbol:cmul(torch.exp(logp_symbol))
            entropy_grad_symbol:mul(g_opts.beta)
            entropy_grad_symbol:cmul(active[t]:view(-1,1):expandAs(entropy_grad_symbol):clone())
            grad_symbol:add(entropy_grad_symbol)
            grad_symbol:div(g_opts.batch_size)

            --acting
            ----baseline
            local baseline = out[2]
            baseline:cmul(active[t])
            stat.bl_cost = (stat.bl_cost or 0) + g_bl_loss:forward(baseline, R)
            stat.bl_count = (stat.bl_count or 0) + active[t]:sum()
            local grad_bl = g_bl_loss:backward(baseline, R):mul(g_opts.alpha)
            
            ----action
            local grad_action = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.nactions):type(g_opts.dtype):zero()
            baseline:add(-1, R)
            grad_action:scatter(2, action[t], baseline)
            ------ entropy regularization
            local logp = out[1]
            local entropy_grad = logp:clone():add(1)
            entropy_grad:cmul(torch.exp(logp))
            entropy_grad:mul(g_opts.beta)
            entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
            grad_action:add(entropy_grad)
            grad_action:div(g_opts.batch_size)

            --backward with grad recurrent
            g_model:backward(input[t], 
                {grad_action, grad_bl, grad_mem_out, grad_hid, grad_cell, grad_symbol, grad_bl_monitoring}
                )
            grad_mem_out = g_modules['prev_mem_out'].gradInput:clone()
            grad_hid = g_modules['prev_hid'].gradInput:clone()
            grad_cell = g_modules['prev_cell'].gradInput:clone()
            
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
