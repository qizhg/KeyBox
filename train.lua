-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.
require('optim')

function train_batch()
    -- start a new episode
    local batch = batch_init(g_opts.batch_size)
    local reward = {}
    local input = {}
    local action = {}
    local active = {}

    -- play the games
    --local ct = 0
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end
        input[t] = batch_input(batch, active[t], t)
        local out = g_model:forward(input[t])
	-- for some reason multinomial fails sometimes
        if not pcall(function() 
	      action[t] = torch.multinomial(torch.exp(out[1]), 1) 
		    end) 
    then
            action[t] = torch.multinomial(torch.ones(out[1]:size()),1)
        end
        --ct =ct + action[t]:eq(6):sum()
        batch_act(batch, action[t]:view(-1), active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t],t == g_opts.max_steps)
    end
    local success = batch_success(batch)
    --ntoggles[batch_num] = ct

    -- increase difficulty if necessary
    if g_opts.curriculum == 1 then
        apply_curriculum(batch, success)
    end

    -- do back-propagation
    g_paramdx:zero()
    local stat = {}
    local R = torch.Tensor(g_opts.batch_size * g_opts.nagents):zero()
    for t = g_opts.max_steps, 1, -1 do
        if active[t] ~= nil and active[t]:sum() > 0 then
            local out = g_model:forward(input[t])
            R:add(reward[t]) -- cumulative reward
            local baseline = out[2]
            baseline:cmul(active[t])
            R:cmul(active[t])
            stat.bl_cost = (stat.bl_cost or 0) + g_bl_loss:forward(baseline, R)
            stat.bl_count = (stat.bl_count or 0) + active[t]:sum()
            local bl_grad = g_bl_loss:backward(baseline, R):mul(g_opts.alpha)
            baseline:add(-1, R)
            local grad = torch.Tensor(g_opts.batch_size * g_opts.nagents, g_opts.nactions):zero()
            grad:scatter(2, action[t], baseline)

            ----  compute listener_grad_action with entropy regularization
            local beta = 0.01
            local logp = out[1]
            local entropy_grad = logp:clone():add(1)
            entropy_grad:cmul(torch.exp(logp))
            entropy_grad:mul(beta)
            entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
            grad:add(entropy_grad)
            grad:div(g_opts.batch_size)
            g_model:backward(input[t], {grad, bl_grad})
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

function train(N)
    for n = 1, N do
        local stat = {}
        for k = 1, g_opts.nbatches do
            batch_num = k
            xlua.progress(k, g_opts.nbatches)
            if g_opts.nworker > 1 then
                g_paramdx:zero()
                for w = 1, g_opts.nworker do
                    g_workers:addjob(w, train_batch_thread,
                        function(paramdx_thread, s)
                            g_paramdx:add(paramdx_thread)
                            for k, v in pairs(s) do
                                stat[k] = (stat[k] or 0) + v
                            end
                        end,
                        g_opts, g_paramx
                    )
                end
                g_workers:synchronize()
            else
                local s = train_batch()
                for k, v in pairs(s) do
                    stat[k] = (stat[k] or 0) + v
                end
            end
            g_update_param(g_paramx, g_paramdx)
        end
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
        print(format_stat(stat))
        table.insert(g_log, stat)
        --g_save_model()
    end
end


function g_update_param(x, dx)
    dx:div(g_opts.nworker)
    if g_opts.max_grad_norm > 0 then
        if dx:norm() > g_opts.max_grad_norm then
            dx:div(dx:norm() / g_opts.max_grad_norm)
        end
    end
    local f = function(x0) return x, dx end
    if not g_optim_state then g_optim_state = {} end
    local config = {learningRate = g_opts.lrate}
    if g_opts.optim == 'sgd' then
        config.momentum = g_opts.momentum
        config.weightDecay = g_opts.wdecay
        optim.sgd(f, x, config, g_optim_state)
    elseif g_opts.optim == 'rmsprop' then
        config.alpha = g_opts.rmsprop_alpha
        config.epsilon = g_opts.rmsprop_eps
        config.weightDecay = g_opts.wdecay
        optim.rmsprop(f, x, config, g_optim_state)
    elseif g_opts.optim == 'adam' then
        config.beta1 = g_opts.adam_beta1
        config.beta2 = g_opts.adam_beta2
        config.epsilon = g_opts.adam_eps
        optim.adam(f, x, config, g_optim_state)
    else
        error('wrong optim')
    end
end
