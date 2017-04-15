require 'optim'
require('nn')
require('nngraph')
--g_disp = require('display')

function train_batch(task_id)
	local batch = batch_init(g_opts.batch_size, task_id)
    local num_batchs = (epoch_num-1)*g_opts.nbatches + batch_num

    local reward = {}
    local input = {}
    local action = {}
    local active = {}

    --play the game (forward pass)
    local dummy = torch.Tensor(#batch, g_opts.hidsz):fill(0.1)
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end
        local mem = batch_input_asker(batch, active[t])
        local context = dummy:clone()
        input[t] = {context, mem}
        local out = ask_model:forward(input[t])
        action[t] = sample_multinomial(torch.exp(out[1]))  --(#batch, 1)

        batch_act_asker(batch, action[t], active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t])
    end
    local success = batch_success(batch)

    -- do back-propagation
    ask_paramdx:zero()
    local R = torch.Tensor(g_opts.batch_size):zero()
    for t = g_opts.max_steps, 1, -1 do
        if active[t] ~= nil and active[t]:sum() > 0 then
            local out = ask_model:forward(input[t])
            R:add(reward[t]) -- cumulative reward
            local baseline = out[2]
            baseline:cmul(active[t])
            R:cmul(active[t])
            local bl_grad = ask_bl_loss:backward(baseline, R):mul(g_opts.alpha)
            baseline:add(-1, R)
            local grad = torch.Tensor(g_opts.batch_size, g_opts.nactions):zero()
            grad:scatter(2, action[t], baseline)
            grad:div(g_opts.batch_size)
            ask_model:backward(input[t], {grad, bl_grad})
        end
    end


    local stat={}
    --stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
end

function train_batch_thread(opts_orig, listener_paramx_orig, speaker_paramx_orig,task_id)
    g_opts = opts_orig
    g_listener_paramx:copy(listener_paramx_orig)
    g_speaker_paramx:copy(speaker_paramx_orig)
    local stat = train_batch(task_id)
    return g_listener_paramdx, g_speaker_paramdx, stat
end

-- EVERYTHING ABOVE RUNS ON THREADS

function train(N)
    for n = 1, N do
        epoch_num= n
        local stat = {} --for the epoch
        for k = 1, g_opts.nbatches do
            batch_num = k
            xlua.progress(k, g_opts.nbatches)
            local s = train_batch()
            merge_stat(stat, s)
        end

        g_update_param(ask_paramx, ask_paramdx, 'ask' )

        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                --stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
                --stat['active' .. s] = stat['active' .. s] / v
                --stat['avg_err' .. s] = stat['avg_err' .. s] / v
            end
        end

        stat.epoch = #g_log + 1
        print(format_stat(stat))
        table.insert(g_log, stat)
    end

end

function g_update_param(x, dx, model_name)
    local f = function(x0) return x, dx end
    if not g_optim_state then
        g_optim_state = {}
        for i = 1, #model_id2name do
            g_optim_state[i] = {} 
        end
    end
    local model_id = model_name2id[model_name]
    local config = {learningRate = g_opts.lrate}
    if g_opts.optim == 'sgd' then
        config.momentum = g_opts.momentum
        config.weightDecay = g_opts.wdecay
        optim.sgd(f, x, config, g_optim_state[model_id])
    elseif g_opts.optim == 'rmsprop' then
        config.alpha = g_opts.rmsprop_alpha
        config.epsilon = g_opts.rmsprob_eps
        config.weightDecay = g_opts.wdecay
        optim.rmsprop(f, x, config, g_optim_state[model_id])
    elseif g_opts.optim == 'adam' then
        config.beta1 = g_opts.adam_beta1
        config.beta2 = g_opts.adam_beta2
        config.epsilon = g_opts.adam_eps
        optim.adam(f, x, config, g_optim_state[model_id])
    else
        error('wrong optim')
    end

end
