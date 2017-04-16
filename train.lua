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
    local baseline = {}

    --play the game (forward pass)
    local dummy = torch.Tensor(#batch, g_opts.hidsz):fill(0.1)
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end
        local mem = batch_input_asker(batch, active[t])
        local context = dummy:clone()
        input[t] = {context, mem}
        local out = ask_model:forward(input[t])
        baseline[t] = out[2]:clone():cmul(active[t])
        

        local ep = g_opts.eps_start- num_batchs*g_opts.eps_start/g_opts.eps_end_batch
        ep = math.max(0.1,ep)
        if torch.uniform() < ep then
            action[t] = torch.LongTensor(#batch,1)
            action[t]:random(1, g_opts.nactions)
        else
            action[t] = sample_multinomial(torch.exp(out[1]))  --(#batch, 1)
        end

        batch_act_asker(batch, action[t], active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t])
    end
    local success = batch_success(batch)

    --prepare for GAE
    --[[
    local delta = {} --TD residual
    delta[g_opts.max_steps] = reward[g_opts.max_steps] - baseline[g_opts.max_steps]
    for t=1, g_opts.max_steps-1 do 
        delta[t] = reward[t] + g_opts.gamma*baseline[t+1] - baseline[t]
    end
    local A_GAE={} --GAE advatage
    A_GAE[g_opts.max_steps] = delta[g_opts.max_steps]
    for t=g_opts.max_steps-1, 1, -1 do 
        A_GAE[t] = delta[t] + g_opts.gamma*g_opts.lambda*A_GAE[t+1] 
    end
    --]]

    -- do back-propagation
    ask_paramdx:zero()
    local reward_sum = torch.Tensor(#batch):zero() --running reward sum
    for t = g_opts.max_steps, 1, -1 do
        if active[t] ~= nil and active[t]:sum() > 0 then
            reward_sum:add(reward[t])
            local out = ask_model:forward(input[t])
            
            local R = reward_sum:clone() --(#batch, )
            local baseline_step = out[2]
            baseline_step:cmul(active[t])
            R:cmul(active[t])
            local bl_grad = ask_bl_loss:backward(baseline_step, R):mul(g_opts.alpha)
            
            local grad = torch.Tensor(g_opts.batch_size, g_opts.nactions):zero()
            --grad:scatter(2, action[t], A_GAE[t]:view(-1,1):neg())
            baseline_step:add(-1, R)
            grad:scatter(2, action[t], baseline_step)
            
        --[[local beta = g_opts.beta_start - num_batchs*g_opts.beta_start/g_opts.beta_end_batch
            beta = math.max(0,beta)
            local logp = out[1]
            local entropy_grad = logp:clone():add(1)
            entropy_grad:cmul(torch.exp(logp))
            entropy_grad:mul(beta)
            entropy_grad:cmul(active[t]:view(-1,1):expandAs(entropy_grad):clone())
            grad:add(entropy_grad)--]]
            
            grad:div(g_opts.batch_size)
            ask_model:backward(input[t], {grad, bl_grad})
        end
    end
    print(ask_paramdx:norm())


    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
end

-- EVERYTHING ABOVE RUNS ON THREADS

function train(N)

    local threashold = 10
    for n = 1, N do
        epoch_num= #g_log + 1
        local stat = {} --for the epoch
        for k = 1, g_opts.nbatches do
            batch_num = k
            xlua.progress(k, g_opts.nbatches)
            local s = train_batch()
            merge_stat(stat, s)
            g_update_param(ask_paramx, ask_paramdx, 'ask' ) --update every minibatch
        end


        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
                --stat['active' .. s] = stat['active' .. s] / v
                --stat['avg_err' .. s] = stat['avg_err' .. s] / v
            end
        end
        stat.epoch = #g_log + 1
        print(format_stat(stat))
        table.insert(g_log, stat)

        g_opts.save = 'model_epoch'
        g_save_model()
        if stat.success > threashold/100.0 then
            g_opts.save = 'model_at'..threashold
            g_save_model()
            threashold  = threashold + 10
        end
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
