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
    local hid_state = torch.Tensor(g_opts.batch_size, g_opts.hidsz):fill(0)
    local cell_state = torch.Tensor(g_opts.batch_size, g_opts.hidsz):fill(0)
    local comm_in_shape = torch.Tensor(#batch, g_opts.answer_num_symbols):fill(0)
    local dummy_comm_in = torch.Tensor(1, g_opts.answer_num_symbols):zero()
    dummy_comm_in[1][1] = 1
    dummy_comm_in = dummy_comm_in:expandAs(comm_in_shape):clone()
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end

        input[t] = {}
        input[t][1] = batch_input_asker(batch, active[t]):clone()
        input[t][2] = dummy_comm_in:clone()
        input[t][3] = hid_state:clone()
        input[t][4] = cell_state:clone()
        local out = ask_model:forward(input[t])
        hid_state = out[3]:clone()
        cell_state = out[4]:clone()

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

    -- do back-propagation
    ask_paramdx:zero()
    local grad_hid = torch.Tensor(g_opts.batch_size, g_opts.hidsz):fill(0)
    local grad_cell = torch.Tensor(g_opts.batch_size, g_opts.hidsz):fill(0)
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
            grad:div(g_opts.batch_size)
            
            local grad_table = {grad, bl_grad, grad_hid, grad_cell}
            ask_model:backward(input[t], grad_table)
            grad_hid = ask_modules['prev_hid'].gradInput:clone()
            grad_cell = ask_modules['prev_cell'].gradInput:clone()

        end
    end
    --print(ask_paramdx:norm())


    local stat={}
    stat.reward = reward_sum:sum()
    stat.success = success:sum()
    stat.count = g_opts.batch_size
    return stat
end

-- EVERYTHING ABOVE RUNS ON THREADS

function train(N)

    local threashold = 1
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
