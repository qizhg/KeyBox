-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.
require('optim')

if g_opts.model == 'FF' then   
    paths.dofile('train_batch/train_batch_FF.lua')
elseif g_opts.model == 'Recurrent' then
    paths.dofile('train_batch/train_batch_Recurrent.lua') 
else
    error('wrong model name')
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
