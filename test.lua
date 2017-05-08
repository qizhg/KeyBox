-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

package.path = package.path .. ';lua/?/init.lua'
g_mazebase = require('mazebase')

 function init_master()
    require('xlua')

    require'gnuplot'
    require'csvigo'
    paths.dofile('util.lua')
    paths.dofile('model.lua')
    paths.dofile('train.lua')
    torch.setdefaulttensortype('torch.FloatTensor')
end

 function init_worker()
    require('xlua')
    paths.dofile('util.lua')
    torch.setdefaulttensortype('torch.FloatTensor')
end

 function init_threads()
    print('starting ' .. g_opts.nworker .. ' workers')
     threads = require('threads')
    threads.Threads.serialization('threads.sharedserialize')
     workers = threads.Threads(g_opts.nworker, init_worker)
    workers:specific(true)
    for w = 1, g_opts.nworker do
        workers:addjob(w,
            function(opts_orig, vocab_orig)
                package.path = package.path .. ';lua/?/init.lua'
                g_mazebase = require('mazebase')
                g_opts = opts_orig
                g_vocab = vocab_orig
                
                paths.dofile('model.lua')
                paths.dofile('train.lua')
                g_init_model()
                g_mazebase.init_game()
            end,
            function() end,
            g_opts, g_vocab
        )
    end
    workers:synchronize()
    return workers
end

 cmd = torch.CmdLine()
-- model parameters
cmd:option('--model', 'A3C', 'A3C | DQN | MLP_A3C')
cmd:option('--conv_sz', 9, '')

cmd:option('--hidsz', 128, 'the size of the internal state vector')
cmd:option('--nonlin', 'relu', 'non-linearity type: tanh | relu | none')
cmd:option('--init_std', 0.2, 'STD of initial weights')
cmd:option('--max_attributes', 6, 'maximum number of attributes of each item')
cmd:option('--memsize', 20, 'size of the memory in MemNN')
cmd:option('--nhop', 1, 'the number of hops in MemNN')
-- game parameters
cmd:option('--nagents', 1, 'the number of acting agents')
cmd:option('--nactions', 6, 'the number of agent actions')
cmd:option('--max_steps', 30, 'force to end the game after this many steps')
cmd:option('--exp_id', 6, '')
-- training parameters
cmd:option('--optim', 'rmsprop', 'optimization method: rmsprop | sgd')
cmd:option('--lrate', 1e-3, 'learning rate')
cmd:option('--max_grad_norm', 0, 'gradient clip value')
cmd:option('--alpha', 0.03, 'coefficient of baseline term in the cost function')
cmd:option('--beta', 0.01, '')

cmd:option('--Gumbel_temp', 1.0, '')
cmd:option('--Gumbel_start', 5.0, '')
cmd:option('--Gumbel_endbatch', 100*75, '')

cmd:option('--eps_end', 0.1, '')
cmd:option('--eps_start', 1.0, '')
cmd:option('--eps_endbatch', 100*20, '')

cmd:option('--epochs', 100, 'the number of training epochs')
cmd:option('--nbatches', 100, 'the number of mini-batches in one epoch')
cmd:option('--batch_size', 100, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--nworker', 1, 'the number of threads used for training')
-- for rmsprop
cmd:option('--rmsprop_alpha', 0.97, 'parameter of RMSProp')
cmd:option('--rmsprop_eps', 1e-6, 'parameter of RMSProp')
--other
cmd:option('--save', '', 'file name to save the model')
cmd:option('--load', 'exp6_run1.t7', 'file name to load the model')
g_opts = cmd:parse(arg or {})
g_opts.games_config_path = 'lua/mazebase/config/exp'..g_opts.exp_id..'.lua'
g_logs={}
g_mazebase.init_game()
g_mazebase.init_vocab()
init_master()
g_mazebase.init_game()
g_init_model()
g_load_model()
--[[]]
batch = batch_init(g_opts.batch_size)
input = {}
comm = torch.Tensor(#batch * g_opts.nagents, g_opts.nsymbols_monitoring):fill(0)
active = batch_active(batch)
input[1] = batch_input_monitoring(batch, active, 1)
input[2] = batch_input(batch, active, 1)
input[3] = comm:clone()

out = g_model:forward(input)
comm = out[1]:clone()
csvigo.save({path = "comm"..g_opts.exp_id..".csv", data = torch.totable(comm)})
matching = batch_matching(batch)
csvigo.save({path = "matching"..g_opts.exp_id..".csv", data = torch.totable(matching:view(-1,1))})

batch = batch_init(g_opts.batch_size)
active = {}
     reward = {}
     input = {}
     action = {}
     symbol = {}
     Gumbel_noise ={}
     comm = {}
     comm_sz = g_opts.nsymbols_monitoring
    
    comm[0] = torch.Tensor(#batch * g_opts.nagents, comm_sz):fill(0)
    active[1] = batch_active(batch)
     oneshot_comm = batch_input_monitoring(batch, active[1], 1)

    -- play the games
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end

        input[t] = {}
        if g_opts.oneshot_comm == true then 
            input[t][1] = oneshot_comm:clone()
        else
            input[t][1] = batch_input_monitoring(batch, active[t], t)
        end
        input[t][2] = batch_input(batch, active[t], t)
        input[t][3] = comm[t-1]:clone()

         out = g_model:forward(input[t])
        action[t] = sample_multinomial(torch.exp(out[2]))

        comm[t] = out[1]:clone()
        
        batch_act(batch, action[t]:view(-1), active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t],t == g_opts.max_steps)
    end
     success = batch_success(batch)
    print(success:sum())

