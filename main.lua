-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

package.path = package.path .. ';lua/?/init.lua'
g_mazebase = require('mazebase')

local function init_master()
    require('xlua')
    paths.dofile('util.lua')
    paths.dofile('model.lua')
    paths.dofile('train.lua')
    torch.setdefaulttensortype('torch.FloatTensor')
end

local function init_worker()
    require('xlua')
    paths.dofile('util.lua')
    torch.setdefaulttensortype('torch.FloatTensor')
end

local function init_threads()
    print('starting ' .. g_opts.nworker .. ' workers')
    local threads = require('threads')
    threads.Threads.serialization('threads.sharedserialize')
    local workers = threads.Threads(g_opts.nworker, init_worker)
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

local cmd = torch.CmdLine()
-- model parameters
cmd:option('--hidsz', 50, 'the size of the internal state vector')
cmd:option('--nonlin', 'relu', 'non-linearity type: tanh | relu | none')
cmd:option('--model', 'Recurrent', 'model type of the acting agent: FF | Recurrent')
cmd:option('--model_monitoring', 'FF', 'model type of the monitoring agent: FF | Recurrent')
cmd:option('--init_std', 0.2, 'STD of initial weights')
cmd:option('--max_attributes', 6, 'maximum number of attributes of each item')
cmd:option('--memsize', 20, 'size of the memory in MemNN')
cmd:option('--nhop', 1, 'the number of hops in MemNN')
-- game parameters
cmd:option('--nagents', 1, 'the number of acting agents')
cmd:option('--nactions', 6, 'the number of agent actions')
cmd:option('--max_steps', 30, 'force to end the game after this many steps')
cmd:option('--games_config_path', 'lua/mazebase/config/keybox5.lua', 'configuration file for games')
-- training parameters
cmd:option('--optim', 'rmsprop', 'optimization method: rmsprop | sgd')
cmd:option('--lrate', 5e-4, 'learning rate')
cmd:option('--max_grad_norm', 0, 'gradient clip value')
cmd:option('--alpha', 0.03, 'coefficient of baseline term in the cost function')
cmd:option('--beta', 0.05, '')
cmd:option('--epochs', 100, 'the number of training epochs')
cmd:option('--nbatches', 100, 'the number of mini-batches in one epoch')
cmd:option('--batch_size', 128, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--nworker', 1, 'the number of threads used for training')
cmd:option('--gpu', 0, '0 | 1')
-- for rmsprop
cmd:option('--rmsprop_alpha', 0.97, 'parameter of RMSProp')
cmd:option('--rmsprop_eps', 1e-6, 'parameter of RMSProp')
--other
cmd:option('--save', '', 'file name to save the model')
cmd:option('--load', '', 'file name to load the model')
g_opts = cmd:parse(arg or {})

if g_opts.gpu == 1 then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(1)
    print(cutorch.getDeviceProperties(1))

    g_opts.batch_size = g_opts.batch_size * g_opts.nworker
    g_opts.nworker = 1

    g_opts.dtype = 'torch.CudaTensor'
else
    g_opts.dtype = 'torch.FloatTensor'

end
print(g_opts)
g_mazebase.init_vocab()
g_mazebase.init_game()
init_master()

if g_opts.nworker > 1 then
    g_workers = init_threads()
end

g_logs={}
for i = 1, 3 do
    g_log = {}
    if g_opts.optim == 'rmsprop' then g_rmsprop_state = {} end
    g_init_model()
    g_load_model()
    g_mazebase.init_game()

    train(g_opts.epochs)
    g_opts.save ='case5_nocomm_id'
    g_save_model()
    g_logs[i] = g_log
end
g_opts.save ='glogs_case5_nocomm_id'
g_save_glogs()

--g_disp = require('display')
--test()
