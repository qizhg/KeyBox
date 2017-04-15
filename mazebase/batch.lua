-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function batch_init(size, task_id)
    local batch = {}
    for i = 1, size do
        batch[i] = g_mazebase.new_game()
        batch[i].task_id = task_id
    end
    return batch
end

function batch_active(batch)
    local active = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        if g:is_active() then
            active[i] = 1
        end
    end
    return active:view(-1)
end

function batch_input_asker(batch, active)
    local input = torch.Tensor(#batch, g_opts.memsize, g_opts.max_attributes)
    input:fill(g_vocab['nil'])
    for i, g in pairs(batch) do
        if active[i] == 1 then
            g:to_sentence_asker(input[i])
        end
    end
    input:resize(#batch, g_opts.memsize * g_opts.max_attributes)
    return input --will be slots for memory
end

function batch_act_asker(batch, action, active)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            g:act(action[i][1])
        end
    end
end

function batch_reward(batch, active)
    local reward = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        if active[i] == 1 then
            reward[i] = g:get_reward()
        end
    end
    return reward:view(-1)
end

function batch_update(batch, active)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            g:update()
        end
    end
end

function batch_success(batch)
    local success = torch.Tensor(#batch):fill(0)
    for i, g in pairs(batch) do
        if g:is_success() then
            success[i] = 1
        end
    end
    return success
end