-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function batch_init(size)
    local batch = {}
    for i = 1, size do
        batch[i] = g_mazebase.new_game()
    end
    return batch
end

function batch_input(batch, active, t)
    if g_opts.model == 'MLP_A3C' then
        return batch_input_mlp(batch, active, t)
    end

    active = active:view(#batch, g_opts.nagents)
    local input = torch.Tensor(#batch, g_opts.nagents, g_opts.memsize, g_opts.max_attributes)
    input:fill(g_vocab['nil'])
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                g:to_sentence(input[i][a])
            end
        end
    end
    input:resize(#batch * g_opts.nagents, g_opts.memsize * g_opts.max_attributes)
    return input
end

function batch_input_monitoring(batch, active, t)

    if g_opts.model == 'MLP_A3C' then
        return batch_input_mlp_monitoring(batch, active, t)
    end

    active = active:view(#batch, g_opts.nagents)
    local input = torch.Tensor(#batch, g_opts.nagents, g_opts.memsize, g_opts.max_attributes)
    input:fill(g_vocab['nil'])
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                g:to_sentence_monitoring(input[i][a])
            end
        end
    end
    input:resize(#batch * g_opts.nagents, g_opts.memsize * g_opts.max_attributes)
    return input
end

function batch_input_mlp(batch, active, t)
    -- total number of words in dictionary:
    local mapwords = g_opts.conv_sz*g_opts.conv_sz*g_opts.nwords
    local nilword = mapwords + 1
    active = active:view(#batch, g_opts.nagents)
    local input = torch.Tensor(#batch, g_opts.nagents, g_opts.memsize * g_opts.max_attributes)
    input:fill(nilword)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                g:to_map_onehot(input[i][a])
            end
        end
    end
    return input:view(#batch * g_opts.nagents, -1)
end

function batch_input_mlp_monitoring(batch, active, t)
    -- total number of words in dictionary:
    local mapwords = g_opts.conv_sz*g_opts.conv_sz*g_opts.nwords
    local nilword = mapwords + 1
    active = active:view(#batch, g_opts.nagents)
    local input = torch.Tensor(#batch, g_opts.nagents, g_opts.memsize * g_opts.max_attributes)
    input:fill(nilword)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                g:to_map_onehot_monitoring(input[i][a])
            end
        end
    end
    return input:view(#batch * g_opts.nagents, -1)
end

function batch_act(batch, action, active)
    active = active:view(#batch, g_opts.nagents)
    action = action:view(#batch, g_opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                g:act(action[i][a])
            end
        end
    end
end

function batch_reward(batch, active, is_last)
    active = active:view(#batch, g_opts.nagents)
    local reward = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                reward[i][a] = g:get_reward(is_last)
            end
        end
    end
    return reward:view(-1)
end

function batch_update(batch, active)
    active = active:view(#batch, g_opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, g_opts.nagents do
            g.agent = g.agents[a]
            if active[i][a] == 1 then
                g:update()
                break
            end
        end
    end
end

function batch_active(batch)
    local active = torch.Tensor(#batch, g_opts.nagents):zero()
    for i, g in pairs(batch) do
        if (not g.sv_on) and (not g.qa_on) then
            for a = 1, g_opts.nagents do
                g.agent = g.agents[a]
                if g:is_active() then
                    active[i][a] = 1
                end
            end
        end
    end
    return active:view(-1)
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
function batch_act_action_label(batch, active)
    local action_label = torch.LongTensor(#batch):fill(5)
    for i, g in pairs(batch) do
        if active[i] == 1 then
            action_label[i] = g:get_action_label()
        end
    end
    return action_label
end