-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function test_batch()
	if g_opts.training_testing then g_opts.training_testing = 0 end
	local batch_size_test = 1000
    local batch = batch_init(batch_size_test)
    local active = {}
    local reward = {}
    local input = {}
    local action = {}

    -- play the games
    for t = 1, g_opts.max_steps do
        active[t] = batch_active(batch)
        if active[t]:sum() == 0 then break end

        input[t] = batch_input_mlp(batch, active[t], t)

        local out = g_model:forward(input[t])
        action[t] = sample_multinomial(torch.exp(out[1]))
        
        batch_act(batch, action[t]:view(-1), active[t])
        batch_update(batch, active[t])
        reward[t] = batch_reward(batch, active[t],t == g_opts.max_steps)
    end
    local success = batch_success(batch)

    -- increase difficulty if necessary
    if g_opts.curriculum == 1 then
        apply_curriculum(batch, success)
    end

    local stat = {}
    local R = torch.Tensor(batch_size_test * g_opts.nagents):zero()
    for t = g_opts.max_steps, 1, -1 do
        if active[t] ~= nil and active[t]:sum() > 0 then
            local out = g_model:forward(input[t])
            R:add(reward[t])
            R:cmul(active[t])
        end
    end

    R:resize(batch_size_test, g_opts.nagents)
    -- stat by game type
    for i, g in pairs(batch) do
        stat.reward = (stat.reward or 0) + R[i]:mean()
        stat.success = (stat.success or 0) + success[i]
        stat.count = (stat.count or 0) + 1
    end
    return stat
end
