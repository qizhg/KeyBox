-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function train_batch(num_batch)
    -- start a new episode
    local batch = batch_init(g_opts.batch_size)
    local active = {}
    local reward = {}
    local input = {}
    local action = {}
    local matching_label = batch_matching(batch):long()

    local stat = {}
    active[1] = batch_active(batch)
    input[1] = batch_input_mlp_monitoring(batch, active[1], 1)
    local out = g_model:forward(input[1])
    local NLLceriterion = nn.ClassNLLCriterion()
    local err = NLLceriterion:forward(out,matching_label)
    stat.bl_cost = err
    stat.bl_count = active[1]:sum()
    local grad = NLLceriterion:backward(out,matching_label)
    g_paramdx:zero()
    g_model:backward(input[1], grad)

    local _, pred = torch.max(out, 2)
    local success = pred:eq(matching_label):squeeze()

    
    for i, g in pairs(batch) do
        stat.reward = 0
        stat.success = (stat.success or 0) + success[i]
        stat.count = (stat.count or 0) + 1
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

function train_batch_thread(opts_orig, paramx_orig, num_batch)
    g_opts = opts_orig
    g_paramx:copy(paramx_orig)
    local stat = train_batch(num_batch)
    return g_paramdx, stat
end
