-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

function train_batch(num_batch)
    local stat = {}
    
    local csv = csvigo.load({path = "./comm1.csv", verbose = false, mode = "raw"})
    local input = torch.Tensor(csv)
    local Gumbel_noise = torch.rand(input:size(1), input:size(2)):log():neg():log():neg()
    local out = g_model:forward({input, Gumbel_noise})
    stat.bl_cost =  g_bl_loss:forward(out[1], input)
    stat.bl_count = 1
    local grad = g_bl_loss:backward(out[1], input)

    g_paramdx:zero()
    g_model:backward({input, Gumbel_noise}, {grad, out[2]:clone():zero()})

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
