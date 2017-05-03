-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('nn')
require('nngraph')
paths.dofile('model/Gumbel.lua')
local model_file = 'model/model_'..g_opts.model..'_'..g_opts.model_id..'.lua'
paths.dofile(model_file)


function g_init_model()
    g_model_target = g_build_model()
    g_paramx_target, g_paramdx_target = g_model_target:getParameters()

    g_model = g_build_model()
    g_paramx, g_paramdx = g_model:getParameters()

    if g_opts.init_std > 0 then
        g_paramx:normal(0, g_opts.init_std)
    end
    g_paramx_target:copy(g_paramx)
    
    g_bl_loss = nn.MSECriterion()
end
