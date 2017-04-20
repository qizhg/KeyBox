-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

require('nn')
require('nngraph')

if g_opts.model == 'FF' then   
    paths.dofile('model/model_FF.lua')
elseif g_opts.model == 'Recurrent' then
    paths.dofile('model/model_Recurrent.lua')
else
    error('wrong model name')
end

function g_init_model()
    g_model = g_build_model()
    g_paramx, g_paramdx = g_model:getParameters()
    if g_opts.init_std > 0 then
        g_paramx:normal(0, g_opts.init_std)
    end
    g_bl_loss = nn.MSECriterion()
end
