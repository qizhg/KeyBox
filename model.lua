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

	if g_opts.model_split~=nil and g_opts.model_split == true then
		g_init_model_split()
	end

    g_model = g_build_model()
    g_paramx, g_paramdx = g_model:getParameters()

    if g_opts.init_std > 0 then
        g_paramx:normal(0, g_opts.init_std)
    end
    
    g_bl_loss = nn.MSECriterion()
end

function g_init_model_split()
	local model_table = g_build_model()
	g_model_monitoring = model_table[1]
	g_model_acting = model_table[2]

	g_paramx_monitoring, g_paramdx_monitoring = g_model_monitoring:getParameters()
	g_paramx_acting, g_paramdx_acting = g_model_acting:getParameters()

    if g_opts.init_std > 0 then
        g_paramx_monitoring:normal(0, g_opts.init_std)
        g_paramx_acting:normal(0, g_opts.init_std)

    end
    
    g_bl_loss = nn.MSECriterion()
end
