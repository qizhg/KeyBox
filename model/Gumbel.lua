require('nn')
require('nngraph')

function g_build_Gumbel(temp)
    local temp = temp or 1.0
    local Gumbel_noise = nn.Identity()()
    local logp =  nn.Identity()()
    g_modules['Gumbel_logp'] = logp.data.module
    local Gumbel_trick = nn.CAddTable()({Gumbel_noise, logp})
    local Gumbel_trick_temp = nn.MulConstant(1.0/temp)(Gumbel_trick)
    local Gumbel_SoftMax = nn.SoftMax()(Gumbel_trick_temp)
    local model = nn.gModule(
        {logp, Gumbel_noise},
        {Gumbel_SoftMax}
        )

    return model
end