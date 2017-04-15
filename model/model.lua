require('nn')
require('nngraph')
paths.dofile('ask.lua')



function g_init_model()
    model_id2name = {}
    model_name2id = {}
    model_id2name[2] = 'ask'
    model_name2id['ask'] = 2

    ask_modules = {}
    ask_shareList = {}
    ask_model = build_ask_model()
    ask_paramx, ask_paramdx = ask_model:getParameters()
    if g_opts.init_std > 0 then
        ask_paramx:normal(0, g_opts.init_std)
    end
    ask_bl_loss = nn.MSECriterion()
end
