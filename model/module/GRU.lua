require('nn')
require('nngraph')

function build_GRU(input, prev_hid, hidsz, inputsz)
    local i2h_update_gate = nn.Linear(inputsz, hidsz)(input)
    local h2h_update_gate  = nn.Linear(hidsz, hidsz)(prev_hid)
    local update_gate = nn.Sigmoid()(nn.CAddTable()({ i2h_update_gate, h2h_update_gate }))

    local i2h_reset_gate = nn.Linear(inputsz, hidsz)(input)
    local h2h_reset_gate  = nn.Linear(hidsz, hidsz)(prev_hid)
    local reset_gate = nn.Sigmoid()(nn.CAddTable()({ i2h_reset_gate, h2h_reset_gate }))

    local gated_hidden = nn.CMulTable()({ reset_gate, prev_hid })
    local p2 = nn.Linear(hidsz, hidsz)(gated_hidden)
    local p1 = nn.Linear(inputsz, hidsz)(input)

    local hidden_candidate = nn.Tanh()(nn.CAddTable()({ p1, p2 }))

    local zh = nn.CMulTable()({ update_gate, hidden_candidate })
    local zhm1 = nn.CMulTable()({ nn.AddConstant(1, false)(nn.MulConstant(-1, false)(update_gate)), prev_hid })
    local next_h = nn.CAddTable()({ zh, zhm1 })

    return next_h


end