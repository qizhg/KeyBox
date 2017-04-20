require('nn')
require('nngraph')

function build_lstm(input, prev_hid, prev_cell, hidsz, inputsz)
    local pre_hid = {}
    table.insert(pre_hid, nn.Linear(inputsz, hidsz * 4)(input))
    table.insert(pre_hid, nn.Linear(hidsz, hidsz * 4)(prev_hid))
    local preactivations = nn.CAddTable()(pre_hid)
    -- gates
    local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * hidsz)(preactivations)
    local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)

    -- input
    local in_chunk = nn.Narrow(2, 3 * hidsz + 1, hidsz)(preactivations)
    local in_transform = nn.Tanh()(in_chunk)

    local in_gate = nn.Narrow(2, 1, hidsz)(all_gates)
    local forget_gate = nn.Narrow(2, hidsz + 1, hidsz)(all_gates)
    local out_gate = nn.Narrow(2, 2 * hidsz + 1, hidsz)(all_gates)

    -- previous cell state contribution
    local c_forget = nn.CMulTable()({forget_gate, prev_cell})
    -- input contribution
    local c_input = nn.CMulTable()({in_gate, in_transform})
    -- next cell state
    local cellstate = nn.CAddTable()({
      c_forget,
      c_input
    })
    local c_transform = nn.Tanh()(cellstate)
    local hidstate = nn.CMulTable()({out_gate, c_transform})
        
    return hidstate, cellstate
end