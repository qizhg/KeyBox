require('nn')
require('nngraph')

local function nonlin()
    if g_opts.nonlin == 'tanh' then
        return nn.Tanh()
    elseif g_opts.nonlin == 'relu' then
        return nn.ReLU()
    elseif g_opts.nonlin == 'none' then
        return nn.Identity()
    else
        error('wrong nonlin')
    end
end

function LeNet(input)
    --local input = nn.Identity()() --a batch of images, 3 channels(RGB) x 32 x 32
    
    local conv1 = nn.SpatialConvolution(3, 6, 5, 5)(input)
    local pool1 = nn.SpatialMaxPooling(2, 2, 2, 2)(conv1)
    local non1 = nonlin()(pool1)

    local conv2 = nn.SpatialConvolution(6, 16, 5, 5)(non1)
    local pool2 = nn.SpatialMaxPooling(2, 2, 2, 2)(conv2)
    local non2 = nonlin()(pool2)

    local flat_view = nn.View(16*5*5):setNumInputDims(3)(non2)
    --local embedding = nonlin()(nn.Linear(16*5*5, 120)(flat_view))

    return flat_view
end