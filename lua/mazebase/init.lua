-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

local mazebase = {}

paths.dofile('MazeBase.lua')
paths.dofile('OptsHelper.lua')
paths.dofile('GameFactory.lua')
paths.dofile('KeyBox.lua')
paths.dofile('batch.lua')

local function init_game_opts()
    local games = {}
    local helpers = {}
    games.KeyBox = KeyBox
    helpers.KeyBox = OptsHelper
    g_factory = GameFactory(g_opts,g_vocab,games,helpers)
    return games, helpers
end

function mazebase.init_vocab()
    local function vocab_add(word)
        if g_vocab[word] == nil then
            local ind = g_opts.nwords + 1
            g_opts.nwords = g_opts.nwords + 1
            g_vocab[word] = ind
            g_ivocab[ind] = word
        end
    end
    g_vocab = {}
    g_ivocab = {}
    g_ivocabx = {}
    g_ivocaby = {}
    g_opts.nwords = 0

    -- general
    vocab_add('nil')
    vocab_add('empty')
    vocab_add('agent')
    for i = 1, 5 do
        vocab_add('agent' .. i)
    end
    vocab_add('goal')
    for i = 1, 10 do
        vocab_add('goal' .. i)
        vocab_add('obj' .. i)
        vocab_add('reward' .. i)
    end
    vocab_add('info')
    vocab_add('block')
    vocab_add('corner')
    vocab_add('water')
    vocab_add('visited')
    vocab_add('crumb')
    vocab_add('left')
    vocab_add('right')
    vocab_add('top')
    vocab_add('bottom')
    vocab_add('if')
    local mh = 12
    local mw = 12
    g_opts.MH = mh
    g_opts.MW = mw
    for y = -mh, mh do
        for x = -mw, mw do
            local w = 'y' .. y .. 'x' .. x
            vocab_add(w)
            g_ivocabx[g_vocab[w]] = x
            g_ivocaby[g_vocab[w]] = y
        end
    end
    -- for KeyBox
    vocab_add('key')
    vocab_add('box')
    vocab_add('block')
    vocab_add('water')
    vocab_add('OnGround')
    vocab_add('PickedUp')
    for i = 1, 2 do
        vocab_add('BoxType' .. i)
    end
    for i = 1, 2 do
        vocab_add('id' .. i)
    end
    for i = 1, 2 do
        vocab_add('color' .. i)
    end
    

end

function mazebase.init_game()
    g_opts = dofile(g_opts.games_config_path)
    local games, helpers = init_game_opts()
end

function mazebase.new_game()
    if g_opts.game == nil or g_opts.game == '' then
        return g_factory:init_random_game()
    else
       return g_factory:init_game(g_opts.game)
    end
end

return mazebase
