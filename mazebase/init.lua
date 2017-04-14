-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant 
-- of patent rights can be found in the PATENTS file in the same directory.

g_mazebase = {}

paths.dofile('MazeBase.lua')
paths.dofile('GameFactory.lua')
paths.dofile('OptsHelper.lua')
paths.dofile('batch.lua')
paths.dofile('KeyBox.lua')

local function init_game_opts()
    local games = {}
    local helpers = {}
    games.KeyBox = KeyBox
    helpers.KeyBox = OptsHelper
    g_factory = GameFactory(g_opts,g_vocab,games,helpers)
    return games, helpers
end

function g_mazebase.init_vocab()
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
    vocab_add('block')
    vocab_add('water')
    vocab_add('key')
    vocab_add('box')
    vocab_add('OnGround')
    vocab_add('PickedUp')
    for i = 1, 10 do
        vocab_add('color' .. i)
    end
    for i = 1, 10 do
        vocab_add('id' .. i)
    end
    for i = 1, 10 do
        vocab_add('BoxType' .. i)
    end

    for y = 1, 10 do
        for x = 1, 10 do
            local w = 'y' .. y .. 'x' .. x
            vocab_add(w)
        end
    end
end

function g_mazebase.init_game()
    g_opts = dofile(g_opts.games_config_path)
    local games, helpers = init_game_opts()
end

function g_mazebase.new_game()
    if g_opts.game == nil or g_opts.game == '' then
        return g_factory:init_random_game()
    else
       return g_factory:init_game(g_opts.game)
    end
end