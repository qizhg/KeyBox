local KeyBox, parent = torch.class('KeyBox', 'MazeBase')

function KeyBox:__init(opts, vocab)
    parent.__init(self, opts, vocab)

    if g_opts.training_testing then
        self:sample_loc()
    end

    self:add_key()
    self:add_box()
    self:add_default_items() -- agent
    self:add_toggle() --toggle action for the acting agent

    self.success_open_total = 0
    self.failure_open_total = 0
    self.success_open = 0
    self.failure_open = 0

    self.finished = false

end
function KeyBox:sample_loc()
    local pos
    if g_opts.training_testing == 1 then --training
        local index = torch.rand(1):mul(g_opts.num_training):ceil()[1]
        pos = g_opts.id2pos[index]
    else --testing
        local index = torch.rand(1):mul(g_opts.num_testing):ceil()[1]
        pos = g_opts.id2pos[index + g_opts.num_training]
    end
    g_opts.loc_keys = {}
    g_opts.loc_boxes = {}
    for i = 1, g_opts.n_keys do
        local y, x = index2yx(pos[i], g_opts.MW)
        g_opts.loc_keys[i] = {}
        g_opts.loc_keys[i].y = y
        g_opts.loc_keys[i].x = x
    end
    for i = 1, g_opts.n_boxes do
        local y, x = index2yx(pos[i+g_opts.n_keys], g_opts.MW)
        g_opts.loc_boxes[i] = {}
        g_opts.loc_boxes[i].y = y
        g_opts.loc_boxes[i].x = x
    end

end

function KeyBox:add_key()
    --attr: id, color, postion, status
    local id = g_opts.id_keys or torch.randperm(g_opts.n_keys) --the order of adding keys
    self.color_keys = g_opts.color_keys or torch.randperm(g_opts.n_color_keys) --color_keys[j] is the color of the key with id=j
    for i = 1, g_opts.n_keys do
        if g_opts.loc_keys then
            self:place_item({
                type = 'key',
                id = 'id'..id[i],
                color = 'color'..self.color_keys[id[i]],
                status = 'OnGround'}, g_opts.loc_keys[i].y,g_opts.loc_keys[i].x)
        else
            self:place_item_rand({
            	type = 'key',
            	id = 'id'..id[i],
            	color='color'..self.color_keys[id[i]],
            	status='OnGround'})
        end
    end
end
function KeyBox:add_box()
    --attr: id, color, postion, status
    local id = g_opts.id_boxes or torch.randperm(g_opts.n_boxes)  --the order of adding boxes
    self.color_boxex = g_opts.color_boxes or torch.randperm(g_opts.n_color_boxes)  --color_boxex[j] is the color of the box with id=j
    self.boxType = torch.Tensor(g_opts.n_boxes):fill(2) --color_boxType[j] is the type of the box with id=j
    if g_opts.status_boxes == 'all' then
        self.boxType:fill(1) --all valuable
    elseif g_opts.status_boxes == 'one' then
        self.boxType[torch.random(g_opts.n_boxes)] = 1
    else
        error('wrong box status')
    end

    self.n_goal_boxes = self.boxType:eq(1):sum()

    for i = 1, g_opts.n_boxes do
        if g_opts.loc_boxes then
            self:place_item({
            	type = 'box',
            	id = 'id'..id[i],
            	color='color'..self.color_boxex[id[i]],
            	status='BoxType'..self.boxType[i]}, g_opts.loc_boxes[i].y,g_opts.loc_boxes[i].x)
        else
            self:place_item_rand({
            	type = 'box',
            	id = 'id'..id[i],
            	color='color'..self.color_boxex[id[i]],
            	status='BoxType'..self.boxType[i]})
        end
    end
end
function KeyBox:get_matching_label()
	local color_key_sorted, sorting_index = torch.sort(self.color_key)
	local mathcing_string = ""
	for i = 1, g_opts.n_boxes do
       mathcing_string = mathcing_string..i..'-'..self.color_box[sorting_index[i]]..' '
    end
    return g_opts.matchingstring2id[mathcing_string]
end
function KeyBox:add_toggle()
    --self.agent = self:place_item_rand({type = 'agent'})
    --self.agent = self:place_item({type = 'agent'},3,3)
    self.agent:add_action('toggle',
        function(self) --self for agent
            local l = self.map.items[self.loc.y][self.loc.x]
            if #l == 1 then --only agent
                --do nothing
            elseif #l==2 then --agent + (box or key)
                local agent, the_second
                for i = 1, #l do
                    if l[i].attr.type == 'agent' then
                        agent = l[i]
                    else
                        the_second = l[i]
                    end
                end
                if the_second.attr.type == 'key' then
                    if the_second.attr.status == 'OnGround' then
                        the_second.attr.status = 'PickedUp'
                    else
                        the_second.attr.status = 'OnGround'
                    end
                end
            elseif #l==3 then --agent + key(pickedup) + (key or box)
                local agent, key_pickedup, the_third
                for i = 1, #l do
                    if l[i].attr.type == 'agent' then
                        agent = l[i]
                    elseif l[i].attr.type == 'key' and l[i].attr.status == 'PickedUp' then
                        key_pickedup = l[i]
                    else
                        the_third = l[i]
                    end
                end
                if the_third.attr.type == 'key' then
                    --do nonthing
                elseif key_pickedup.attr.id ~= the_third.attr.id then
                    self.maze.finished = true
                else --open the box
                    if the_third.attr.status == 'BoxType'..1 then
                        self.maze.success_open = 1
                        self.maze.success_open_total = self.maze.success_open_total +  1
                    end
                    self.maze:remove_item(key_pickedup)
                    self.maze:remove_item(the_third)

                end
            end
        end)
end

function KeyBox:update()
    parent.update(self) -- t = t+1

    --picked up items follow
    for i = 1, #self.items do
        local e = self.items[i]
        if e.attr.status == 'PickedUp' then
            self.map:remove_item(e)
            e.loc.y = self.agent.loc.y
            e.loc.x = self.agent.loc.x
            self.map:add_item(e)
        end
    end

    --finished (compare total_open vs total_type1)
    if self.n_goal_boxes == self.success_open_total then
        self.finished = true
    end
end


function KeyBox:get_reward()
    local reward = parent.get_reward(self)
    reward = reward - self.success_open * self.costs.success_open
    self.success_open = 0
    return reward
end

function KeyBox:to_sentence_item(e, sentence, visibile_attr)
    local s = e:to_sentence_visible(self.agent.loc.y, self.agent.loc.x, visibile_attr)
    for i = 1, #s do
        sentence[i] = self.vocab[s[i]]
    end
end
function KeyBox:to_sentence_item_monitoring(e, sentence, visibile_attr)
    local s = e:to_sentence_visible(0, 0, visibile_attr)
    for i = 1, #s do
        sentence[i] = self.vocab[s[i]]
    end
end


-- representation for MemNN
function KeyBox:to_sentence(sentence)
    local visibile_attr = g_opts.visibile_attr
    local count=0
    local sentence = sentence or torch.Tensor(#self.items, self.max_attributes):fill(self.vocab['nil'])
    for i = 1, #self.items do
        if not self.items[i].attr._invisible then
            count= count + 1
            if count > sentence:size(1) then error('increase memsize!') end
            self:to_sentence_item(self.items[i], sentence[count], visibile_attr)
        end
    end
    return sentence
end

-- representation for MemNN
function KeyBox:to_sentence_monitoring(sentence)
    local visibile_attr = g_opts.visibile_attr_monitoring
    local count=0
    local sentence = sentence or torch.Tensor(#self.items, self.max_attributes):fill(self.vocab['nil'])
    for i = 1, #self.items do
        if self.items[i].attr.type ~= 'agent' then
            count= count + 1
            if count > sentence:size(1) then error('increase memsize!') end
            self:to_sentence_item_monitoring(self.items[i], sentence[count], visibile_attr)
        end
    end
    return sentence
end

-- onehot representation for MLP model
function KeyBox:to_map_onehot(sentence)
    local visibile_attr = g_opts.visibile_attr
    local count = 0
    local c = 0
    for _, e in pairs(self.items) do
        if e.attr.type ~='agent' then
            local d
            local tofar = false
            if e.loc then
                local dy = e.loc.y - self.agent.loc.y + torch.ceil(g_opts.conv_sz/2)
                local dx = e.loc.x - self.agent.loc.x + torch.ceil(g_opts.conv_sz/2)
                if dx > g_opts.conv_sz or dy > g_opts.conv_sz or dx < 1 or dy < 1 then
                    tofar = true
                end
                d = (dy - 1) * g_opts.conv_sz + dx - 1
            else
                c = c + 1
                d = g_opts.conv_sz * g_opts.conv_sz + c - 1
            end
            if not tofar then
                local s = e:to_sentence_visible(self.agent.loc.y, self.agent.loc.x, visibile_attr)
                for i = 1, #s do
                    count = count + 1
                    if count > sentence:size(1) then error('increase memsize!') end
                    sentence[count] = self.vocab[s[i]] + d * self.nwords
                end
            end
        end
    end
end

-- onehot representation for MLP model
function KeyBox:to_map_onehot_monitoring(sentence)
    local visibile_attr = g_opts.visibile_attr_monitoring
    
    if g_opts.loc_monitoring == true then
        local count = 0
        local c = 0
        local ref_y = self.agent.loc.y
        local ref_x = self.agent.loc.x
        if g_opts.actingloc_monitoring == false then
        	ref_y = 1
        	ref_x = 1
        end
        for _, e in pairs(self.items) do
            if e.attr.type ~='agent' then
                local d
                local tofar = false
                if e.loc then
                    local dy = e.loc.y - ref_y + torch.ceil(g_opts.conv_sz/2)
                    local dx = e.loc.x - ref_x + torch.ceil(g_opts.conv_sz/2)
                    if dx > g_opts.conv_sz or dy > g_opts.conv_sz or dx < 1 or dy < 1 then
                        tofar = true
                    end
                    d = (dy - 1) * g_opts.conv_sz + dx - 1
                else
                    c = c + 1
                    d = g_opts.conv_sz * g_opts.conv_sz + c - 1
                end
                if not tofar then
                    local s = e:to_sentence_visible(ref_y, ref_x, visibile_attr)
                    for i = 1, #s do
                        count = count + 1
                        if count > sentence:size(1) then error('increase memsize!') end
                        sentence[count] = self.vocab[s[i] ] + d * self.nwords
                    end
                end
            end
        end

    else
        local count = 0
        local c = 0
        for _, e in pairs(self.items) do
            if e.attr.type ~='agent' then
                c = c + 1
                local s = e:to_sentence_visible(self.agent.loc.y, self.agent.loc.x, visibile_attr)
                for i = 1, #s do
                    count = count + 1
                    if count > sentence:size(1) then error('increase memsize!') end
                    sentence[count] = self.vocab[ s[i] ] + (c-1) * self.nwords
                end
            end
        end
    end



end

function KeyBox:is_success()
    if self.n_goal_boxes == self.success_open_total then
        return true
    else
        return false
    end
end

function KeyBox:get_action_label()
    local x = self.agent.loc.x
    local y = self.agent.loc.y
    local l = self.map.items[y][x]
    local res = 5
    if #l == 1 then --go to key
        local dst = self.key.loc
        if y > dst.y then
            res = 1
        elseif y < dst.y then
            res = 2
        elseif x > dst.x then
            res = 3
        elseif x < dst.x then
            res = 4
        else
            --
        end
    elseif #l == 2 then
        local agent, the_second
        for i = 1, #l do
            if l[i].attr.type == 'agent' then
                agent = l[i]
            else
                the_second = l[i]
            end
        end
        local dst
        if the_second==nil then
            for i = 1, #l do
                print(l[i].attr)
            end
        end
        if the_second.attr.type == 'box'then
            dst = self.key.loc
        elseif self.key.attr.status == 'OnGround' then
            res = 6
        elseif self.key.attr.status == 'PickedUp' then
            dst = self.box.loc
        end
        if dst~= nil then
            if y > dst.y then
            res = 1
            elseif y < dst.y then
                res = 2
            elseif x > dst.x then
                res = 3
            elseif x < dst.x then
                res = 4
            else
                --
            end
        end

    elseif #l == 3 then
        res = 6
    else
        print('!!!!!!!!!!!!!!!!!!!')
    end

    return res
end
