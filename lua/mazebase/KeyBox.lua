local KeyBox, parent = torch.class('KeyBox', 'MazeBase')

function KeyBox:__init(opts, vocab)
    parent.__init(self, opts, vocab)

    self.n_keyboxpairs = opts.n_keyboxpairs
    self.n_colors = opts.n_colors
    
    self:add_default_items() -- blocks, waters
    self:add_toggle() --toggle action for the acting agent
    self:add_key()
    self:add_box()
    

    self.success_open_total = 0
    self.failure_open_total = 0
    self.success_open = 0
    self.failure_open = 0

    self.finished = false
end
function KeyBox:add_test()
end

function KeyBox:add_key()
    --attr: id, color, postion, status
    local id = torch.randperm(self.n_keyboxpairs)
    local color = torch.randperm(self.n_colors)
    for i = 1, self.n_keyboxpairs do 
        self.key = self:place_item_rand({type = 'key', id = 'id'..id[i], color='color'..color[i], status='OnGround'}) 
    end
end
function KeyBox:add_box()
    --attr: id, color, postion, status
    local id = torch.randperm(self.n_keyboxpairs)
    local color = torch.randperm(self.n_colors)
    local boxType = torch.Tensor(self.n_keyboxpairs):type(g_opts.dtype):fill(2)
    if g_opts.boxstatus == 'all' then
        boxType:fill(1) --all valuable
    elseif g_opts.boxstatus == 'one' then
        boxType[torch.random(self.n_keyboxpairs)] = 1
    else
        error('wrong box status')
    end

    self.n_goal_boxes = boxType:eq(1):sum()

    for i = 1, self.n_keyboxpairs do 
        self.box = self:place_item_rand({type = 'box', id = 'id'..id[i], color='color'..color[i], status='BoxType'..boxType[i]}) 

    end
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


function KeyBox:to_sentence(sentence)
    local visibile_attr = g_opts.visibile_attr
    local count=0
    local sentence = sentence or torch.Tensor(#self.items, self.max_attributes):type(g_opts.dtype):fill(self.vocab['nil'])
    for i = 1, #self.items do
        if not self.items[i].attr._invisible then
            count= count + 1
            if count > sentence:size(1) then error('increase memsize!') end
            self:to_sentence_item(self.items[i], sentence[count], visibile_attr)
        end
    end
    return sentence
end

function KeyBox:to_sentence_monitoring(sentence)
    local visibile_attr = g_opts.visibile_attr_monitoring
    local count=0
    local sentence = sentence or torch.Tensor(#self.items, self.max_attributes):type(g_opts.dtype):fill(self.vocab['nil'])
    for i = 1, #self.items do
        if not self.items[i].attr._invisible then
            count= count + 1
            if count > sentence:size(1) then error('increase memsize!') end
            self:to_sentence_item(self.items[i], sentence[count], visibile_attr)
        end
    end
    return sentence
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
