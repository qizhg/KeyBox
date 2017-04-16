local KeyBox, parent = torch.class('KeyBox', 'MazeBase')

function KeyBox:__init(opts, vocab)
    parent.__init(self, opts, vocab)

    self.n_keys = opts.n_keys
    self.n_colors = opts.n_colors
    self.n_boxes =  opts.n_boxes
    self.n_boxTypes =  opts.n_boxTypes

    
    self:add_default_items() -- blocks, waters
    self:add_asker()
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
    local id = torch.randperm(self.n_keys)
    local color = torch.Tensor(self.n_keys):random(1, self.n_colors)
    for i = 1, self.n_keys do 
        self:place_item_rand({type = 'key', id = 'id'..id[i], color='color'..color[i], status='OnGround'}) 
        --self:place_item({type = 'key', id = 'id'..id[i], color='color'..color[i], status='OnGround'},
        --                     3, 2) 

    end
end
function KeyBox:add_box()
    --attr: id, color, postion, status
    local id = torch.randperm(self.n_boxes)
    local color = torch.Tensor(self.n_boxes):random(1, self.n_colors)
    local boxType = torch.Tensor(self.n_boxes):random(1, self.n_boxTypes)
    if boxType:eq(1):sum() == 0 then
        boxType[torch.random(self.n_boxes)] = 1
    end
    self.n_goal_boxes = boxType:eq(1):sum()

    for i = 1, self.n_boxes do 
        self:place_item_rand({type = 'box', id = 'id'..id[i], color='color'..color[i], status='BoxType'..boxType[i]}) 
        --self:place_item({type = 'box', id = 'id'..id[i], color='color'..color[i], status='BoxType'..boxType[i]},
        --                    3, 4)
    end
end
function KeyBox:add_asker()
    self.agent = self:place_item_rand({type = 'agent'})
    --self.agent = self:place_item({type = 'agent'},3,3)
    self.agent:add_action('toggle',
        function(self) --self for agent
            local l = self.map.items[self.loc.y][self.loc.x]
            if #l == 1 then --only agent 
                --do nothing
            elseif #l==2 then --agnet + (box or key)
                for i = 1, #l do
                    if l[i].attr.type == 'key' then
                        if l[i].attr.status == 'OnGround' then 
                            l[i].attr.status = 'PickedUp'
                        else 
                            l[i].attr.status = 'OnGround'
                        end
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
                    --do nothing
                else --open the box
                    if the_third.attr.status == 'BoxType'..1 then
                        self.maze.success_open = 1
                        self.maze.success_open_total = self.maze.success_open_total +  1
                    else
                        self.maze.failure_open = 1
                        self.maze.failure_open_total = self.maze.failure_open_total + 1
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
        --print('!!!!!!!!')
    end
end

function KeyBox:get_reward()
    local reward = parent.get_reward(self)
    reward = reward - self.success_open * self.costs.success_open
    reward = reward - self.failure_open * self.costs.failure_open
    self.success_open = 0
    self.failure_open = 0
    return reward
end

function KeyBox:to_sentence_visible(e, sentence, visibile_attr)
    local s = e:to_sentence_visible(visibile_attr)
    for i = 1, #s do
        sentence[i] = self.vocab[s[i]]
    end
end

-- Tensor representation that can be feed to a mem odel
function KeyBox:to_sentence_asker(sentence)
    local visibile_attr = {'type', 'color', 'loc', 'id', 'status'}
    local count=0
    local sentence = sentence or torch.Tensor(#self.items, self.max_attributes):fill(self.vocab['nil'])
    for i = 1, #self.items do
        if not self.items[i].attr._invisible then
            count= count + 1
            if count > sentence:size(1) then error('increase memsize!') end
            self:to_sentence_visible(self.items[i], sentence[count], visibile_attr)
        end
    end
    return sentence
end

function KeyBox:is_success()
    if self.finished == true then
        return true
    else
        return false
    end
end
