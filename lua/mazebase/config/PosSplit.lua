
math.randomseed( os.time() )

local function has_value (tab, val)
    for index, value in ipairs(tab) do
        if value == val then
            return true
        end
    end

    return false
end

local function combs(k, n)
    if k*n == 0 then 
        local ret = {{}}
        return ret
    elseif k >= n then 
        local ret={}
        ret[1] = {}
        for i = 1, n do
            ret[1][1+#ret[1]] = i
        end
        return ret
    else
        local ret_case1 = combs(k, n-1)
        local ret_case2 = combs(k-1, n-1)
        for i = 1, #ret_case2 do
            ret_case2[i][1+#ret_case2[i]] = n
            ret_case1[1+#ret_case1] = ret_case2[i]
        end
        return ret_case1
    end
end

local function shuffleTable( t )
    local rand = math.random
    assert( t, "shuffleTable() expected a table, got nil" )
    local iterations = #t
    local j
    
    for i = iterations, 2, -1 do
        j = rand(i)
        t[i], t[j] = t[j], t[i]
    end
end

id2pos={}
local H = 4
local W = 4
local n_keys = 2
local n_boxes = 2
local position_comb = combs(n_keys+n_boxes,H*W)
local keybox_within_comb = combs(n_keys, n_keys+n_boxes)
for k, v in ipairs(position_comb) do 
	for kk, vv in ipairs(keybox_within_comb) do 
		local cur_key = 1
		local cur_box = 1
		local pos={}
		for item = 1, n_keys+n_boxes do 
			if has_value (vv, item) then --key
				pos[cur_key] = v[vv[cur_key]]
				cur_key = cur_key +1
			else
				pos[cur_box + n_keys] = v[item]
				cur_box = cur_box +1
			end
		end
		id2pos[#id2pos+1] = pos
	end
end

shuffleTable(id2pos)

f = {}
f.pos = id2pos
torch.save('PosSplit.t7', f)
