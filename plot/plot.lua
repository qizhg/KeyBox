
require'gnuplot'

local exp_id=4
local f = torch.load('exp'..exp_id..'_run1.t7')
g_logs ={}
g_logs [1] = f.log
epochs = #g_logs[1]

num_of_experiments = #g_logs
x1 = torch.rand(epochs)
for n = 1, epochs do
	x1[n]=n
end

reward = torch.rand(num_of_experiments, epochs)
success = torch.rand(num_of_experiments, epochs)
for i = 1, num_of_experiments do
	for n = 1, epochs do
		success[i][n] = g_logs[i][n].success
		reward[i][n] = g_logs[i][n].reward
	end
end

---------------------------------------------------

gnuplot.pngfigure('reward_exp'..exp_id..'.png')
gnuplot.plot(
	{'run 1',x1,reward[1],'with lines ls 1'}
	--{'run 2',x1,reward[2],'with lines ls 2'},
	--{'run 3',x1,reward[3],'with lines ls 3'}
	--{'run 4',x1,reward[4],'with lines ls 4'},
	--{'run 5',x1,reward[5],'with lines ls 5'}
	)
gnuplot.xlabel('epochs(1 epoch = 100 rmsprop iterations)')
gnuplot.ylabel('reward')
gnuplot.plotflush()


gnuplot.pngfigure('success_exp'..exp_id..'.png')
gnuplot.plot(
	{'run 1',x1,success[1],'with lines ls 1'}
	--{'run 2',x1,success[2],'with lines ls 2'},
	--{'run 3',x1,success[3],'with lines ls 3'}
	--{'run 4',x1,success[4],'with lines ls 4'},
	--{'run 5',x1,success[5],'with lines ls 5'}
	)
gnuplot.xlabel('epochs (1 epoch = 100 rmsprop iterations)')
gnuplot.ylabel('success')
gnuplot.plotflush()
