
require'gnuplot'

local exp='exp_PosSplit_2a_1D_std00-model_epoch'
local f = torch.load(exp..'.t7')
g_logs ={}
g_logs [1] = f.log
g_logs_test ={}
g_logs_test [1] = f.log_test
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

reward_test = torch.rand(num_of_experiments, epochs)
success_test = torch.rand(num_of_experiments, epochs)

for i = 1, num_of_experiments do
	for n = 1, epochs do
		success_test[i][n] = g_logs_test[i][n].success
		reward_test[i][n] = g_logs_test[i][n].reward
	end
end

---------------------------------------------------

gnuplot.pngfigure(exp..'.png')
--gnuplot.pngfigure('exp_2bb_1D_std05_success'..'.png')

gnuplot.plot(
	{'training',x1,success[1],'with lines ls 1'},
	{'testing', x1,success_test[1],'with lines ls 2'}
	)
gnuplot.xlabel('epochs (1 epoch = 100 rmsprop iterations)')
gnuplot.ylabel('success')
gnuplot.plotflush()
--gnuplot.title('(id, global position), noise std = 2.0, dim256')
gnuplot.axis{0,'', 0, 1}
gnuplot.grid(true)
