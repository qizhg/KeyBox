
matching = csvread('matching_exp_2a_1D_std05.csv');
channelIn = csvread('channelIn_exp_2a_1D_std05.csv');
channelOut = csvread('channelOut_exp_2a_1D_std05.csv');


%num_nonmatching = sum(matching==1);
%num_matching = sum(matching==2);
channelIn_mathcing = channelIn(matching==1);
channelOut_mathcing= channelOut(matching==1);
channelIn_nonmathcing= channelIn(matching==2);
channelOut_nonmathcing= channelOut(matching==2);

% ct_matching = 0;
% ct_nonmatching = 0;
% for i = 1:100
%     if matching(i) == 1
%         ct_matching = ct_matching +1;
%         channelIn_mathcing(ct_matching) = comm(i,1);
%         channelOut_mathcing(ct_matching) = comm(i,2);
%     else
%         ct_nonmatching = ct_nonmatching +1;
%         channelIn_nonmathcing(ct_nonmatching) = comm(i,1);
%         channelOut_nonmathcing(ct_nonmatching) = comm(i,2);
%     end
% end
% 
figure;
x = linspace(-8, 8,1000);
y = 1./(1+exp(-x));
plot(x,y)
hold on
 
 
sz = 10;
scatter(channelIn_mathcing,channelOut_mathcing,sz,'MarkerEdgeColor','b',...
              'MarkerFaceColor','b',...
              'LineWidth',1.5)
hold on

scatter(channelIn_nonmathcing,channelOut_nonmathcing,sz,'MarkerEdgeColor','r',...
              'MarkerFaceColor','r',...
              'LineWidth',1.5)
xlabel('channel input')
xlim([-15 15])
title('(id, color), noise std = 0.5')