matching = csvread('matching_loc.csv');
comm = csvread('comm_loc.csv');

num_mathcing = sum(matching);
num_nonmatching = 1000 - num_mathcing;
pc1_mathcing = zeros(num_mathcing,1);
pc2_mathcing= zeros(num_mathcing,1);
pc1_nonmathcing= zeros(num_nonmatching,1);
pc2_nonmathcing= zeros(num_nonmatching,1);

ct_matching = 0;
ct_nonmatching = 0;
for i = 1:1000
    if matching(i) == 1
        ct_matching = ct_matching +1;
        pc1_mathcing(ct_matching) = comm(i,1);
        pc2_mathcing(ct_matching) = comm(i,2);
    else
        ct_nonmatching = ct_nonmatching +1;
        pc1_nonmathcing(ct_nonmatching) = comm(i,1);
        pc2_nonmathcing(ct_nonmatching) = comm(i,2);
    end
end

figure;
sz = 10;
scatter(pc1_mathcing,pc2_mathcing,sz,'MarkerEdgeColor','b',...
              'MarkerFaceColor','b',...
              'LineWidth',1.5)
hold on

scatter(pc1_nonmathcing,pc2_nonmathcing,sz,'MarkerEdgeColor','r',...
              'MarkerFaceColor','r',...
              'LineWidth',1.5)