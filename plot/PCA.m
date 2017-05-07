matching_original = csvread('matching.csv');
comm_original = csvread('comm.csv');
[comm,ia,ic] = unique(comm_original,'rows');
matching = matching_original(ia);
comm_mean = mean(comm); 
comm_mean = repmat(comm_mean,16,1);
comm =  comm-comm_mean;
cov =  transpose(comm)*comm;
cov= cov /16.0;

[V,D] = eig(cov);
[D,I] = sort(diag(D),'descend');
V = V(:, I);

pc1 = V(:,1);
pc2 = V(:,2);
pc3 = V(:,3);
x = comm(1,:);
matching(1)
num_mathcing = sum(matching);
num_nonmatching = 16 - num_mathcing;
pc1_mathcing = zeros(num_mathcing,1);
pc2_mathcing= zeros(num_mathcing,1);
pc3_mathcing= zeros(num_mathcing,1);
pc1_nonmathcing= zeros(num_nonmatching,1);
pc2_nonmathcing= zeros(num_nonmatching,1);
pc3_nonmathcing= zeros(num_nonmatching,1);
ct_matching = 0;
ct_nonmatching = 0;
for i = 1:16
    if matching(i) == 1
        ct_matching = ct_matching +1;
        pc1_mathcing(ct_matching) = dot(pc1, comm(i,:));
        pc2_mathcing(ct_matching) = dot(pc2, comm(i,:));
        pc3_mathcing(ct_matching) = dot(pc3, comm(i,:));
    else
        ct_nonmatching = ct_nonmatching +1;
        pc1_nonmathcing(ct_nonmatching) = dot(pc1, comm(i,:));
        pc2_nonmathcing(ct_nonmatching) = dot(pc2, comm(i,:));
        pc3_nonmathcing(ct_nonmatching) = dot(pc3, comm(i,:));
    end
end

figure;
sz = 40;
scatter(pc1_mathcing,pc2_mathcing,sz,'MarkerEdgeColor','b',...
              'MarkerFaceColor','b',...
              'LineWidth',1.5)
hold on

scatter(pc1_nonmathcing,pc2_nonmathcing,sz,'MarkerEdgeColor','r',...
              'MarkerFaceColor','r',...
              'LineWidth',1.5)
    
figure;
sz = 40;
scatter3(pc1_mathcing,pc2_mathcing,pc3_mathcing,sz,'MarkerEdgeColor','b',...
              'MarkerFaceColor','b',...
              'LineWidth',1.5)
hold on

scatter3(pc1_nonmathcing,pc2_nonmathcing,pc3_nonmathcing,sz,'MarkerEdgeColor','r',...
              'MarkerFaceColor','r',...
              'LineWidth',1.5)
        
