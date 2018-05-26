clear; 
clc

load('R_matrix.mat');
load('spam_users.mat');

idx = randperm(size(R,1));
R = R(idx,:);

autoenc = trainAutoencoder(R,500);
      
      
R_reconstructed = predict(autoenc,R);

spam = spam_users(idx);
MSE = abs(sum(R - R_reconstructed,2));
predicted_spam = MSE > 125;


true_pos = sum(spam == 1 & predicted_spam == 1);
pred_pos = sum(predicted_spam);
actual_pos = sum(spam);

precision = true_pos/pred_pos
recall = true_pos/actual_pos


spam_data = [spam MSE];
idx_spam = find(spam_data(:,1) == 1);
spam_data = [idx_spam,spam_data(idx_spam,2)];

figure(1)
plot(MSE,'o');
hold all;
scatter(spam_data(:,1),spam_data(:,2));
title('Spam User Detection');
ylabel('RMSE');
xlabel('User ID');
legend('Non-Spam Users','Spam Users');
hold off;

% figure(1)
% plot(MSE,'o');
% hold all;
% plot(MSE.*spam,'o');
% title('Spam User Detection');
% hold off;