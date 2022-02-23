%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Run this file after running Environment_P6_SNR30_40_50.m
% which initializes environment parameters for generating data
% Generate training/testing data for Deep Learning with uncalibrated alpha and b              %
% theta_vec  :  flatten SLF image     
% b  :  bias.       alpha  :  path loss exponent
% ab_norm  :  [b_norm, alpha_norm]
% sig_epsilon_class  :  noise level
% y  :  noisy RSS measurement input
% y_denoised  :  clean RSS measurement input (as target)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATE some RANDOM IMAGE %
% please set different random seed for training set and testing set
rng(2022);    % set random seed 
NI = 40000;                 %number of images
theta_vec = NaN(K,NI);             %real environment images
% theta_vec_ENR = NaN(K,NI);         %ENR estimated environment images
theta_vec_noise = NaN(K,NI);
alpha = NaN(1,NI);                        %path loss
b = NaN(N_Link,NI);                     %bias
y = NaN(N_Link*N_Tw,NI);                 %RSS measurements data
sig_epsilon_class = NaN(NI,1);           %noise level
sig_epsilon = NaN(NI,1);
sig_theta = NaN(NI,1);
epsilon = NaN(N_Link*N_Tw,NI);

% random sig_epsilon
for i = 1:NI
    sig_epsilon_index_rand = randperm(3);
    sig_epsilon_class(i) = sig_epsilon_index_rand(1)-1;   %class  0  1  2
    sig_epsilon(i) = sig_epsilon_range(sig_epsilon_index_rand(1));
    sig_theta(i) = sig_theta_range(sig_epsilon_index_rand(1));
end

% Generate training data
for N_img = 1:NI
    N_img
    b(:,N_img) = rand(N_Link,1)*(b_range(2)-b_range(1)) + b_range(1);  %N_img th bias
    alpha(1,N_img) = rand*(alpha_range(2)-alpha_range(1))+alpha_range(1);
    theta_img = f_gen_randSLFimg(imgdims(1),imgdims(2),dBwhite,wallcount_vec);
    theta_vec(:,N_img) = reshape(theta_img,K,1);
    theta_vec_noise(:,N_img) = theta_vec(:,N_img) + sig_theta(N_img)*R_theta'*randn(K,1);
    epsilon(:,N_img) = sig_epsilon(N_img)* randn(N_Link*N_Tw,1);
end 

y = Z*b - 2*W*theta_vec_noise - d*alpha + epsilon;
y(y<0) = 0;

y_denoise = Z*b - 2*W*theta_vec - d*alpha;
y_denoise(y_denoise<0) = 0;

% % ENR estimate SLF img
% lanpt1 = 0.5;
% lanpt2 = 5;
% Cmodel_inv = inv(C_theta); Gamma = chol(Cmodel_inv); Gamma(abs(Gamma)<1e-6) = 0;
% tic;
% parfor N_img = 1:NI
%     N_img
%     RSS_noise = y(:, N_img);
%     theta_en = Estimate_slf_ENR(K,N_Link,RSS_noise,Z,W,d,lanpt1,lanpt2,Gamma);
%     theta_vec_ENR(:,N_img) = reshape(theta_en,K,1);
% end
% toc;

% y_min = min(y_denoise,[],2);
% y_max = max(y_denoise,[],2);

% y_norm = (y-y_min)./(y_max-y_min);
% y_denoise_norm = (y_denoise-y_min)./(y_max-y_min);

b_norm = (b - b_range(1))/(b_range(2)-b_range(1));
alpha_norm = (alpha - alpha_range(1))/(alpha_range(2)-alpha_range(1));
ab_norm = [b_norm;alpha_norm];

RSS = transpose(y);
RSS_denoise = transpose(y_denoise);
slf_img = reshape(theta_vec, [imgdims(1), imgdims(2), NI]);
slf_img = permute(slf_img, [3, 1, 2]);
% slf_img_ENR = reshape(theta_vec_ENR, [imgdims(1), imgdims(2), NI]);
% slf_img_ENR = permute(slf_img_ENR, [3, 1, 2]);
ab_norm = transpose(ab_norm);

save('..\data\training_data.mat','RSS','RSS_denoise', 'slf_img', 'ab_norm', 'sig_epsilon_class');
