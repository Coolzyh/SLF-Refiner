%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize environment parameters for generating training data and
% testing data
% Generate data for Deep Learning with uncalibrated alpha and b              %
% theta  :  SLF image     
% W  :  weight 
% b  :  bais.       alpha  :  path loss exponent
% nodepos  :  coordinates of UWB node
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INPUTS %
%%%%%%%%%%
M  = 4;              %number of mobile RF nodes
Tw = 6;              %number of waypoints for each node to gather data from
sig_epsilon_range = [0.3,1,3];    % SNR is about 30,40,50
alpha_range = [0.9 1];     % path loss exponent
b_range = [90 100];        %global avg TX "level" (db)

sig_theta_range = [0.01,0.03,0.09];
kappa = 0.21;

areabounds = [0 5 0 5];    %size of the node traverse area (m) [xmin xmax ymin ymax]
imgorg = [0.5 0.5];        %lower left corner of the image [x y] (m)
imgdims = [40 40];         %image size in pixels
pixelsize = 0.1;           %l/w of each square pixel (m)
dBwhite = 1;              %dB atten of a white pixel       
wallcount_vec = [0 2 0 2 1 3]; %min/max for vertical , horizontal and square walls

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%preallocate data 
K = imgdims(1)*imgdims(2);         %pixel count
N_Link = M*(M-1);                  %Number of UWB Links 
N_Tw = Tw^2;                       %number of UWB data per link
x_1=0; y_1=0; 
x_2 = imgdims(2)*pixelsize+2*imgorg(1);      %UWB node range
y_2 = imgdims(1)*pixelsize+2*imgorg(2);
x_distance = (imgdims(2)-1)*pixelsize/(Tw-1);  %distance of two position of per node
y_distance = (imgdims(1)-1)*pixelsize/(Tw-1);
x_org = imgorg(1)+0.5*pixelsize;         %coordinate of first UWB node
y_org = imgorg(2)+0.5*pixelsize;

% all coordinates of each node
pos_node = NaN(M,Tw);
for Twi = 1:Tw          %Tw point per node
    pos_node(1,Twi) = x_org + (Twi-1)*x_distance + 1j * y_1;
    pos_node(2,Twi) = x_2 + 1j * (y_org + (Twi-1)*y_distance);
    pos_node(3,Twi) = x_org + (Twi-1)*x_distance + 1j * y_2;
    pos_node(4,Twi) = x_1 + 1j * (y_org + (Twi-1)*y_distance);
end 

%store the data
nodepos = NaN(2,N_Tw,N_Link);     
d = NaN(N_Link*N_Tw,1);                 %distance of UWB node for each measurements
W = NaN(N_Link*N_Tw,K);                 %weight of slf
W_ENR = NaN(N_Link*N_Tw,K);             %weight of slf for mismatched experiments
Z = zeros(N_Link*N_Tw,N_Link);          %weight of bias

%node position calculation
N_Linki = 0;
for i1 = 1:M
    for i2 = 1:M
        if i1 ~= i2
            N_Linki = N_Linki + 1;
            for Twi = 1:Tw
                nodepos(1,(Twi-1)*Tw+1:Twi*Tw,N_Linki)=pos_node(i1,Twi)*ones(1,Tw);
                nodepos(2,(Twi-1)*Tw+1:Twi*Tw,N_Linki)=pos_node(i2,:);
            end 
        end
    end
end 

%generate weight and distance
for N_Linki = 1:N_Link
    W((N_Linki-1)*N_Tw+1:N_Linki*N_Tw,:) = ...
        f_gen_Wi_ellipse(nodepos(:,:,N_Linki),pixelsize,imgorg,imgdims,1,2);
        % Inverse Area Elliptical Model
%     W((N_Linki-1)*N_Tw+1:N_Linki*N_Tw,:) = ...
%         f_gen_Wi(nodepos(:,:,N_Linki),pixelsize,imgorg,imgdims,1,2);     % Normalized Ellipse Model
    di = abs( nodepos(1,:,N_Linki) - nodepos(2,:,N_Linki) )';              %link lengths
    d((N_Linki-1)*N_Tw+1:N_Linki*N_Tw,1) = 20 * log10( di );
    Z((N_Linki-1)*N_Tw+1:N_Linki*N_Tw,N_Linki) = ones(N_Tw,1);
end 
    

C_theta = f_gen_C(imgdims,pixelsize,kappa);   % spatial correlated Gaussian random field
R_theta = chol(C_theta);

