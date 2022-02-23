function r=Estimate_slf_ENR(K,N_Link,RSS_noise,Z,W,d,lanpt1,lanpt2,Gamma)
cvx_begin quiet
    variable theta_en(K,1) nonnegative
    variable b_en(N_Link,1) nonnegative
    variable alpha_en(1,1) nonnegative
    minimize norm(RSS_noise - Z*b_en + 2*W*theta_en + d*alpha_en) +...
        lanpt1 * norm(theta_en,1) + lanpt2 * norm(Gamma * theta_en)
    subject to 
        0 <= theta_en <=1;
        90 <= b_en <= 100;
        0.9 <= alpha_en <=1;
cvx_end
r=theta_en;
end