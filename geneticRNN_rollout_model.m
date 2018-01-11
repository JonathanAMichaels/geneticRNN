function net = geneticRNN_rollout_model(policyInitFunLocal, policyInitInputsLocal, policyInitInputsOptionalLocal, allMutationPower, allDecay1, allDecay2, weightDecay, seeds)

% net = geneticRNN_rollout_model(policyInitFunLocal, policyInitInputsLocal, policyInitInputsOptionalLocal, allMutationPower, allDecay1, allDecay2, weightDecay, seeds)
%
% This function rolls out the policy of a recurrent neural network using a sequence of random seeds
%
% INPUTS:
%
% policyInitFunLocal -- function handle for initializing the policy
%
% policyInitInputsLocal -- Inputs for the policy initialization function
%
% policyInitInputsOptionalLocal -- Optional inputs for the policy initialization function
%
% allMutationPower -- History of mutation power over all past generations
%
% allDecay1 -- History of decay parameter 1 over all past generations
%
% allDecay2 -- History of decay parameter 2 over all past generations
%
% weightDecay -- Whether or not to perform weight decay (logical)
%
% seeds - Vector of random generator seeds to rollout
%
%
% OUTPUTS:
%
% net -- the network structure
%
%
% Copyright (c) Jonathan A Michaels 2018
% Stanford University
% jonathanamichaels AT gmail DOT com


for gen = 1:length(seeds)
    %% Grab the right parameters for this generation
    mutationPowerLocal = allMutationPower(gen);
    decay1Local = allDecay1(gen);
    decay2Local = allDecay2(gen);
    if gen == 1
        rng(seeds(gen)) % set the precious seed
        %% Generate initial network
        net = policyInitFunLocal(policyInitInputsLocal, policyInitInputsOptionalLocal);
    else
        if ~isnan(seeds(gen))
            rng(seeds(gen)) % set the precious seed
            
            %% Make some babies
            net.wIn = (net.wIn + (randn(size(net.wIn)) * mutationPowerLocal .* (net.wIn ~= 0))) .* (decay1Local * ones(size(net.wIn)));
            net.wFb = (net.wFb + (randn(size(net.wFb)) * mutationPowerLocal .* (net.wFb ~= 0))) .* (decay1Local * ones(size(net.wFb)));
            net.wOut = (net.wOut + (randn(size(net.wOut)) * mutationPowerLocal .* (net.wOut ~= 0))) .* (decay1Local * ones(size(net.wOut)));
            net.J = (net.J + (randn(size(net.J)) * mutationPowerLocal .* (net.J ~= 0))) .* (decay1Local * ones(size(net.J)));
            net.bJ = (net.bJ + (randn(size(net.bJ)) * mutationPowerLocal .* (net.bJ ~= 0))) .* (decay1Local * ones(size(net.bJ)));
            net.bOut = (net.bOut + (randn(size(net.bOut)) * mutationPowerLocal .* (net.bOut ~= 0))) .* (decay1Local * ones(size(net.bOut)));
            net.x0 = (net.x0 + (randn(size(net.x0)) * mutationPowerLocal .* (net.x0 ~= 0))) .* (decay1Local * ones(size(net.x0)));
            
            %% Decay to zero if desired
            if weightDecay
                net.wIn = net.wIn - decay2Local*(net.wIn-decay2Local > 0) + decay2Local*(net.wIn+decay2Local < 0);
                net.wFb = net.wFb - decay2Local*(net.wFb-decay2Local > 0) + decay2Local*(net.wFb+decay2Local < 0);
                net.wOut = net.wOut - decay2Local*(net.wOut-decay2Local > 0) + decay2Local*(net.wOut+decay2Local < 0);
                net.J = net.J - decay2Local*(net.J-decay2Local > 0) + decay2Local*(net.J+decay2Local < 0);
                net.bJ = net.bJ - decay2Local*(net.bJ-decay2Local > 0) + decay2Local*(net.bJ+decay2Local < 0);
                net.bOut = net.bOut - decay2Local*(net.bOut-decay2Local > 0) + decay2Local*(net.bOut+decay2Local < 0);
                net.x0 = net.x0 - decay2Local*(net.x0-decay2Local > 0) + decay2Local*(net.x0+decay2Local < 0);
            end
        end
    end
end
end