function net = geneticRNN_create_model(N, B, I, p, g, dt, tau, varargin)

% net = geneticRNN_create_model(N, B, I, p, g, dt, tau, varargin)
%
% This function initializes a recurrent neural network for later training
% and execution
%
% INPUTS:
%
% N -- the number of recurrent neurons in network
%
% B -- the number of outputs
%
% I -- the number of inputs
%
% p -- the sparseness of the J (connectivity) matrix, (range: 0-1)
%
% g -- the spectral scaling of J
%
% dt - the integration time constant
%
% tau - the time constant of each neuron
%
%
% OPTIONAL INPUTS:
%
% actFun -- the activation function used to tranform activations into
% firing rates
% Default: 'tanh'
%
% netNoiseSigma - the variance of random gaussian noise added at each time
% point
% Default: 0
%
% feedback -- whether or not to feed the output of the network back
% Default: false
%
% energyCost -- how much to weight the firing rate of the output units in
% the calculation of total error. This parameter encourages the network to
% find solutions with as low firing rate as possible.
% Default: 0
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


actFunType = 'tanh'; % Default activation function
netNoiseSigma = 0.0; % Default noise-level
feedback = false; % Default use of output feedback
energyCost = 0; % Default weight of cost function
optargin = size(varargin,2);

for i = 1:2:optargin
    switch varargin{i}
        case 'actFun'
            actFunType = varargin{i+1};
        case 'netNoiseSigma'
            netNoiseSigma = varargin{i+1};
        case 'feedback'
            feedback = varargin{i+1};
        case 'energyCost'
            energyCost = varargin{i+1};         
    end
end

%% Assertions
assert(islogical(feedback), 'Must be logical.')
assert(p >= 0 && p <= 1, 'Sparsity must be between 0 and 1.')

%% Initialize internal connectivity
% Connectivity is normally distributed, scaled by the size of the network,
% the sparity, and spectral scaling factor, g.
J = zeros(N,N);
for i = 1:N
    for j = 1:N
        if rand <= p
            J(i,j) = g * randn / sqrt(p*N);
        end
    end
end

net.I = I;
net.B = B;
net.N = N;
net.p = p;
net.g = g;
net.J = J;
net.netNoiseSigma = netNoiseSigma;
net.dt = dt;
net.tau = tau;

%% Initialize input weights
net.wIn = randn(N,I) / sqrt(I);

%% Initialize feedback weights
net.wFb = zeros(N,B);
if feedback
    net.wFb = randn(N,B) / sqrt(B);
end

%% Initialize output weights
net.wOut = randn(B,N) / sqrt(N);

%% Initialize J biases
net.bJ = randn(N,1) / sqrt(N);

%% Initialize output biases
net.bOut = randn(B,1) / sqrt(B);

%% Initialize starting activation
net.x0 = randn(N,1) / sqrt(N);

%% Activation function
switch actFunType
    case 'tanh'
        net.actFun = @tanh;
    case 'recttanh'
        net.actFun = @(x) (x > 0) .* tanh(x);
    case 'baselinetanh' % Similar to Rajan et al. (2010)
        net.actFun = @(x) (x > 0) .* (1 - 0.1) .* tanh(x / (1 - 0.1)) ...
            + (x <= 0) .* 0.1 .* tanh(x / 0.1);
    case 'linear'
        net.actFun = @(x) x;
    otherwise
        assert(false, 'Nope!');
end

%% Cost function
net.energyCost = energyCost;
end