function [Z0, Z1, R, dR, X, varargout] = geneticRNN_run_model(net, varargin)

% net = hebbRNN_run_model(x0, net, F, varargin)
%
% This function runs the network structure (net) initialized by
% hebbRNN_create_model and trained by hebbRNN_learn_model with the desired
% input.
% NOTE: Networks must have been trained by hebbRNN_learn_model in order to be run
% by this function
%
%
% INPUTS:
%
% x0 -- the initial activation (t == 0) of all neurons
% Must be of size: net.N x 1
%
% net -- the network structure created by hebbRNN_create_model
%
% F -- the desired output
% Must be a cell of size: 1 x conditions
% Each cell must be of size: net.B x time points
%
%
% OPTIONAL INPUTS:
%
% input -- the input to the network
% Must be a cell of size: 1 x conditions
% Each cell must be of size: net.I x time points
% Default: []
%
%
% OUTPUTS:
%
% Z -- the output of the network
%
% R -- the firing rate of all neurons in the network
%
% X -- the activation of all neurons in the network
%
% errStats -- the structure containing error information from learning
% (optional)
%
% targetOut -- structure containing the output produced by targetFun
% (optional)
%
%
% Copyright (c) Jonathan A Michaels 2016
% German Primate Center
% jonathanamichaels AT gmail DOT com
%
% If used in published work please see repository README.md for citation
% and license information: https://github.com/JonathanAMichaels/hebbRNN


% Variable output considerations
nout = max(nargout,1)-1;

% Variable input considerations
optargin = size(varargin,2);

inp = [];
niters = [];
targetFun = @defaultTargetFunction;
targetFunPassthrough = [];
for iVar = 1:2:optargin
    switch varargin{iVar}
        case 'input'
            inp = varargin{iVar+1};
        case 'niters'
            niters = varargin{iVar+1};
            
        case 'targetFun'
            targetFun = varargin{iVar+1};
        case'targetFunPassthrough'
            targetFunPassthrough = varargin{iVar+1};
    end
end

N = net.N;
B = net.B;
I = net.I;

% The input can be either empty, or specified at each time point by the user.
hasInput = ~isempty(inp);
if (hasInput)
    assert(size(inp{1},1) == I, 'There must be an input entry for each input vector.');
    condList = 1:length(inp);
else
    assert(~isempty(niters), 'If no input is present the number of timepoints must be specified')
    condList = 1;
end

J = net.J;
wIn = net.wIn;
wFb = net.wFb;
wOut = net.wOut;
x0 = net.x0;
bJ = net.bJ;
bOut = net.bOut;
dt = net.dt;
tau = net.tau;
dt_div_tau = dt/tau;
netNoiseSigma = net.netNoiseSigma;
actFun = net.actFun;
actFunDeriv = net.actFunDeriv;

Z1 = cell(1,length(condList));
Z0 = cell(1,length(condList));
R = cell(1,length(condList));
dR = cell(1,length(condList));
X = cell(1,length(condList));
saveTarg = [];
for cond = 1:length(condList)
    thisCond = condList(cond);
    if hasInput
        thisInp = inp{thisCond};
        niters = size(thisInp,2);
    end
    targetFeedforward = [];
    
    allZ0 = zeros(niters,B);
    allZ1 = zeros(niters,B);
    allR = zeros(niters,N);
    alldR = zeros(niters,N);
    allX = zeros(niters,N);
    
    x = x0;
    
    %% Activation function
    r = actFun(x);
    dr = actFunDeriv(r);
    out = wOut*r + bOut;
    
    %% Calculate output using supplied function
    [z, targetFeedforward] = targetFun(0, out, targetFunPassthrough, targetFeedforward);

    for i = 1:niters
        if (hasInput)
            input = wIn*thisInp(:,i);
        else
            input = 0;
        end
        
        allZ0(i,:) = out;
        allZ1(i,:) = z;
        allR(i,:) = r;
        alldR(i,:) = dr;
        allX(i,:) = x;
        if i == niters
            saveTarg = targetFeedforward;
        end
        
        %% Calculate change in activation
        excitation = -x + J*r + input + wFb*z + bJ + netNoiseSigma*randn(N,1);
        %% Add all activation changes together
        x = x + dt_div_tau*excitation;
        
        %% Activation function
        r = actFun(x);
        dr = actFunDeriv(r);
        out = wOut*r + bOut;
        
        %% Calculate output using supplied function
        [z, targetFeedforward] = targetFun(i, out, targetFunPassthrough, targetFeedforward);
    end
    %% Save all states
    Z0{cond} = allZ0';
    Z1{cond} = allZ1';
    R{cond} = allR';
    dR{cond} = alldR';
    X{cond} = allX';
    if ~isempty(saveTarg)
        targetOut(cond) = saveTarg;
    end
end


%% Output error statistics if required
if (nout >= 5)
    if exist('targetOut', 'var')
        varargout{1} = targetOut;
    else
        varargout{1} = [];
    end
end

%% Default output function
    function [z, targetFeedforward] = defaultTargetFunction(~, r, ~, targetFeedforward)
        z = r; % Just passes firing rate
    end
end