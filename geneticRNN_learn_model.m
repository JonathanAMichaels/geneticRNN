function [winner, varargout] = geneticRNN_learn_model(inp, mutationPower, populationSize, truncationSize, fitnessFunInputs, policyInitInputs, varargin)

% net = geneticRNN_learn_model(inp, mutationPower, populationSize, truncationSize, fitnessFunInputs, policyInitInputs, varargin)
%
% This function trains a recurrent neural network using a simple genetic algorithm
% to complete the desired goal.
%
% INPUTS:
%
% inp -- Inputs to the network. Must be present, but can be empty.
%
% mutationPower -- Standard deviation of normally distributed noise to add in each generation
%
% populationSize -- Number of individuals in each generation
%
% truncationSize -- Number of individuals to save for next generation
%
% fitnessFunInputs -- Target information for calculating the fitness
%
% policyInitInputs -- Inputs for the policy initialization function
%
%
% OPTIONAL INPUTS:
% 
% mutationPowerDecay -- Natural decay rate of mutation power
%
% mutationPowerDrop -- Decay rate of mutation power when we don't learn anything on a given generation
%
% weightCompression -- Whether or not to compress policy (logical)
%
% weightDecay -- Whether or not to decay policy (logical)
%
% fitnessFun -- function handle for assessing fitness
% Default: @defaultFitnessFunction
%
% policyInitFun -- function handle for initializing the policy
% Default: @geneticRNN_create_model
%
% policyInitInputsOptional -- Optional inputs for the policy initialization function
%
% targetFun -- The handle of a function that uses the firing rates of the
% output units to produce some desired output. Function must follow
% conventions of supplied default function.
% Default: @defaultTargetFunction
%
% targetFunPassthrough -- A user-defined structure that is automatically
% passed through to the targetFun, permitting custom variables to be passed
% Default: []
%
% plotFun -- The handle of a function that plots information about the
% network during the learning process. Function must follow conventions
% of supplied default function.
% Default: @defaultPlottingFunction
%
% evalOpts -- A vector of size 2, specifying how much information should be
% displayed during training (0 - nothing, 1 - text only, 2 - text +
% figures), and how often the network should be evaluated. This vector is
% passed to the plotting function.
% Default: [1 1]
%
%
% OUTPUTS:
%
% winner -- the network structure
%
% errStats -- the structure containing error information from learning
% (optional)
%
%
% Copyright (c) Jonathan A Michaels 2018
% Stanford University
% jonathanamichaels AT gmail DOT com


% Start counting
tic

% Variable output considerations
nout = max(nargout,1)-1;

% Variable input considerations
optargin = size(varargin,2);

mutationPowerDecay = 0.99;
mutationPowerDrop = 0.7;
weightCompression = true; % By default we will compress
weightDecay = false; % By default we won't use decay
targetFun = @defaultTargetFunction; % Default output function (native)
plotFun = @defaultPlottingFunction; % Default plotting function (native)
fitnessFun = @defaultFitnessFunction; % Default fitness function (native)
policyInitFun = @geneticRNN_create_model;
policyInitInputsOptional = [];
targetFunPassthrough = []; % Default passthrough to output function
evalOpts = [1 1]; % Default evaluation values [plottingOptions evaluateEveryXIterations]

for iVar = 1:2:optargin
    switch varargin{iVar}            
        case 'mutationPowerDecay'
            mutationPowerDecay = varargin{iVar+1};
        case 'mutationPowerDrop'
            mutationPowerDrop = varargin{iVar+1};
            
        case 'weightCompression'
            weightCompression = varargin{iVar+1};
        case 'weightDecay'
            weightDecay = varargin{iVar+1};
            
        case 'fitnessFun'
            fitnessFun = varargin{iVar+1};
        case 'policyInitFun'
            policyInitFun = varargin{iVar+1};
        case 'policyInitInputsOptional'
            policyInitInputsOptional = varargin{iVar+1};
            
        case 'targetFun'
            targetFun = varargin{iVar+1};
        case 'targetFunPassthrough'
            targetFunPassthrough = varargin{iVar+1};
            
            
        case 'plotFun'
            plotFun = varargin{iVar+1};
        case 'evalOpts'
            evalOpts = varargin{iVar+1};
    end
end

errStats.fitness = []; errStats.generation = []; % Initialize error statistics
g = 1; % Initialize generation
allDecay1 = []; allDecay2 = []; allMutationPower = []; % Initialize decay history

%% Main Program %%
% Runs until tolerated error is met or stop button is pressed
figure(97)
set(gcf, 'Position', [0 50 100 50], 'MenuBar', 'none', 'ToolBar', 'none', 'Name', 'Stop', 'NumberTitle', 'off')
UIButton = uicontrol('Style', 'togglebutton', 'String', 'STOP', 'Position', [0 0 100 50], 'FontSize', 25);
while UIButton.Value == 0
    %% Initialize parameters
    if weightCompression
        decay1 = 1 - mutationPower;
    else
        decay1 = 1;
    end
    decay2 = mutationPower * 1e-1;
    allDecay1 = cat(2, allDecay1, decay1);
    allDecay2 = cat(2, allDecay2, decay2);
    allMutationPower = cat(2, allMutationPower, mutationPower);
    fitness = zeros(length(inp),populationSize);
    
    %% Generate random seeds
    theseSeeds = randsample(1e8, populationSize);
    if g > 1
        previousSeeds = masterSeeds(randsample(size(masterSeeds,1), populationSize, true), :);
        previousSeeds(1,:) = masterSeeds(1,:); % Save the elite!
        theseSeeds(1) = nan; % Save the elite!
        sendSeeds = [previousSeeds, theseSeeds];
    else
        sendSeeds = theseSeeds;
    end
    
    %% Heavy lifting
    parfor i = 1:populationSize
        % Hack the random number generator
        stream = RandStream('mrg32k3a');
        RandStream.setGlobalStream(stream);
        stream.Substream = i;
        % Rollout the model based on the random seeds
        net = geneticRNN_rollout_model(policyInitFun, policyInitInputs, policyInitInputsOptional, allMutationPower, allDecay1, allDecay2, weightDecay, sendSeeds(i,:));
        % Run model
        [~, Z1, ~, ~] = geneticRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
        % Assess fitness
        fitness(:,i) = fitnessFun(Z1, fitnessFunInputs);
    end
    
    %% Sort and save best policies
    [~, sortInd] = sort(mean(fitness,1), 'descend');
    fitness = fitness(:,sortInd(1:truncationSize));
    masterSeeds = sendSeeds(sortInd(1:truncationSize),:);
    
    %% Recalculate best network for plotting or output
    % Hack the random number generator
    stream = RandStream('mrg32k3a');
    RandStream.setGlobalStream(stream);
    stream.Substream = sortInd(1);
    % Rollout the model based on the random seeds
    net = geneticRNN_rollout_model(policyInitFun, policyInitInputs, policyInitInputsOptional, allMutationPower, allDecay1, allDecay2, weightDecay, masterSeeds(1,:));
    % Run model
    [~, Z1, R, ~] = geneticRNN_run_model(net, inp, 'targetFun', targetFun, 'targetFunPassthrough', targetFunPassthrough);
    
    %% Save stats
    errStats.fitness(:,end+1) = fitness(:,1);
    errStats.generation(end+1) = g;
    
    %% Populate statistics for plotting function
    plotStats.fitness = fitness;
    plotStats.mutationPower = mutationPower;
    plotStats.generation = g;
    plotStats.bigZ1 = Z1;
    plotStats.bigR = R;
    plotStats.targ = fitnessFunInputs;
    
    %% Run supplied plotting function
    if mod(g,evalOpts(2)) == 0
        plotFun(plotStats, errStats, evalOpts)
    end
    
    %% Decay mutation power
    if sortInd(1) == 1
        mutationPower = mutationPower * mutationPowerDrop; % Big drop if we didn't learn anything
    else
        mutationPower = mutationPower * mutationPowerDecay; % Small drop if we learned something
    end
    
    g = g + 1;
end

%% Output error statistics if required
if ( nout >= 1 )
    varargout{1} = errStats;
end

%% Save hard-earned elite network
winner = net;

disp('Training time required:')
toc

    %% Default plotting function
    function defaultPlottingFunction(plotStats, errStats, evalOptions)
        if evalOptions(1) >= 0
            disp(['Generation: ' num2str(plotStats.generation) '  Fitness: ' num2str(mean(plotStats.fitness(:,1))) '  Mutation Power: ' num2str(plotStats.mutationPower)])
        end
        if evalOptions(1) >= 1
            figure(98)
            set(gcf, 'Name', 'Error', 'NumberTitle', 'off')
            c = lines(size(plotStats.fitness,1));
            for type = 1:size(plotStats.fitness,1)
                h1(type) = plot(plotStats.generation, plotStats.fitness(type,1), '.', 'MarkerSize', 20, 'Color', c(type,:));
                hold on
            end
            plot(plotStats.generation, mean(plotStats.fitness(:,1),1), '.', 'MarkerSize', 40, 'Color', [0 0 0]);
            set(gca, 'XLim', [1 plotStats.generation+0.1])
            xlabel('Generation')
            ylabel('Fitness')
        end
        if evalOptions(1) >= 2
            figure(99)
            set(gcf, 'Name', 'Output and Neural Activity', 'NumberTitle', 'off')
            clf
            subplot(4,1,1)
            hold on
            c = lines(length(plotStats.bigZ1));
            for condCount = 1:length(plotStats.bigZ1)
                h2(condCount,:) = plot(plotStats.bigZ1{condCount}', 'Color', c(condCount,:));
                h3(condCount,:) = plot(plotStats.targ{condCount}', '.', 'MarkerSize', 8, 'Color', c(condCount,:));
            end
            legend([h2(1,1) h3(1,1)], 'Network Output', 'Target Output', 'Location', 'SouthWest')
            xlabel('Time Steps')
            ylabel('Output')
            set(gca, 'XLim', [1 size(plotStats.bigZ1{1},2)])
            for n = 1:3
                subplot(4,1,n+1)
                hold on
                for condCount = 1:length(plotStats.bigR)
                    plot(plotStats.bigR{condCount}(n,:)', 'Color', c(condCount,:))
                end
                xlabel('Time Steps')
                ylabel(['Firing Rate (Neuron ' num2str(n) ')'])
                set(gca, 'XLim', [1 size(plotStats.bigR{1},2)])
            end
        end
        drawnow
    end

    %% Default fitness function
    function fitness = defaultFitnessFunction(Z1, targ)
        fitness = zeros(1,length(Z1));
        for cond = 1:length(Z1)
            ind = ~isnan(targ{cond});
            useZ1 = Z1{cond}(ind);
            useF = targ{cond}(ind);
            
            err(1) = sum(abs(useZ1(:)-useF(:)));       
            fitness(cond) = -sum(err);
        end
    end

    %% Default output function
    function [z, targetFeedforward] = defaultTargetFunction(~, r, ~, targetFeedforward)
        z = r; % Just passes firing rate
    end
end