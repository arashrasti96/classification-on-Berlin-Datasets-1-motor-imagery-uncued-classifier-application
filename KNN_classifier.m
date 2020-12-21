%                                 ###############
%                                 #   ########  #
%                                 #   #    ##   #
%                                 #   #   ##    #
%                                 #   # #       #
%                                 #   #  #      #
%                                 #   #   #     #
%                                 #  ###   ###  #
%                                 ###############
%
% Project Name: classification of right hand/left movement imagination using motor imagery EEG signals acquired from motor cortex area
% 
%
% Project Dataset:   Berlin BCI (http://www.bbci.de/competition/iv/desc_1.html)
%
% project Developer: Arash Rasti-meymandi
%
%% initialization
clc, clear all, close all;
subjects = ['a' 'b' 'c' 'd' 'e' 'f' 'g'];
sub_num=0;
k = 10; % K-fold validation
accuracy = zeros(length(subjects),k);
%% Loading Data

for sub=subjects
    sub_num=sub_num+1;
    sub_file=['BCICIV_calib_ds1' num2str(sub) '_100Hz.mat'];
    [path,filename,ext] = fileparts(which(mfilename));
    fullPath = [path,'\Datasets\',sub_file];
    load(fullPath);
    signal = 0.1*double(cnt);
    fs = nfo.fs;
    labels = mrk.y;
    pos = mrk.pos;
    %% Frequency Domain Filtering
    [b,a] = butter(3,[8 12]/(fs/2),'bandpass');
    signal = filter(b,a,signal);
    %% CAR  (Common Average Filtering)
    avg = mean(signal,2);
    for i_ch = 1:size(signal,2)
        signal(:,i_ch) = signal(:,i_ch) - avg;
    end
    %% Channel Selection (Not Always here)
    ch_ind = 26:32;
    ch_ind(4) = [];
    signal = signal(:,ch_ind);
    %% Epoching
    n1 = length(find(labels == -1));
    n2 = length(find(labels == 1));
    sig1 = cell(n1,1);
    sig2 = cell(n2,1);
    ind1 = 1;
    ind2 = 1;
    for i_trial = 1:length(pos)
        idx = pos(i_trial):pos(i_trial)+4*fs;
        if(labels(i_trial) == -1)
            sig1{ind1} = signal(idx,:);
            ind1 = ind1 + 1;
        else
            sig2{ind2} = signal(idx,:);
            ind2 = ind2 + 1;
        end
    end
    
    %% Feature Extraction
    feat1 = [];
    feat2 = [];
    for i_trial = 1:length(sig1)
        feat1 = [feat1; var(sig1{i_trial})];
    end
    for i_trial = 1:length(sig2)
        feat2 = [feat2; var(sig2{i_trial})];
    end
    feat = [feat1;feat2];
    labels = [ones(size(feat1,1),1);2*ones(size(feat2,1),1)];
    
    %% Validation
    % The first 70% trials of each class -> training data
    % The rest of trials -> test data
    indices = crossvalind('Kfold',labels,k);
    
    
    for i_fold = 1:k
        test = indices==i_fold;
        train = ~test;
        featureTrain = feat(train,:);
        featureTest = feat(test,:);
        
        %% Classification
        
        Model = fitcknn(featureTrain,labels(train),'NumNeighbors',5);
        class = predict(Model, featureTest);
        accuracy(sub_num,i_fold) = 100*length(find(class == labels(test)))/length(labels(test));
        %disp(['fold ',num2str(i_fold),', accuracy = ',num2str(accuracy(i_fold))])
    end
    disp(['Subject_1' sub '_accuracy= ' ,num2str(mean(accuracy(sub_num,:))),' ± ',num2str(std(accuracy(sub_num,:)))])
end

