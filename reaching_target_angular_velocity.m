function [psycho,choices]=reaching_target_angular_velocity(input,these_stim,this_dir,analysis_methods)
    close all;
    % stim_type = unique(these_stim);
    % stims_for_thresh = unique(these_stim);
    % stims_for_thresh(stim_type==0) = [];
    n_trial_per_animal=size(input,1);
    stim_direction=these_stim./abs(these_stim);% Retrieve the stimulus direction irrespective to coherence level
    save_fig=analysis_methods.save_fig;
    saveas_svg_file=analysis_methods.saveas_svg_file;
    saveas_jpg_file=analysis_methods.saveas_jpg_file;
    analyse_angular_velocity=analysis_methods.analyse_angular_velocity;
    apply_subsampling=analysis_methods.apply_subsampling;
    num_iteration=99;
    use_best_threshold=analysis_methods.use_best_threshold;
    stim_onset_frame=analysis_methods.stim_onset_frame;
    stim_offset_frame=analysis_methods.stim_offset_frame;
    bin_length=analysis_methods.bin_length;
    [gc,~] = groupcounts(these_stim');%[~,~,ix] = unique(these_stim);C = accumarray(ix,1);  do the same trick
    min_trial_per_condition=min(gc);

    %% Computing threshold and AUC for each frame
    % myCols = parula(size(input,2));
    % AUCs = nan(1,size(input,2));
    % thresh = nan(1,size(input,2));
    % inputStim = input(:,stim_onset_frame:stim_offset_frame);
    %inputStim = input;
%     if use_best_threshold && apply_subsampling
%         Thresholds=nan(1,num_iteration);
%         for this_iteration=1:num_iteration
%             balanced_trials=nan(numel(stims_for_thresh),min_trial_per_condition);
%             for istim=1:numel(stims_for_thresh)
%                 cohIdx =these_stim==stims_for_thresh(istim);
%                 tmp=find(cohIdx);
%                 sub_sample=datasample(tmp,min_trial_per_condition,'Replace',false);
%                 balanced_trials(istim,:)=sub_sample;
%             end
%         balanced_index=reshape(balanced_trials,1,[]);
%         balanced_stim_direction=stim_direction(balanced_index);
%         input_for_classification=inputStim(balanced_index,:);
%         %balanced_stim_direction(balanced_stim_direction==0) = sign(randn(sum(balanced_stim_direction==0),1));
% %         stim_directin_for_classification=balanced_stim_direction(balanced_stim_direction~=0);
% %         input_for_classification=inputStim(balanced_stim_direction~=0,:);
%         [FPR,TPR,T,~,opt]=perfcurve(repmat(balanced_stim_direction,1,size(inputStim,2)),reshape(double(input_for_classification),1,[]),1);
%         balancedThresh = T(FPR==opt(1) & TPR==opt(2));% Get the best threshold from the ROC curve
%         Thresholds(this_iteration)=balancedThresh;
%         end
%     theThresh=mean(Thresholds);
%     elseif use_best_threshold
%         [FPR,TPR,T,~,opt]=perfcurve(repmat(stim_direction,1,size(inputStim,2)),reshape(double(inputStim),1,[]),1);
%         theThresh = T(FPR==opt(1) & TPR==opt(2));% Get the best threshold from the ROC curve
%     else
%         roc_ori_fig=figure;
%         for iframe = 1: size(input,2)
%             % Compute the ROC curve, AUC, and optimal classification threshold for each frame
%             [FPR,TPR,T,AUC,opt]=perfcurve(stim_direction,double(input(:,iframe)),1);
%             plot(FPR,TPR,'color',myCols(iframe,:));% Plot the ROC curve for each frame, to observe how classification performance evolves over time.
%             AUCs(iframe) = AUC; % Store AUC value for each frame
%             thresh(iframe) = T(FPR==opt(1) & TPR==opt(2));% Store the optimal threshold
%             hold on
%         end
%         hold off
%         xlabel('False Positive Rate')
%         ylabel('True Positive Rate')
%         title('ROC curve over time')
%         theThresh=thresh(iframe);
% 
%         % Scatter plot of AUC values across frames
%         auc_ori_fig=figure;
%         scatter(1:numel(AUCs),AUCs,40,myCols)
%         title('AUC values over time')
%         xlabel('Time (sec)')
%         set(gca,'xtick',[stim_onset_frame stim_offset_frame])
%         set(gca,'xticklabel',{'0','25'})
% 
%         % Plot the threshold values across frames
%         optimal_thres_ori_fig=figure;
%         plot(thresh)
%         title('optimal threshold values over time')
%         xlabel('Time (sec)')
%         set(gca,'xtick',[stim_onset_frame stim_offset_frame])
%         set(gca,'xticklabel',{'0','25'})
%     end
    %% Computing accuracy for each frame and each coherence using frame-based threshold
    theThresh=analysis_methods.target_angular_velocity;
    pCorr = nan(1,size(input,2));
    % myStims = unique(these_stim); % coherence and direction
    myStims = unique(abs(these_stim)); % absolute coherence
    % Define the Correct timulus direction resonse which is used  to handle any
    % 0 coherence stimulus
    corrected_stim_dir = stim_direction;
    corrected_stim_dir(stim_direction==0) = sign(randn(sum(stim_direction==0),1)); % generating uniform random values of 1 or -1

    %corrected_stim_dir(stim_direction==0) =randi([0, 1], 1, sum(stim_direction==0)) * 2 - 1;% generating random values of 1 or -1
    %%sequence = [ones(1, sum(stim_direction==0)/2), -ones(1, sum(stim_direction==0)/2)];% generating a sequence of even number of 1 or -1
    % Randomly shuffle the sequence
    %shuffled_sequence = sequence(randperm(sum(stim_direction==0)));
    psycho = nan(numel(myStims),size(input,2)); % Pre-allocate psychometric data
    correctClassifications_across_stim=nan(size(input));
    choices=nan(size(input));
    for iframe = 1:size(input,2)
         % Classify the stimuli based on the threshold
        choices(:,iframe) = input(:,iframe)>=theThresh;
        correctClassifications = (input(:,iframe)>=theThresh) == (corrected_stim_dir==1)'; % including unfair stimuli!
        pCorr(iframe) = mean(correctClassifications);% Store accuracy
        correctClassifications_across_stim(:,iframe)=correctClassifications;
%         histo_fig=figure;
%         histogram(correctClassifications)
    % Loop through each stimulus coherence level
        for istim = 1:numel(myStims)
            if apply_subsampling
                sub_sample_psycho=nan(1,num_iteration);
                for this_iteration=1:num_iteration
                    if myStims(istim)==0
                        cohIdx = abs(these_stim)==myStims(istim);
                        tmp=find(cohIdx);
                        sub_sample=datasample(tmp,min_trial_per_condition,'Replace',false);
                    else
                        cohIdx = these_stim==myStims(istim);
                        tmp=find(cohIdx);
                        sub_sample1=datasample(tmp,min_trial_per_condition,'Replace',false);
                        cohIdx = these_stim==myStims(istim)*-1;
                        tmp=find(cohIdx);
                        sub_sample2=datasample(tmp,min_trial_per_condition,'Replace',false);
                        sub_sample=cat(2,sub_sample2,sub_sample1);
                    end
                    theseDirs=corrected_stim_dir(sub_sample);
                    theseAngles = input(sub_sample,iframe);
                    sub_sample_psycho(this_iteration) = mean((theseAngles>=theThresh) == (theseDirs==1)');% Compute classification accuracy
                end
                psycho(istim,iframe) = mean(sub_sample_psycho);
            else
                cohIdx = abs(these_stim)==myStims(istim);%%fix here to take coherence values irespective to direction
                theseDirs = corrected_stim_dir(cohIdx);
                % theseCoh = these_stim(cohIdx);
                theseAngles = input(cohIdx,iframe);
                stimCorrectClassifications = (theseAngles>=theThresh) == (theseDirs==1)';% Compute classification accuracy
                psycho(istim,iframe) = mean(stimCorrectClassifications);
            end
        end
    
    end
    % Visualize the psychometric data to show the classification accuracy as a function of the stimulus coherence level across time.

    pscho_over_time_bin_ori_fig=figure;
    imagesc(psycho)
    title('Correct responses over time')
    xlabel('Time (sec)')
    set(gca,'xtick',[stim_onset_frame stim_offset_frame])
    set(gca,'xticklabel',{'0','25'})
    ylabel('Coherence level')
    set(gca,'ytick',[1 2 3 4])
    set(gca,'yticklabel',{'0%','33%','66%','100%'})
    
    
    %% Plotting accuracy with frame-based threshold
    %% important figure
    frameBins = stim_onset_frame:bin_length:stim_offset_frame;% Define frame bins for analysis
    pscho_over_time_scatter_ori_fig=figure;
    title('Correct responses over time')
    xlabel('Coherence level (%)')
    binCols = parula(numel(frameBins)-1);% Colormap for frame bins
    for ibin = 1:numel(frameBins)-1
    myFrames = frameBins(ibin):frameBins(ibin+1);% Define the frame range for each bin
    meanPsycho = mean(psycho(:,myFrames),2);% Compute the mean psychometric data for the bin
    plot(meanPsycho,'-o','Color',binCols(ibin,:)),hold on% Plot the mean psychometric data
    end
    ylabel('Probability of Correct')
    set(gca,'xtick',[1 2 3 4])
    set(gca,'xticklabel',{'0%','33%','66%','100%'})
    
    %% Plotting autocorrelation of the stimulus direction over time to understand any temporal dependencies in the stimulus sequence.
    % stim_direction=int8(stim_direction);
    % auto_stim_direction_fig=figure;
    % ax1 = subplot(2,1,1);
    % plot(xcorr(stim_direction(1:n_trial_per_animal),'coeff'))
    % ax2 = subplot(2,1,2);
    % plot(xcorr(stim_direction,'coeff'))
    % title('Autocorrelation of the stimulus direction')
    % xlabel('Lag')

    %%%

    if save_fig
        fig_ext='.fig';
        if analyse_angular_velocity
            name_suffix='_ori';
        else
            name_suffix='_diff_ori';
        end
        if saveas_svg_file
            file_ext='.svg';
        elseif saveas_jpg_file
            file_ext='.jpg';    
        end
        if use_best_threshold
            name_threshold='_bestThresh';
        else
            name_threshold='_fbfThresh';
            saveas(roc_ori_fig,fullfile(this_dir,append('roc',name_suffix,file_ext)));
            saveas(auc_ori_fig,fullfile(this_dir,append('auc',name_suffix,file_ext)));
            savefig(roc_ori_fig,fullfile(this_dir,append('roc',name_suffix,fig_ext)));
            savefig(auc_ori_fig,fullfile(this_dir,append('auc',name_suffix,fig_ext)));
            saveas(optimal_thres_ori_fig,fullfile(this_dir,append('optimal_thres_ori',name_suffix,file_ext)));
            savefig(optimal_thres_ori_fig,fullfile(this_dir,append('optimal_thres_ori',name_suffix,fig_ext)));
        end
        saveas(pscho_over_time_bin_ori_fig,fullfile(this_dir,append('pscho_over_time_bin',name_threshold,name_suffix,file_ext)));
        saveas(pscho_over_time_scatter_ori_fig,fullfile(this_dir,append('pscho_over_time_scatter',name_threshold,name_suffix,file_ext)));
        savefig(pscho_over_time_bin_ori_fig,fullfile(this_dir,append('pscho_over_time_bin',name_threshold,name_suffix,fig_ext)));
        savefig(pscho_over_time_scatter_ori_fig,fullfile(this_dir,append('pscho_over_time_scatter',name_threshold,name_suffix,fig_ext)));
        savefig(auto_stim_direction_fig,fullfile(this_dir,append('auto_stim_direction',fig_ext)));
    else
        disp('do not save any figures')
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end