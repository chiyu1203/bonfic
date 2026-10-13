function [psycho,choices]=reaching_target_angular_velocity(input,these_stim,this_dir,analysis_methods)
    close all;
    stim_type = unique(these_stim);
    % stims_for_thresh = unique(these_stim);
    % stims_for_thresh(stim_type==0) = [];
    num_iteration=99;
    stim_direction=these_stim./abs(these_stim);% Retrieve the stimulus direction irrespective to coherence level
    save_output=analysis_methods.save_output;
    saveas_svg_file=analysis_methods.saveas_svg_file;
    saveas_jpg_file=analysis_methods.saveas_jpg_file;
    target_consecutive_frame=analysis_methods.target_consecutive_frame;
    analyse_angular_velocity=analysis_methods.analyse_angular_velocity;
    apply_subsampling=analysis_methods.apply_subsampling;
    use_best_threshold=analysis_methods.use_best_threshold;
    stim_onset_frame=analysis_methods.stim_onset_frame;
    stim_offset_frame=analysis_methods.stim_offset_frame;
    camera_fps=analysis_methods.camera_fps;
    analysis_window=analysis_methods.analysis_window*camera_fps;
    target_consecutive_frame=analysis_methods.target_consecutive_frame;
    bin_length=analysis_methods.bin_length;
    [gc,~] = groupcounts(these_stim');%[~,~,ix] = unique(these_stim);C = accumarray(ix,1);  do the same trick
    min_trial_per_condition=min(gc);

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
    %psycho = nan(numel(myStims),analysis_window(2)-analysis_window(1));
    %cumulative_follow_across_coherence= nan(numel(stim_type),analysis_window(2)-analysis_window(1));
    cumulative_follow_across_coherence= nan(numel(stim_type),size(input,2));
    choices=nan(size(input,1),size(input,2));
    choices_follow=nan(size(input,1),size(input,2));
    correctClassifications_across_stim=nan(size(input));
    %choices=nan(size(input,1),analysis_window(2)-analysis_window(1));
    %choices_follow=nan(size(input,1),analysis_window(2)-analysis_window(1));
    if target_consecutive_frame>1
        for i = 1:size(stim_type,2)
            this_stim_type=stim_type(i);
            if this_stim_type<0
                these_angular_velocity_flipped_direction=input(these_stim==this_stim_type,:)*-1;
            else
                these_angular_velocity_flipped_direction=input(these_stim==this_stim_type,:);
            end
            these_angular_velocity=input(these_stim==this_stim_type,:);
            [follow_masks,~]=classify_angular_velocity(these_angular_velocity_flipped_direction(:,:),theThresh,target_consecutive_frame);
            [cw_masks,~]=classify_angular_velocity(these_angular_velocity(:,:),theThresh,target_consecutive_frame);
            follow_across_trial=sum(follow_masks,1);
            cumulative_follow_across_trial=cumsum(follow_across_trial);
            cumulative_follow_across_coherence(i,:)=cumulative_follow_across_trial;
            choices(these_stim==this_stim_type,:)=cw_masks;
            choices_follow(these_stim==this_stim_type,:)=follow_masks;
        end
        for i = 1:numel(myStims)
            these_data=choices_follow(abs(these_stim)==myStims(i),:);
            psycho(i,:)=sum(these_data,1)/(size(these_data,1)*size(these_data,2));
        end
    else
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

    if save_output
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