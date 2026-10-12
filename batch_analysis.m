clear;
close all;

%to do list: using Time-binned analysis to plot individual's AUC (or probability of
%correct) and overlay that with the mean response of AUC (pool across animals)
%to do list2: classify animals based on walking distance during isi or
%pre-stim interval and compare that classification with walking distance
%during the entire experiment or trial by trial
%to do list3: plot AUC values over time with different lengh of ISI

analysis_methods.example_data=0;
analysis_methods.target_angular_velocity=64*pi/180;
analysis_methods.loop_analysis_animal_by_animal=1;
analysis_methods.save_fig=0;
analysis_methods.saveas_svg_file=0;
analysis_methods.saveas_jpg_file=0;
analysis_methods.analyse_angular_velocity=1;
analysis_methods.apply_subsampling=1;
analysis_methods.use_best_threshold=1;
analysis_methods.stim_onset_frame=1000;
analysis_methods.stim_offset_frame=3500;
analysis_methods.bin_length=500;

analyse_angular_velocity=analysis_methods.analyse_angular_velocity;
use_best_threshold=analysis_methods.use_best_threshold;
stim_onset_frame=analysis_methods.stim_onset_frame;
stim_offset_frame=analysis_methods.stim_offset_frame;
bin_length=analysis_methods.bin_length;

active_animals_only=0;
frame_rate=100;
num_iteration=99;
if analysis_methods.example_data
    % Define folder and file paths for example data
    this_folder='Z:\DATA\experiment_trackball_Optomotor\MatrexVR\GN24064\240803\coherence\session1';
    matfile= 'behavioural_summary.mat';
    this_file=fullfile(this_folder,matfile);
    behavioural_summary=load(this_file);
    % filename ='cum_jaw_rotation.npy';
    filename ='angular_velocity_tbt.npy';
    this_file=fullfile(this_folder,filename);
else
    % Define folder and file paths for population data
    this_folder='Z:\DATA\experiment_trackball_Optomotor\MatrexVR';
    matfile='matrex_vr_coherence_database.mat';
    this_file=fullfile(this_folder,matfile);
    behavioural_summary=load(this_file);
    %filename ='rotation_summary.npy';
    filename ='angular_velocity_summary.npy';
    this_file=fullfile(this_folder,filename);
end
% Example implementation of memory mapping an NPY file using readNPYheader
[arrayShape, dataType, fortranOrder, littleEndian, totalHeaderLength, npyVersion] = readNPYheader(this_file);
raw_fig=figure;
if fortranOrder
    f = memmapfile(this_file, 'Format', {dataType, arrayShape, 'd'}, 'Offset', totalHeaderLength);
    image(f.Data.d)
    
else
    % Note! In this case, the dimensions of the array will be transposed,
    % e.g. an AxBxCxD array becomes DxCxBxA. 
    f = memmapfile(this_file, 'Format', {dataType, arrayShape(end:-1:1), 'd'}, 'Offset', totalHeaderLength);
    tmp = f.Data.d;
    input = permute(tmp, length(arrayShape):-1:1); % note here you have to reverse the dimensions. 
    %image(img./255)
    imagesc(input)
end
if analysis_methods.example_data
    title('Raw turning responses')
    ylabel('Trials (#)')
    xlabel('Time (sec)')
    set(gca,'xtick',[stim_onset_frame stim_offset_frame])
    set(gca,'xticklabel',{'0','25'})
    saveas(raw_fig,fullfile(this_folder,'turn.jpg'));
else
    close all;
end

%coherence_threshold=50;%filter data based on coherence threshold
%coherence_of_interest= abs(behavioural_summary.stim_type)>coherence_threshold;
%input=input(coherence_of_interest,:);
%stim_of_interest=behavioural_summary.stim_type(coherence_of_interest);
%n_trial_per_animal=arrayShape(1)/(max(behavioural_summary.ID)+1);
num_animals=max(behavioural_summary.ID)+1;
IDs=behavioural_summary.ID;
animal_IDs=unique(behavioural_summary.ID);
num_type_stim=size(unique(abs(behavioural_summary.stim_type)),2);
these_coherence=unique(abs(behavioural_summary.stim_type));
histo_fig=figure;
histogram(behavioural_summary.stim_type)

if analyse_angular_velocity
    %delta_orientation
    input_variable=input;
    %input_variable=diff(input,1,2);in 2024, the original turning variables
    %was saved as cumulative orientation
else
    %input_variable=input;in 2024, the original turning variables
    %was saved as cumulative orientation
    input_variable=cumsum(input,2);
end

if analysis_methods.loop_analysis_animal_by_animal
    AUCs_across_animals=nan(num_animals,size(input_variable,2));
    psycho_across_animals=nan(num_animals,num_type_stim,size(input_variable,2));
    total_travel_distance_across_animals=nan(num_animals,1);
    boolean_across_animals=nan(num_animals,size(input_variable,1)/num_animals,size(input_variable,2));
    threshold_across_animals=nan(num_animals,num_iteration);
    for thisAnimal=1:num_animals
%         if ismember([4 8 16 17 20 36 39 66],thisAnimal)
%             continue
%         end
        animal_of_interest=behavioural_summary.ID==thisAnimal-1;
        these_stim= behavioural_summary.stim_type(animal_of_interest);%there was an adjustment in the +/- direction in the stimulus type from 2024 to 2026
        this_input=input_variable(animal_of_interest,:);
        Pathes=behavioural_summary.Path(animal_of_interest);
        total_travel_distance_across_animals(thisAnimal)=sum(behavioural_summary.travel_distance(animal_of_interest));%there was an adjustment in the unit of travel distance from 2024 to 2026
        [psycho,choices_across_stim]=reaching_target_angular_velocity(this_input,these_stim,char(Pathes(1)),analysis_methods);
        %[AUCs,psycho,choices_across_stim,thresholds]=roc_analysis(this_input,these_stim,char(Pathes(1)),analysis_methods);
        if use_best_threshold
            disp('no AUCs across time when using best threshold in the experiment');
        else
            AUCs_across_animals(thisAnimal,:)=AUCs;
        end
        psycho_across_animals(thisAnimal,:,:)=psycho;
        boolean_across_animals(thisAnimal,:,:)=choices_across_stim;
        %threshold_across_animals(thisAnimal,:)=thresholds;
    end
    close all
%     auc_across_animals=figure;
%     myPlot = scatter(1:size(AUCs_across_animals,2),AUCs_across_animals',2,[0 0 0.5],...
%         'filled','o','MarkerFaceAlpha',0.1);
    if use_best_threshold
        disp('no AUCs across time when using best threshold in the experiment'); 
    else
        auc_across_animals=figure; hold on
        for ilocust = 1:size(AUCs_across_animals,1)
            myPlot = plot(1:size(AUCs_across_animals,2),AUCs_across_animals(ilocust,:),'LineWidth',1);
            myPlot.Color(4) = 0.1;
        end
        myPlot = plot(1:size(AUCs_across_animals,2),mean(AUCs_across_animals,1),'LineWidth',2,'Color','k');
        if analyse_angular_velocity
            title('AUCs (delta orientation)');
        else
            title('AUCs (orientation)');
        end
        xlabel('Time (sec)')
        set(gca,'xtick',[stim_onset_frame stim_offset_frame])
        set(gca,'xticklabel',{'0','25'})
        ylabel('values')
        hold off
    end
    time_binned_accuracy_width=bin_length; %default 500, now 700
    time_binned_accuracy_onset= stim_onset_frame-time_binned_accuracy_width-1;%default 499 now 299
    time_binned_accuracy_offset=size(input_variable,2);
    time_binned_timepoints=time_binned_accuracy_onset:time_binned_accuracy_width:time_binned_accuracy_offset;

    binCols = parula(num_type_stim);
    binCols4=[binCols, ones(4,1)*0.2];
    meanPsycho=nan(size(psycho_across_animals,1),size(psycho_across_animals,2),numel(time_binned_timepoints)-1);
    this_timepoint=nan(size(psycho_across_animals,1),size(psycho_across_animals,2),numel(time_binned_timepoints)-1);
    psycho_fig=figure;
    for ibin = 1:numel(time_binned_timepoints)-1
    myFrames = time_binned_timepoints(ibin):time_binned_timepoints(ibin+1);% Define the frame range for each bin
    meanPsycho(:,:,ibin)=mean(psycho_across_animals(:,:,myFrames),3);
    this_timepoint(:,:,ibin)=ones(size(psycho_across_animals,1),size(psycho_across_animals,2)).*mean(myFrames);
    end

    for thisCoherence = 1:size(psycho_across_animals,2)

        myPlot = plot(mean(squeeze(psycho_across_animals(:,thisCoherence,:)),1),'LineWidth',1,'Color',binCols(thisCoherence,:));
        hold on
        %myPlot = plot(squeeze(psycho_across_animals(:,thisCoherence,:))','LineWidth',0.1,'Color',[.7 .7 .7]);
        plot(squeeze(this_timepoint(:,thisCoherence,:))',squeeze(meanPsycho(:,thisCoherence,:))','-','LineWidth',0.1,'Color',binCols4(thisCoherence,:));
        plot(mean(squeeze(this_timepoint(:,thisCoherence,:)),1),mean(squeeze(meanPsycho(:,thisCoherence,:)),1),'-o','LineWidth',2,'Color','k');
%         lgd=legend(gca);
%         txt=sprintf('%.0f%%',these_coherence(thisCoherence));
%         lgd.String{thisCoherence}=txt;
    end
    if analyse_angular_velocity
        title('Chronometric curves (rotational speed)');
    else
        title('Chronometric curves (orientation)');
    end
%     lgd.Location='northeast';
%     lgd.Interpreter='latex';
%     lgd.Title.String='Coherence level';
%     lgd.Box='off';
    xlabel('Time (sec)')
    set(gca,'xtick',[stim_onset_frame stim_offset_frame])
    set(gca,'xticklabel',{'0','25'})
    ylabel('Probability of Correct')
    ylim([0.4,1])
    hold off
    %% sort out data for modeling work
    %plot psychomatric curve
    these_coherence_direction=unique(behavioural_summary.stim_type);
    % Narrow down the analysis to binned timepoints
    time_binned_timepoints=time_binned_accuracy_onset:time_binned_accuracy_width:time_binned_accuracy_offset;
    %time_binned_timepoints=1:time_binned_accuracy_offset;%%[Optional] load entire timepoints
    time_binned_boolean=boolean_across_animals(:,:,time_binned_timepoints);
    %reshape matrix
    time_binned_boolean_reshape=pagectranspose(time_binned_boolean);
    stim_type_across_animals=reshape(behavioural_summary.stim_type,[size(boolean_across_animals,2),size(boolean_across_animals,1)]);
    %remove animals not engaged
    if active_animals_only
        animal_not_engaged=sort([4 8 16 17 20 36 39 66],'descend');
        for i= 1:numel(animal_not_engaged)
            boolean_across_animals(animal_not_engaged(i),:,:)=[];
            time_binned_boolean_reshape(:,animal_not_engaged(i),:)=[];
            stim_type_across_animals(:,animal_not_engaged(i))=[];
            threshold_across_animals(animal_not_engaged(i),:)=[];
        end
    else
        disp("analyse all animals");
    end
    fig=figure;binCols = parula(size(these_coherence_direction,2));
    timepoints_title=(time_binned_timepoints+1-stim_onset_frame)/frame_rate;
    for ibin = 1:numel(time_binned_timepoints)-1
        subplot(1,numel(time_binned_timepoints)-1,ibin);
        p_left=time_binned_boolean_reshape(:,:,ibin);
        for i = 1:size(these_coherence_direction,2)
           this_coherence=these_coherence_direction(i);
           this_response=p_left(stim_type_across_animals==this_coherence);
           SEM = std(this_response)./sqrt(size(this_response,1));
           mean_data=mean(this_response);
           scatter(this_coherence,mean_data,25,'MarkerEdgeColor',binCols(i,:),'MarkerFaceColor',binCols(i,:));
           %scatter(this_coherence,mean_data)
           eb = errorbar(this_coherence,mean_data,SEM, 'vertical', 'LineStyle', 'none');
           set(eb, 'color', binCols(i,:), 'LineWidth', 2)
           hold on
           title(sprintf('Time Point: %d',timepoints_title(ibin)));
           xlabel('Coherence level x Direction (%)')
           ylabel('Probability of Left (%)')
           ylim([0,1])
        end
    end
    set(fig, 'Position', [0, 0, 3000, 300]);
    stim_type_across_frames=repmat(transpose(stim_type_across_animals),[1,1,size(input_variable,2)]);
    x=repmat(1:size(input_variable,2),[numel(these_coherence_direction),1]);
    binCols = parula(size(these_coherence_direction,2));
    temp_this_response=nan(size(these_coherence_direction,2),size(stim_type_across_frames,3));
    for j = 1:size(stim_type_across_frames,3)
        coherence_this_frame=stim_type_across_frames(:,:,j);
        response_this_frame=boolean_across_animals(:,:,j);
        for i = 1:size(these_coherence_direction,2)
           this_coherence=these_coherence_direction(i);
           this_response=response_this_frame(coherence_this_frame==this_coherence);
           temp_this_response(i,j)=mean(this_response);
        end
    end
    for i = 1:size(these_coherence_direction,2)
        plot(x(i,:),temp_this_response(i,:),'LineWidth',1,'Color',binCols(i,:))
        hold on
        lgd=legend(gca,'FontSize',4);
        txt=sprintf('%.0f%%',these_coherence_direction(i));
        lgd.String{i}=txt;
    end
    lgd.Location='northeast';
    lgd.Interpreter='latex';
    lgd.Title.String='Coherence level';
    lgd.Box='off';
    xlabel('Time (sec)')
    set(gca,'xtick',[stim_onset_frame stim_offset_frame])
    set(gca,'xticklabel',{'0','25'})
    ylabel('Probability of Left')
    ylim([0,1])
    %save data for modeling work
    save('boolean_all_tav.mat', 'time_binned_boolean_reshape');
    save('threshold_across_animals.mat','threshold_across_animals')
    save('stim_type_across_animals.mat', 'stim_type_across_animals');
    save('time_binned_timepoints.mat', 'time_binned_timepoints');
    save('boolean_across_animals_tav.mat', 'boolean_across_animals');

    %use scatter plot if no line is prefered
    %scatter(repmat(1:numel(AUCs),[num_animals 1]),AUCs_across_animals,1,'k');
    % if color needs to be specified
    % [RowSorted, indices] = sort((total_travel_distance_across_animals), 'descend');
    % matrix_sorted = AUCs_across_animals(indices,:);
    % for i = 1:num_animals
    %     plot(AUCs_across_animals(i, :), 'color', binCols(i, :));
    %     hold on% Use each row from binCols as an RGB triplet
    % end
    % hold off;
else
    n_time_sample=size(input,2); % Number of time samples in the data
    stim_turn_fig=figure;
    orientation_limit=150;% Limit for y-axis orientation values
    y = linspace(1,n_time_sample,n_time_sample);
    plot(y,input)
    ylabel('Orientation (Rad)')
    title('Stimulus-evoked turning responses')
    ylim([-orientation_limit orientation_limit])
    set(gca,'ytick',[-orientation_limit 0 orientation_limit])
    set(gca,'yticklabel',[-orientation_limit 0 orientation_limit])
    xlabel('Time (sec)')
    set(gca,'xtick',[stim_onset_frame stim_offset_frame])
    set(gca,'xticklabel',{'0','25'})
    %saveas(stim_turn_fig,'stim_turn.jpg');

    these_stim= behavioural_summary.stim_type;
    this_input=input_variable;
    [AUCs,psycho,~,~]=roc_analysis(this_input,these_stim,this_folder,analysis_methods);
    close all
    if use_best_threshold
        disp('no AUCs across time when using best threshold in the experiment'); 
    else
        auc_across_animals=figure; hold on
        myPlot = plot(1:size(AUCs,2),AUCs,'LineWidth',1);
        myPlot.Color(4) = 0.1;
        if analyse_angular_velocity
            title('AUCs (delta orientation)');
        else
            title('AUCs (orientation)');
        end
        xlabel('Time (sec)')
        set(gca,'xtick',[stim_onset_frame stim_offset_frame])
        set(gca,'xticklabel',{'0','25'})
        ylabel('values')
        hold off
    end
    binCols = parula(num_type_stim);
    psycho_fig=figure;hold on
    for thisCoherence = 1:size(psycho,1)
        myPlot = plot(psycho(thisCoherence,:),'LineWidth',1,'Color',binCols(thisCoherence,:));
        lgd=legend(gca);
        txt=sprintf('%.0f%%',these_coherence(thisCoherence));
        lgd.String{thisCoherence}=txt;
    end
    if analyse_angular_velocity
        title('Chronometric curves (rotational speed)');
    else
        title('Chronometric curves (orientation)');
    end
    lgd.Location='northeast';
    lgd.Interpreter='latex';
    lgd.Title.String='Coherence level';
    lgd.Box='off';
    xlabel('Time (sec)')
    set(gca,'xtick',[stim_onset_frame stim_offset_frame])
    set(gca,'xticklabel',{'0','25'})
    ylabel('Probability of Correct')
    ylim([0.4,1])
    hold off
end



