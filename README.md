# General info about using fictrac + bonsai to run experiment on tethered animals:

## RoomZ745 or matrexVR room 

we try to keep the temperature between 30-32 degree during recording. This is meant to avoid introducing new variables into the behavioural matrics.

However, the setup does not let us to keep the temperature precise, so we still need to record the fluctuation of temperature. In Room745, when running bonsai-based experiment, the temperature is recored with SHT45 Temp+Hum module. when running Unity-based program, the temperature is recorded with DL220THP (or EL-USB-2). Note: LED panels can heat up the matrexVR setup up to 39 degree but monitors only heat up the set until 32 degree max (usually around 28 degree)

In the future, let's try to push this fluctuation only 1 degree differences

Below is the protocol when running experiments in Room Z745.

To raise the temperature on, keep the radiator on at level 5, turn on the air conditioner and set the temperature to 32 degree.

Turn on this computer, as monitors do contribute to a lot of heat to the setup

It takes around 3 hours to raise temperature from 27 degree to 30 degree so start this process early. The monitor will heat up the setup so eventually the setup will be around 32 degree.

Set up dehumitilifier overnight

## Animal preparation: 

animals should be left without food at least overnight (no more than over 2 nights) and they should be kept in VCC Zt611 

as there is automatic lighting to keep animals' biological clock consistent (make sure to turn off the automatic lighting system when not running new experiments). 

Keeping animals in Z building is possible but please be aware that the animals holding room can not go up to 30 degree (probably 27, 28 degree is the maximum already).

Tethering animals: teathering animal itself is not that hard but electrophysiology, we want to fix animals by head, which is not trivial (as locust's head is small and round). When not doing electrophysiology, I fix animals by thorax (the anterior part of the thorax) and apply additional drop in between head and thorax to dampen animals' head movement. 

For ephys, we used a 3D-printed headbar and for behaviours, we used needles. To glue animals, we use two kind of UHU glue for tethering. First use sand paper to get rid of wax on animal's head and thorax (for matrexVR, use the motorised grinder).

Then use a sharp thing to place the red UHU glue and stick it on the animal.

Then apply blue UHU glue on the holes of the headbar (or at the edge of needles) and use sharp thing to make sure the glue cover all edges. Shine the animal with UV light for 30 seconds from multiple angle.

(There is a small ball you can use while tethering the animal before the stage is available, just put the ball under a tethered animal, the animal shall climb it due to its instinct. 

But dont start tethering animals too early because even if you use the ball to keep locusts calm, locusts will eat the ball)

## Lighting: 

switch on extension located behind the rig. This will turn on both IR lights and OpenEphys DAQ. 

These light create some noises so they are wrapped with tin foil.

There are two ways to run the experiment, using online fictrac or offline fictrac. Online fictrac is essential for closed-loop experiment. 

However, due to some potential incompatability issue between AMD CPU and OpenCL, we drop a lot of frames when doing online fictrac analysis and saving frames into video 

(On average, 40% of the frame are dropped at 100Hz those videos can hardly be backtrack), so highly recommend to use bonsai to save videos when online fictrac is not necessary. When using online fictrac, it would be great to set up multi-camera 

filming so that cameras can save the video data

## Running the experiment (online fictrac):

Open Command Prompt

Press **Windows + LEFT twice** to move the command prompt to 2nd monitor

Switch to fictrac folder by typing
```
cd C:\src\fictrac\bin\Release
```
Then get fictrac ready to run by preparing the script
```
fictrac.exe config.txt
```
Standby

Open Bonsai workflow (for example rdk_3coherence_1hr)

Prepare the animal

Add optical glue to the animal and wait for the glue to dry out

Start the workflow

Start Ephys recording (Optional)

Move the animal from the tethering stage to the ephys rig

Closed the front cloth

(if using arduino trigger basler camera) press **C** to start the camera trigger

Press **Enter** to start the fictrac on the command prompt

Press **F** to start fictrac module

Press **R** to start recording

Press **S** to start stimulus


At the end of the experiment:

Stop Ephys recording (Optional)

Press **Ctrl + C** on Command Prompt to stop Fictrac

(if using arduino trigger basler camera) press **D** to stop the camera trigger

Stop workflow by pressing **Stop** on Bonsai GUI

Close the workflow (re-open the workflow if you want to continue with the next trial)

---------------------------------------------------------------------------------------------------------------------------------
The following command is used when you need to do posthoc analysis for led detection and fictrac, which means we only need to deliver stimuli and film the animal

Open Bonsai workflows (for example **rdk_3coherence_random_duration_336trials** and exploration workflow: **stimulus_toolkits**)

Prepare the animal

Add optical glue to the animal and wait for the glue to dry out

For ephys: inserting reference pin into the locust brain

Open Ephys GUI

Move the animal from the tethering stage to the ephys rig

Ephys only: Dip the electrode with dyes and wait for 5 mins so that the dyes can be stained on the electrode

Move the probe into the rig and stick the reference pins together. Attaching the remaining part of reference wire with tape to the headbar so that the animal can not touch it

Go deeper every 50um and deliver visual stimuli to find a good location with the exploration workflow

Closed the front cloth, and turn off room light (For ephys: turn off amd unplug micromanipulator, remove binoculars and open **rdk_3coherence_random_duration_336trials**)

Press **S** to start stimulus

press **C** to start the camera trigger (if using arduino trigger basler camera) then press **R** to start saving the incoming frames into video.

Starting Ephys recording

At the end of the experiment:

Stop Ephys recording (Optional)

Pressing **D** to stop arduino first (note that turning on LinLab GUI will automatically turn on the camera trigger)

Pressing Stop button to leave Bonsai GUI

## Posthoc analysis:

### fictrac

Open anaconda prompt and activate a virtual environment. 
```
conda activate tracking_analysis
```
Then move to fictrac folder
```
cd C:\src\fictrac\bin\Release
```
Then open a jupyter notebook called **fictrac_batch_analysis**

Run the first 3 cells to start batch processing of fictrac offline analysis

For an one-hour recording, it takes around 10 mins to finish the posthoc analysis

if not using the notebook, then just update the video directory in the config file and call

```
fictrac config_zball_camera1_9618_500x520.txt
```


### LED detection

Open **get_led_value_batch_process** in bonVision folder

Click **GetFiles** node and choose current path (basically the path where video files are kept)

And then run start workflow to extract led value from videos.

**NOTE** This does not open any windows for debugging so ensure your LED is indeed within the range of interest.

Use **get_led_value** first to have a rough idea where the ROI should be set.

Move those files into their respective folders

---------------------------------------------------------------------------------------------------------------------------------

# python environment for data analysis

## creating virtual environment and installing packages via commands

Below is the command to install packages via conda environment 

```
conda create --name tracking_analysis --channel conda-forge python=3.11
conda activate tracking_analysis
conda update -n base -c defaults conda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install --yes -c conda-forge -v numpy h5py opencv ipython jupyter matplotlib pandas matplotlib scipy openpyxl ipyparallel
```

## creating virtual environment from an environment.yml file or a requirement.txt
```
conda env create -f environment.yml
```
```
conda create --name tracking_analysis --file requirements.txt
```

[Optional] Below packages are used to read parquet file
```
conda install pyarrow fastparquet
```

When using timestamp to align stimulus onset with fictrac data, be aware that there is a recent fictrac bug outputting timestamp, which affects airball systems in Z745 but not in VCC. Hence, a customised code is written to extract timestamp from log file, which requires a package called **chardet**. 

```
pip install chardet
```

In addition, the log file is not encoded in utf-8 but in another scheme called iso-8859-5

Notably, this bug only applies to using timestamp to align stimuluus, so not the case when using LED sync pulse to align stimulus


Other than those packages, most of the package shall be the same as behavioural_analysis.

### Other packages required to run this analysis

There are useful tools stored in other packages so after you cloned this repo, remember to clone the other as well.

```
git clone https://github.com/chiyu1203/bonfic
git clone https://github.com/chiyu1203/utilities.git
```

## How to use the analysis pipeline and jupyter notebook

Use **bonfic_tutorial.ipynb**, which includes the essence of **analyse_stimulus_evoked_response.py** in this repo and **preprocess_fictrac_data** from **utilities** and plotting some group data.

The analysis methods are organised into a json file, which is created from a python file. This json file tells the notebook and python script how to analysis the data (with some boolean options) and include meta information of the experiment that is not included during data collection phase.

NOTE: from times to time, bonsai logs one trial into 2 rows, which causes error in sorting_trial_info

Therefore, each project (or each experiment) has its own json file. Below explains what those analysis methods are about.


    "overwrite_curated_dataset": whether to overwrite the existing pickle file or not, which is generated from fictrac. If True, then the programme will overwrite the pickle file

    "save_output": whether to save any output during data analysis. If True, then save any output
    
    "fictrac_posthoc_analysis": whether the fictrac data comes from online tracking or posthoc analysis. If True, then entering the analysis pipeline for posthoc analysis.
    
    "use_led_to_align_stimulus_timing": whether to use independent light source to track stimulus onset and offset. If True, then entering the analysis pipeline for that. If False, the programme will try to use timestamp coming from trials and from cameras to isolate the onset (this can only be achieved when posthoc analysis is not applied)
    
    "align_with_isi_onset": whether to use align the stimulus-evoked response based on stimulus onset or ISI onset. If True, then aligning the data based on ISI onset (this only works with Bonsai projects where ISI is logged)
    
    "mark_jump_as_nan": whether to use fft to identify jumping event in fictrac data and mark the event as nan

    "active_trials_only":  work in progress (to include walking trials)

    "filtering_method": what kind of filter to apply for tracking animals
    
    "plotting_tbt_overview": whether to plot a heat map of animal's velocity trial by trial and two histograms of animals travel distance and optomotor index.
    
    "plotting_trajectory": whether to plot the trajectory of the animals in the experiment.
    
    "plotting_event_related_trajectory": whether to plot a heat map of animal's trajectory after the stimulus onset
    
    "plotting_deceleration_accerleration": whether to plot the average deceleration and accerleration onset of the animals (this is still under construction)

    "plotting_position_dependant_fixation": whether to plot animal's angular velocity in response time. This is used in conflict experiment
    
    "plotting_optomotor_response": whether to plot the animal's mean optomotor index across stimulus type and the trajectory of yaw rotation vector trial by trial.
    
    "load_experiment_condition_from_database": this is used in the jypter notebook to see whether you want to access google-sheet-based database. If False, all animals stored in the server will be included.
    
    "select_animals_by_condition": this is used in the jypter notebook to extract specific animals for analysis. If True, you need to specify what condition in a dictionary. If False, all animals in the database (google sheet) will be included.
    
    "analysis_by_stimulus_type": this is used in the python script to see whether you want to plot the velocity heatmap based on stimlus type or based on time. If True, the heatmap will be stored based on stimulus type.




