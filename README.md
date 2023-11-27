# General info about using fictrac + bonsai to run experiment on tethered animals:

### RoomZ745: 

we try to keep the temperature between 30-32 degree during recording. This is meant to avoid introducing new variables into the behavioural matrics. 

However, the setup does not let us to keep the temperature precise, so we still need to record the fluctuation of temperature. 

In the future, let's try to push this fluctuation only 1 degree differences

To raise the temperature on, keep the radiator on at level 5, turn on the air conditioner and set the temperature to 32 degree.

Turn on this computer, as monitors do contribute to a lot of heat to the setup

It takes around 3 hours to raise temperature from 27 degree to 30 degree so start this process early.

### Animal preparation: 

animals should be left without food at least overnight (no more than over 2 nights) and they should be kept in VCC Zt611 

as there is automatic lighting to keep animals' biological clock consistent (make sure to double check with me how to use the automatic lighting system). 

Keeping animals in Z building is possible but please be aware that the animals holding room can not go up to 30 degree (I need to double check. this probably 27, 28 degree is the maximum already).

Tethering animals: teathering animal itself is not that hard but electrophysiology, we want to fix animals by head, which is not trivial (as locust's head is small and round). 

So far, we used Pavan's needle to tether animals and try to glue their head and thorax together so from time to time, animals do escape from the tethering, which is really scary for electrophysiology. 

Imagine locusts got rid of the tether and headfix and break your 6000-euro probes :(

We use two kind of UHU glue for tethering. First use sand paper to get rid of wax on animal's head and thorax. 

Then dip the tip of the customised needles with red UHU glue and stick it on the animal.

Then apply blue UHU glue around the needles and the animal. Shine the animal with UV light for 30 seconds from multiple angle.

(There is a small ball you can use while tething the animal before the stage is avaiable, just put the ball under a tethered animal, the animal shall climb it due to its instinct. 

But dont start tethering animals too early because even if you use the ball to keep locusts calm, locusts will eat the ball)

### Lighting: 

switch on extension located at the bottom of the rig. This will turn on both IR light by the side of the camera and a light above the animal. 

It seems that these light will create some noise so I need to test them and perhaps replacing them with othe noise-free lighting.

There are two ways to run the experiment, using online fictrac or offline fictrac. Online fictrac is essential for closed-loop experiment. 

However, due to some potential incompatability issue between AMD CPU and OpenCL, we drop a lot of frames when doing online fictrac analysis and saving frames into video 

(On average, 40% of the frame are dropped at 100Hz. This means those videos can hardly be backtrack).


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

Press **S** to start stimulus


At the end of the experiment:

Stop Ephys recording (Optional)

Press **Ctrl + C** on Command Prompt to stop Fictrac

(if using arduino trigger basler camera) press **D** to stop the camera trigger

Stop workflow by pressing **Stop** on Bonsai GUI

Close the workflow (re-open the workflow if you want to continue with the next trial)

---------------------------------------------------------------------------------------------------------------------------------
The following command is used when you need to do posthoc analysis for led detection and fictrac, which means we only need to deliver stimuli and film the animal

Open Bonsai workflow (for example **rdk_3coherence_1hr_posthoc**)

Prepare the animal

Add optical glue to the animal and wait for the glue to dry out

Start the workflow

Start Ephys recording (Optional)

Move the animal from the tethering stage to the ephys rig

Closed the front cloth

(if just let camera run as default) press **R** to start saving the incoming frames into video
(if using arduino trigger basler camera) press **C** to start the camera trigger

Press **S** to start stimulus

At the end of the experiment:

Stop Ephys recording (Optional)

Stop workflow by pressing Stop on Bonsai GUI

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

### LED detection

Open **get_led_value_batch_process** in bonVision folder

Click **GetFiles** node and choose current path (basically the path where video files are kept)

And then run start workflow to extract led value from videos.

**NOTE** This does not open any windows for debugging so ensure your LED is indeed within the range of interest.

Use **get_led_value** first to have a rough idea where the ROI should be set.

Move those files into their respective folders

---------------------------------------------------------------------------------------------------------------------------------

### Additional package for data analysis

When using timestamp to align stimulus onset with fictrac data, be aware that there is a recent fictrac bug outputting timestamp.

This affects Zball system but not VCC ball system. Hence, a customised code is writen to extract timestamp from log file, which requires a package called **chardet**. 

Otherwise, most of the package shall be the same as behavioural_analysis (need to check this soon though). 
```
pip install chardet
```
In addition, the log file is not encoded in utf-8 but in another scheme called iso-8859-5

Notably, this bug only applies to using timestamp to align stimuluus, so not the case when using LED sync pulse to align stimulus
