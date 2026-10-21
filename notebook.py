import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Session 0: loading package depandancies and environmental variables
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    ## a useful function to generate a data list for further analysis
    import os,json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from analyse_stimulus_evoked_response import plot_yaw_angular_velocity
    from stimulus_gated_weibull import boolean_array_to_event_table
    from reliability.Probability_plotting import Exponential_probability_plot
    from reliability.Fitters import Fit_Weibull_2P,Fit_Everything,Fit_Exponential_1P,Fit_Weibull_DS,Fit_Weibull_Mixture,Fit_Exponential_2P
    from cumulative_probability_bootstrap import bootstrap_cumulative_probability
    from matplotlib.colors import ListedColormap
    custom_5 = ListedColormap(
        ["blue","cyan","red","magenta", "purple"],
        name="custom_5")
    return (
        Fit_Everything,
        Fit_Exponential_1P,
        Fit_Exponential_2P,
        Fit_Weibull_2P,
        Fit_Weibull_DS,
        Fit_Weibull_Mixture,
        Path,
        boolean_array_to_event_table,
        bootstrap_cumulative_probability,
        custom_5,
        json,
        np,
        pd,
        plot_yaw_angular_velocity,
        plt,
    )


@app.cell
def _(plt):
    ##introduce some helper class to define colour gradient
    import matplotlib as mpl
    from matplotlib import cm
    class MplColorHelper:
        def __init__(self, cmap_name, start_val, stop_val):
            self.cmap_name = cmap_name
            self.cmap = plt.get_cmap(cmap_name)
            self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
            self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        def get_rgb(self, val):
            return self.scalarMap.to_rgba(val)
    #colormap_name = "viridis"
    colormap_name = "coolwarm"
    # COL = MplColorHelper(colormap_name, 0, len(df["stim_type"].unique()))
    # sm = cm.ScalarMappable(cmap=colormap_name)
    return (mpl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session 0.1: Loading analysis methods into a dictionary variable
    """)
    return


@app.cell
def _():
    _folder_name = 'MatrexVR'
    _Datasets = 'Z:/DATA/experiment_trackball_Optomotor'
    this_dataset_dir = f'{_Datasets}/{_folder_name}'
    return (this_dataset_dir,)


@app.cell
def _(json):
    _file_name = './analysis_methods_dictionary.json'
    with open(_file_name, 'r') as _f:
        analysis_methods = json.loads(_f.read())
    return (analysis_methods,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Session 1: Loading dataset as pandas dataframe
    """)
    return


@app.cell
def _(Path, pd, this_dataset_dir):
    _file_name = 'matrex_vr_3coherence_database.parquet.gzip'
    df = pd.read_parquet(Path(this_dataset_dir) / _file_name)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Session 2: Visualising visually-evoked angular velocity
    """)
    return


@app.cell
def _(analysis_methods):
    _camera_fps=analysis_methods.get('camera_fps')
    analysis_window = analysis_methods.get('analysis_window')
    stim_duration = analysis_methods.get('stim_duration')
    init_frame = _camera_fps * abs(analysis_window[0])
    end_frame = _camera_fps * (abs(analysis_window[0]) + stim_duration)
    print(init_frame, end_frame)
    return end_frame, init_frame, stim_duration


@app.cell
def _(analysis_methods, df, end_frame, init_frame, plot_yaw_angular_velocity):
    _colormap_name = "viridis"
    fig,_fig_tbt=plot_yaw_angular_velocity(df,analysis_methods,init_frame,end_frame,"all",_colormap_name)
    _fig_tbt
    return (fig,)


@app.cell
def _(fig):
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Session 3: Analysing temporal patterns of the steering event
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session 3.1: Converting angular velocity into the steering event
    """)
    return


@app.function
def converting_follow_velocity(row):
    if row['stim_type'] < 0:
        row['stim_type'] = row['stim_type'] * -1  # Check condition on the first column
        row.loc['angular_velocity_f0':] = row.loc['angular_velocity_f0':] * -1
    return row


@app.cell
def _(df):
    df_angular_speed=df.apply(converting_follow_velocity, axis=1)
    return (df_angular_speed,)


@app.cell
def _(df_angular_speed, end_frame, init_frame):
    arr = df_angular_speed.iloc[:, init_frame+16:end_frame+16].values
    return (arr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session 3.2: A GUI to output fitting results with varying parameters
    """)
    return


@app.cell
def _(
    Fit_Exponential_1P,
    Fit_Exponential_2P,
    Fit_Weibull_2P,
    Fit_Weibull_DS,
    Fit_Weibull_Mixture,
    censored_data,
    model_choice,
):
    def fit_model(input_data):
        output=[]
        if censored_data.value:
            _faiures=input_data[input_data['event'] == 1]['time'].values
            _right_censored=input_data[input_data['event'] == 0]['time'].values
        else:
            _faiures=input_data[input_data['event'] == 1]['time'].values
            _right_censored=None
        if model_choice.value=="All of the above":
            _1=Fit_Weibull_DS(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
            _2=Fit_Weibull_Mixture(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
            _3=Fit_Weibull_2P(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
            _4=Fit_Exponential_2P(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
            _5=Fit_Exponential_1P(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
            output.append([_1,_2,_3,_4,_5])
        elif model_choice.value=="Weibull defective subpopulation":
            output=Fit_Weibull_DS(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
        elif model_choice.value=="Weibull mixtures":
            output=Fit_Weibull_Mixture(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
        elif model_choice.value=="Weibull":
            output=Fit_Weibull_2P(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
        elif model_choice.value=="2-parameter Exponential":
            output=Fit_Exponential_2P(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
        elif model_choice.value=="Exponential":
            output=Fit_Exponential_1P(failures=_faiures, right_censored=_right_censored, show_probability_plot=False)
        else:
            print("unidentified model")

    return (fit_model,)


@app.cell
def _(boolean_array_to_event_table, camera_fps, np):
    def prepare_input(
        _this_sample_size: int,_this_target_speed: float, _coherence_index: int,arr:np.ndarray, df):
        timepoints = np.arange(arr.shape[1]) / camera_fps
        coherence_levels=[0,33,66,100]
        _coherence_to_analyse=coherence_levels[_coherence_index]
        _hit_target_epochs = arr > _this_target_speed
        _trial_to_analyse = abs(df['stim_type']) == _coherence_to_analyse
        _this_data = _hit_target_epochs[df[_trial_to_analyse].index,]
        _output_table, _ = boolean_array_to_event_table(_this_data, timepoints=timepoints, stimulus_onset=0, consecutive_samples=_this_sample_size)
        return _output_table

    return (prepare_input,)


@app.cell
def _(mo):
    form = mo.ui.form(
        mo.ui.dictionary({
            "sample_size": mo.ui.slider(start=1, stop=100,value=42,step=1,show_value=True),
            "speed_thresholds": mo.ui.slider(start=0.1, stop=2, value=1,step=0.1,show_value=True),
            "coherence_levels": mo.ui.slider(start=0, stop=3, value=1, step=1, show_value=True)},label="parameters"),
        submit_button_label="Go, Hidden Scientists!",
        bordered=False
    )
    form
    return (form,)


@app.cell
def _(mo):
    censored_data = mo.ui.switch(value=True, label="include unsuccessful trials")
    model_choice = mo.ui.dropdown(
        options=["Weibull defective subpopulation","Weibull mixtures","Weibull","2-parameter Exponential","Exponential","All of the above"],
        value="Exponential",
        label="model choices",
    )
    mo.hstack([censored_data, model_choice], wrap=True, justify="start")
    return censored_data, model_choice


@app.cell
def _(arr, df, fit_model, form, mo, prepare_input):
    mo.stop(form.value is None, mo.md("Click **Go, Hidden Scientists!** to fit a model to the data"))
    _v = form.value
    input_data = prepare_input(_v["sample_size"], _v["speed_thresholds"], _v["coherence_levels"],arr,df)
    fit_model(input_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session 3.2: Visualising steering event
    """)
    return


@app.cell
def _(df, np):
    #parameter space to explore
    coherence_levels=np.sort(abs(df['stim_type']).unique())
    sample_sizes = np.arange(8, 101, 7)
    _stimulus_speed = 64 * np.pi / 180
    speed_thresholds = [round(0.5 * _stimulus_speed, 2), round(_stimulus_speed, 2), round(2 * _stimulus_speed, 2)]
    return coherence_levels, sample_sizes, speed_thresholds


@app.cell
def _(analysis_methods, arr, np):
    onset_in_sec = 0
    camera_fps = analysis_methods.get('camera_fps')
    timepoints = np.arange(arr.shape[1]) / camera_fps
    post_times = timepoints[timepoints > onset_in_sec] - onset_in_sec
    return camera_fps, onset_in_sec, post_times, timepoints


@app.cell
def _(
    arr,
    boolean_array_to_event_table,
    bootstrap_cumulative_probability,
    coherence_levels,
    df,
    onset_in_sec,
    plt,
    post_times,
    sample_sizes,
    speed_thresholds,
    stim_duration,
    timepoints,
):
    colours = ['y', 'c', 'm', 'k']
    _use_single_threshold = True
    if _use_single_threshold:
        _stimulus_speed = speed_thresholds[1]
        _hit_target_epochs = arr > _stimulus_speed
    else:
        _stimulus_speed = [speed_thresholds[0], speed_thresholds[2]]
        _hit_target_lower = arr > _stimulus_speed[0]
        _hit_target_higher = arr < _stimulus_speed[1]
        _hit_target_epochs = _hit_target_higher & _hit_target_lower
    _fig, _axes = plt.subplots(nrows=1, ncols=sample_sizes.shape[0], figsize=(24, 5), tight_layout=True, sharex=True, sharey=True)
    for _j, _this_sample_size in enumerate(sample_sizes):
        for _i, _this_coherence in enumerate(coherence_levels):
            _trial_to_analyse = abs(df['stim_type']) == _this_coherence
            _this_data = _hit_target_epochs[df[_trial_to_analyse].index,]
            _, _qualifying = boolean_array_to_event_table(_this_data, timepoints=timepoints, stimulus_onset=onset_in_sec, consecutive_samples=_this_sample_size)
            probability, lower, upper = bootstrap_cumulative_probability(_qualifying, n_resamples=99, confidence_level=0.95, seed=42, subject_ids=None)
            event_times = post_times[:_qualifying.shape[1]]
            # _axes[_j].step(event_times, probability, where='post', color=colours[_i], label='Cumulative probability')
            #_axes[_j].fill_between(event_times, lower, upper, step='post', color=colours[_i], alpha=0.25, label='95% pointwise bootstrap CI')
            _axes[_j].plot(event_times, probability, color=colours[_i], label=f"{_this_coherence}%")
            _axes[_j].fill_between(event_times, lower, upper,color=colours[_i], alpha=0.25)
            if _j==0:
                _axes[_j].set(xlabel='Time since stimulus onset (s)', ylabel='Probability of Hit (%)', xticks=[onset_in_sec, stim_duration], ylim=(0, 1), yticks=[0, 0.5, 1], yticklabels=['0', '50', '100'])
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Session 4: Applying survival analysis/reliability engineering to steering events
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session4.1: Using Fit_everything function to compare the performance of each model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### The Fit_everything function can output parameter values, log-likelihood, AIC and BIC.
    """)
    return


@app.cell
def _(
    Fit_Everything,
    arr,
    boolean_array_to_event_table,
    coherence_levels,
    df,
    np,
    onset_in_sec,
    sample_sizes,
    speed_thresholds,
    timepoints,
):
    _number_of_models=5
    _coherence_to_analyse=coherence_levels[3]
    ready_to_plot_AIC = np.ones((len(speed_thresholds), sample_sizes.shape[0], _number_of_models))
    ready_to_plot_BIC = np.ones((len(speed_thresholds), sample_sizes.shape[0], _number_of_models))
    for _i, _this_target_speed in enumerate(speed_thresholds):
        _hit_target_epochs = arr > _this_target_speed
        for _j, _this_sample_size in enumerate(sample_sizes):
            _trial_to_analyse = abs(df['stim_type']) == _coherence_to_analyse
            _this_data = _hit_target_epochs[df[_trial_to_analyse].index,]
            _time_to_success_pd, _ = boolean_array_to_event_table(_this_data, timepoints=timepoints, stimulus_onset=onset_in_sec, consecutive_samples=_this_sample_size)
            every = Fit_Everything(failures=_time_to_success_pd[_time_to_success_pd['event'] == 1]['time'].values, right_censored=_time_to_success_pd[_time_to_success_pd['event'] == 0]['time'].values, exclude=['Normal_2P', 'Gumbel_2P', 'Lognormal_2P', 'Lognormal_3P', 'Loglogistic_2P', 'Loglogistic_3P', 'Gamma_2P', 'Gamma_3P', 'Weibull_CR', 'Weibull_3P'], show_probability_plot=False, print_results=False, show_best_distribution_probability_plot=False, show_histogram_plot=False, show_PP_plot=False)
            sorted_pd = every.results.sort_values('Distribution')
            ready_to_plot_AIC[_i, _j, :] = sorted_pd['AICc'].values
            ready_to_plot_BIC[_i, _j, :] = sorted_pd['BIC'].values
    return (ready_to_plot_AIC,)


@app.cell
def _(custom_5, np, plt, ready_to_plot_AIC, sample_sizes):
    _xlims = [(13000, 17500), (9500, 17000), (3000, 13500)]
    _yticks = [8, 50, 99]
    ix, iy, iz = np.indices(ready_to_plot_AIC.shape)
    _fig, _axes = plt.subplots(nrows=3, ncols=1, figsize=(30, 20), tight_layout=True, sharey=False)  #sharey='row'
    for _i in range(ready_to_plot_AIC.shape[0]):
        _axes[_i].scatter(ready_to_plot_AIC[_i, :, :], np.transpose(np.tile(sample_sizes, (ready_to_plot_AIC.shape[2], 1))), s=40, c=iz[_i, :, :], cmap=custom_5, marker='^', alpha=1)
        _axes[_i].set_xlim(_xlims[_i])
        _axes[_i].set_xticks(_xlims[_i])
        _axes[_i].set_yticks(_yticks)
        for _label in _axes[_i].get_xticklabels():
            _label.set_fontsize(20)  #axes[i].set_xscale('log')
        for _label in _axes[_i].get_yticklabels():
            _label.set_fontsize(20)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### To output the confidence interval of each parameter, it is required to apply speicific fitting functions, for example: Fit_Weibull_2P, these functions accept the same input arguement.
    """)
    return


@app.cell
def _(
    Fit_Weibull_2P,
    Fit_Weibull_DS,
    arr,
    boolean_array_to_event_table,
    coherence_levels,
    df,
    np,
    onset_in_sec,
    sample_sizes,
    speed_thresholds,
    timepoints,
):
    _coherence_to_analyse=coherence_levels[3]
    _consider_censored_data=True
    ready_to_plot_kappa = np.zeros((len(speed_thresholds), sample_sizes.shape[0]))
    ready_to_plot_kappa_lower = np.zeros((len(speed_thresholds), sample_sizes.shape[0]))
    ready_to_plot_kappa_upper = np.zeros((len(speed_thresholds), sample_sizes.shape[0]))
    for _i, _this_target_speed in enumerate(speed_thresholds):
        _hit_target_epochs = arr > _this_target_speed
        for _j, _this_sample_size in enumerate(sample_sizes):
            _trial_to_analyse = abs(df['stim_type']) == _coherence_to_analyse
            _this_data = _hit_target_epochs[df[_trial_to_analyse].index,]
            output_table, _qualifying = boolean_array_to_event_table(_this_data, timepoints=timepoints, stimulus_onset=onset_in_sec, consecutive_samples=_this_sample_size)
            if _consider_censored_data:
                _wbf = Fit_Weibull_DS(failures=output_table[output_table['event'] == 1]['time'].values, right_censored=output_table[output_table['event'] == 0]['time'].values, show_probability_plot=False, print_results=True)
            else:
                _wbf = Fit_Weibull_2P(failures=output_table[output_table['event'] == 1]['time'].values, show_probability_plot=False, print_results=True)
            ready_to_plot_kappa[_i, _j] = _wbf.beta
            ready_to_plot_kappa_lower[_i, _j] = _wbf.beta_lower
            ready_to_plot_kappa_upper[_i, _j] = _wbf.beta_upper
    return (ready_to_plot_kappa,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Session 4.1: Using heatmap to visualise shape parameter
    """)
    return


@app.cell
def _(np, plt):
    def heatmap(data, row_labels, col_labels, ax=None,
                cbar_kw=None, cbarlabel="", **kwargs):
        """
        Create a heatmap from a numpy array and two lists of labels.

        Parameters
        ----------
        data
            A 2D numpy array of shape (M, N).
        row_labels
            A list or array of length M with the labels for the rows.
        col_labels
            A list or array of length N with the labels for the columns.
        ax
            A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
            not provided, use current Axes or create a new one.  Optional.
        cbar_kw
            A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
        cbarlabel
            The label for the colorbar.  Optional.
        **kwargs
            All other arguments are forwarded to `imshow`.
        """

        if ax is None:
            ax = plt.gca()

        if cbar_kw is None:
            cbar_kw = {}

        # Plot the heatmap
        im = ax.imshow(data,cmap='coolwarm',vmin=0,vmax=2,**kwargs)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal", location='top',aspect=80 ,**cbar_kw)
        cbar.ax.set_ylabel(cbarlabel,rotation=-90,va="bottom")


        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(range(data.shape[1]), labels=col_labels,
                      rotation=-30, rotation_mode="xtick")
        ax.set_yticks(range(data.shape[0]), labels=row_labels)
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=False, bottom=True,
                       labeltop=False, labelbottom=True)

        # Turn spines off and create white grid.
        ax.spines[:].set_visible(False)

        ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
        ax.set_xlabel("step size (n consecutive frames)", fontsize=14)
        ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
        ax.set_ylabel(r'threshold (rad s$^{-1}$)', fontsize=14)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        return im, cbar

    return (heatmap,)


@app.cell
def _(mpl, np):
    def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                         textcolors=("black", "white"),
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max())/2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = mpl.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    return (annotate_heatmap,)


@app.cell
def _(
    annotate_heatmap,
    heatmap,
    ready_to_plot_kappa,
    sample_sizes,
    speed_thresholds,
):
    im, cbar = heatmap(ready_to_plot_kappa, list(map(str, speed_thresholds)), sample_sizes.astype(str).tolist())
    texts = annotate_heatmap(im, valfmt='{x:.2f}', fontsize=8)
    im
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Session 5: Others in the dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Session 5.1: visualising travel distance in the dataset
    """)
    return


@app.cell
def _(df, plt):
    _fig, (_ax1,_ax2) = plt.subplots(nrows=1, ncols=2,figsize=(5, 5))
    df_aba = df.groupby('ID').agg(travel_distance=('travel_distance', 'mean'), travel_distance_isi=('travel_distance_isi', 'mean'), ID=('ID', 'first'))
    _ax1.hist(df_aba['travel_distance'])
    _ax2.hist(df['travel_distance'], bins=100)
    _ax2.axvline(x=10, ymin=0, ymax=1, color='red', ls='--')
    _fig
    return


if __name__ == "__main__":
    app.run()
