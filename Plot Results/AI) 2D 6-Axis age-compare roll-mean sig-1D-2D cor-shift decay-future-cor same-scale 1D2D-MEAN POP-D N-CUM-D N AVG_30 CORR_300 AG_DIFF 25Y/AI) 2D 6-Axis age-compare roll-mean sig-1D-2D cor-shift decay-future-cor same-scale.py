import json  
import pandas as pd
import plotly.graph_objects as go
import os
import shutil
from plotly.subplots import make_subplots
import numpy as np
import textwrap
import plotly.express as px
import xarray as xr
import numpy as np
import scipy.stats as stats
from matplotlib import colors as mcolors 
import csv
import colorsys

# This script processes data from pivot CSV files located in the TERRA folder, 
# which were generated from a Czech Freedom of Information request (Vesely_106_202403141131.csv). 
# The pivot files were created using the DB Browser for SQLite.

def main():

    # Create an instance of PlotConfig
    plot_config = PlotConfig(
        title_text="1st 2nd derivate - decay time - roll correl significance - pahse shift correl", # Title of plot
        plot_name_append_text="AG_DIFF 25Y",       # apend text - to plotfile name and directory, to save in uniqe file location
        window_size_correl=300,         # window size for rolling pearson correlation
        max_phase_shift=300,            # max phase shift for phase shift correlation
        window_size_mov_average=30,     # window size for moving means
        target_decay_time=70,           # paramter to adjust decay target_percentage to calculate the dacy time 
        normalize=True,                 # normalize dvd values
        normalize_cumulate_deaths=True, # normalize cumulated deaths bevore cummulation
        population_minus_death=True,    # deducts the deceased from the total population
        custom_legend_column = 10,      # A change could require adjustments to the code. 
        axis_y1_logrithmic=True,        # show logarithmic y1 axis
        savetraces_to_csv=False,        # save calcualted results of all traces into a csv file
        calc_derivate_mean = True       # smooth 1sd and 2nd derivate by moving mean before and after calc
    )

    # If you're calculating pearson significance and shift correlation on raw data without rolling mean, 
    # you'll be dealing with the raw variability in the data,
    # this can lead to spurious results by the  noise in the data. 
    # In most time-series analyses, smoothing is recommended to help reduce this
           
    pairs = [
            ('Avg NUM_VDA','Avg NUM_DVDA'),
            ('Avg NUM_VDA','Avg NUM_D'),
            ('Avg NUM_VDA','Avg NUM_DVX'),
            ('Avg NUM_VDA','Avg NUM_DUVX'),
            ('Avg 1D NUM_VDA','Avg 1D NUM_DVDA'),
            ('Avg 1D NUM_VDA','Avg 1D NUM_D'),
            ('Avg 1D NUM_VDA','Avg 1D NUM_DVX'),
            ('Avg 1D NUM_VDA','Avg 1D NUM_DUVX'),
            ('Avg 2D NUM_VDA','Avg 2D NUM_DVDA'),
            ('Avg 2D NUM_VDA','Avg 2D NUM_D'),
            ('Avg 2D NUM_VDA','Avg 2D NUM_DVX'),
            ('Avg 2D NUM_VDA','Avg 2D NUM_DUVX'),
            ('Avg 2D NUM_VDA','Avg Decay NUM_DVDA'),
            ('Avg 2D NUM_VDA','Avg Decay NUM_D'),            
            ('Avg 2D NUM_VDA','Avg Decay NUM_DVX'),
            ('Avg 2D NUM_VDA','Avg Decay NUM_DUVX'),
            ('Avg Decay NUM_D', 'Avg Decay NUM_DVX'),
            ('Avg Decay NUM_D', 'Avg Decay NUM_DUVX'),
            ('Avg Decay NUM_D', 'Avg Decay NUM_DVDA'),
            ('Avg Decay NUM_DVX', 'Avg Decay NUM_DUVX'),
            ('Avg Decay NUM_DVX', 'Avg Decay NUM_DVDA'),
            ('Avg Decay NUM_DUVX', 'Avg Decay NUM_DVDA'),
            ('Avg NUM_VDA','Avg Decay NUM_DVDA'),
            ('Avg NUM_VDA','Avg Decay NUM_D'),
            ('Avg NUM_VDA','Avg Decay NUM_DVX'),
            ('Avg NUM_VDA','Avg Decay NUM_DUVX'),
            ('Avg NUM_D', 'Avg Decay NUM_D'),
            ('Avg NUM_DVX', 'Avg Decay NUM_DVX'),
            ('Avg NUM_DUVX', 'Avg Decay NUM_DUVX'),
            ('Avg NUM_DVDA', 'Avg Decay NUM_DVDA'),        
    ]
    
    
    # Comment this line in - if you want to show pairs text in the plot annotation: 
    # plot_config.update_pairs(pairs)
    

    # age_band_pairs is used to compare traces between age bands and calculate 
    # significance correlation and correlation shift. 
    # A new list, updated_age_band_pairs, is generated for each tuple in 
    # the age_band_compare list below.
    #
    # The output will be: 
    # updated_age_band_pairs = [
    #    ('Avg NUM_VDA 55', 'Avg NUM_VDA 75'),
    #    ('Avg NUM_D 55', 'Avg NUM_D 75'),
    #    ...
    # ]
    age_band_pairs = [
        ('Avg NUM_VDA', 'Avg NUM_VDA'),
        ('Avg NUM_D', 'Avg NUM_D'),
        ('Avg NUM_DVX', 'Avg NUM_DVX'),
        ('Avg NUM_DUVX', 'Avg NUM_DUVX'),
        ('Avg NUM_DVDA', 'Avg NUM_DVDA'),         
        ('Avg 1D NUM_D', 'Avg 1D NUM_D'),
        ('Avg 1D NUM_DVX', 'Avg 1D NUM_DVX'),
        ('Avg 1D NUM_DUVX', 'Avg 1D NUM_DUVX'),
        ('Avg 1D NUM_DVDA', 'Avg 1D NUM_DVDA'), 
        ('Avg 2D NUM_D', 'Avg 2D NUM_D'),
        ('Avg 2D NUM_DVX', 'Avg 2D NUM_DVX'),
        ('Avg 2D NUM_DUVX', 'Avg 2D NUM_DUVX'),
        ('Avg 2D NUM_DVDA', 'Avg 2D NUM_DVDA'), 
        ('Avg Decay NUM_D', 'Avg Decay NUM_D'),
        ('Avg Decay NUM_DVX', 'Avg Decay NUM_DVX'),
        ('Avg Decay NUM_DUVX', 'Avg Decay NUM_DUVX'),
        ('Avg Decay NUM_DVDA', 'Avg Decay NUM_DVDA'),        
    ]

    # List of tuples with the age bands you want to compare
    age_band_compare = [
        ('0-4', '25-29'),
        ('5-9', '30-34'),
        ('10-14', '35-39'),
        ('15-19', '40-44'),
        ('20-24', '45-49'),
        ('25-29', '50-54'),
        ('30-34', '55-59'),
        ('35-39', '60-64'),
        ('40-44', '65-69'),
        ('45-49', '70-74'),
        ('50-54', '75-79'),
        ('55-59', '80-84'),
        ('60-64', '85-89'),
        ('65-69', '90-94'),
        ('70-74', '95-99'),
        ('75-79', '100-104'),
        ('80-84', '105-109'),
        ('85-89', 'gr109'),
    ]
    
    # CSV file pairs with age_band with death and population/doses data  
    csv_files_dvd = [
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_D.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DUVX.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVX.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVD1.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVD2.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVD3.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVD4.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVD5.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVD6.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVD7.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVDA.csv",
    ]
    csv_files_vd = [
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_POP.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_UVX.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VX.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VD1.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VD2.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VD3.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VD4.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VD5.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VD6.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VD7.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VDA.csv",
    ]
    
    # CSV file pairs used to calcualate decay curves
    csv_files_decay_dvd = [       
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_D.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DUVX.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVX.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_DVDA.csv"
    ]

    csv_files_decay_vd = [
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_POP.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_UVX.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VX.csv",
        r"C:\Github\CzechFOI-DA\TERRA\PVT_NUM_VDA.csv"
    ]

    try:
        dataframes_dvd = [pd.read_csv(file) for file in csv_files_dvd]
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")

    try:
        dataframes_vd = [pd.read_csv(file) for file in csv_files_vd]   
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")
    
    try:
        dataframes_decay_dvd = [pd.read_csv(file) for file in csv_files_decay_dvd]
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")

    try:
        dataframes_decay_vd = [pd.read_csv(file) for file in csv_files_decay_vd]   
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")

    # Initialize the directory and copy py script
    # full_plotfile_name = init_function(f"{plot_config_data['name']} {plot_config_data['pairs_text']}")
    full_plotfile_name = init_function(f"{plot_config.plot_name}")

    # Get the color shades for the current standard dataset traces (legends)
    # 11 rows x 10 columns
    color_palette = px.colors.qualitative.Dark24
    color_palette_r = px.colors.qualitative.Dark24_r
    # Generate shades for all color pairs
    color_shades = generate_color_shades(color_palette, n_pairs=11)
    color_shades_r = generate_color_shades(color_palette_r, n_pairs=11)
                              
    # Loop through each pair of age bands in the list
    pair_nr = 0    
    for age_band_pair in age_band_compare:    
        # Create an instance of TraceManager
        trace_manager = TraceManager()
        # Generate updated pairs by calling the function with a single tuple
        updated_age_band_pairs = generate_age_band_pairs(age_band_pair, age_band_pairs)
        # Print the updated pairs
        for pair in updated_age_band_pairs: print(pair)

        # Get the shades for the current index and age band pair
        for idx, age_band in enumerate(age_band_pair):

            # Add traces for each dataframe (CSV-file)
            for i in range(0, len(dataframes_dvd)):   
                
                # Get the color shades for the current dataset (ensure the shades list is long enough)
                if idx == 0:   # First age band
                   shades_1, shades_2 = color_shades[i]                    
                elif idx == 1:  # Second age band reversd coloros
                   shades_1, shades_2 = color_shades_r[i]                
                
                age_band_extension = age_band.split('-')[0]  
                                                  
                # Calculate cumulative VD data trace
                if i == 0:  # csv file[0] POP
                    if plot_config.population_minus_death:
                        # POP - cum D
                        cum_dataframes_vd = dataframes_vd[0][age_band] - dataframes_dvd[0][age_band].cumsum() 
                    else :
                        # POP
                        cum_dataframes_vd = dataframes_vd[0][age_band]                
                elif i == 1: # csv file [1] UVX
                    if plot_config.population_minus_death:
                        # POP - cum D - cum VX
                        cum_dataframes_vd = dataframes_vd[0][age_band] - dataframes_dvd[1][age_band].cumsum() + dataframes_vd[1][age_band].cumsum() 
                    else :
                        # POP - cum VX
                        cum_dataframes_vd =  dataframes_vd[0][age_band]  + dataframes_vd[1][age_band].cumsum() 
                else:   # csv files [i]
                    if plot_config.population_minus_death:
                        # VX..VDX - cum D 
                        cum_dataframes_vd = dataframes_vd[i][age_band].cumsum() - dataframes_dvd[i][age_band].cumsum()
                    else :
                        # VX..VDX
                        cum_dataframes_vd = dataframes_vd[i][age_band].cumsum()   
                                             
                # Normalize the data per 100,000 
                norm_dataframes_dvd = (dataframes_dvd[i][age_band] / cum_dataframes_vd) * 100000
                if plot_config.normalize:                    
                    plt_dataframes_dvd = norm_dataframes_dvd 
                else :
                    plt_dataframes_dvd = dataframes_dvd[i][age_band]                
                
                if idx==0: yaxis='y1' 
                else: yaxis='y1'                    
                # Add traces for DVD on primary y-axis
                trace_manager.add_trace(
                    name=f'{os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {age_band_extension}', 
                    x=dataframes_dvd[i].iloc[:, 0],
                    y=plt_dataframes_dvd,
                    mode='lines',
                    line=dict(dash='solid', width=1, color=shades_1[0]),                    
                    secondary=False,
                    axis_assignment=yaxis)

                if idx==0: yaxis='y1' 
                else: yaxis='y1'
               # Calculate add moving average trace for DVD
                moving_average_dvd = plt_dataframes_dvd.rolling(window=plot_config.window_size_mov_average).mean()                                
                trace_manager.add_trace(
                    name=f'Avg {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {age_band_extension}', 
                    x=dataframes_dvd[i].iloc[:, 0], 
                    y=moving_average_dvd,
                    line=dict(dash='solid', width=1, color=shades_1[1]), 
                    secondary=False, 
                    axis_assignment=yaxis)

                # Calculate the first derivative (approximate as the difference between consecutive moving averages)
                first_derivative_dvd = moving_average_dvd.diff()
                # Initialize variables for the first and second derivatives
                smoothed_first_derivative_dvd = first_derivative_dvd  # Default to first derivative
                second_derivative_dvd = None

                # Scenario 1: Smooth the first derivative before calculating the second derivative
                if plot_config.calc_derivate_mean:
                    # Smooth the first derivative
                    smoothed_first_derivative_dvd = first_derivative_dvd.rolling(window=plot_config.window_size_mov_average).mean()

                    # Calculate the second derivative from the smoothed first derivative
                    second_derivative_dvd = smoothed_first_derivative_dvd.diff()
                    second_derivative_dvd = second_derivative_dvd.rolling(window=plot_config.window_size_mov_average).mean()                    
                else:
                    # Calculate the second derivative directly from the first derivative
                    second_derivative_dvd = first_derivative_dvd.diff()

                if idx==0: yaxis='y5' 
                else: yaxis='y5'
                # Add trace for the moving average of the first derivative
                trace_manager.add_trace(
                    x=dataframes_dvd[i].iloc[:, 0],
                    y=smoothed_first_derivative_dvd,
                    mode='lines',
                    line=dict(dash='solid', width=1, color=shades_1[2]),
                    name=f'Avg 1D {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {age_band_extension}',
                    secondary=True,
                    axis_assignment=yaxis)  

                if idx==0: yaxis='y5' 
                else: yaxis='y5'
                # Add trace for the moving average of the second derivative
                trace_manager.add_trace(
                    x=dataframes_dvd[i].iloc[:, 0],
                    y=second_derivative_dvd,
                    mode='lines',
                    line=dict(dash='solid', width=1, color=shades_1[3]),
                    name=f'Avg 2D {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {age_band_extension}',
                    secondary=True,
                    axis_assignment=yaxis)
                
                # First normalize deaths then cummulate 
                if plot_config.normalize_cumulate_deaths:                   
                   cum_dataframes_dvd = norm_dataframes_dvd.cumsum()
                   #print (f'norm: {norm_dataframes_dvd}')
                else:
                   cum_dataframes_dvd = dataframes_dvd[i][age_band].cumsum()   
                   #print (f'cum: {norm_dataframes_dvd}')
                
                if idx==0: yaxis='y3' 
                else: yaxis='y3'
                # Add cumulative DVD data trace on the secondary y-axis
                trace_manager.add_trace(
                    name=f'cum {os.path.splitext(os.path.basename(csv_files_dvd[i]))[0][4:]} {age_band_extension}', 
                    x=dataframes_dvd[i].iloc[:, 0],  
                    y=cum_dataframes_dvd,            
                    line=dict(dash='dot', width=1.5, color=shades_1[4]),
                    secondary=True,
                    axis_assignment=yaxis)

                if idx==0: yaxis='y2' 
                else: yaxis='y2'
                # Add trace for VD
                trace_manager.add_trace(
                    name=f'{os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {age_band_extension}',
                    x=dataframes_vd[i].iloc[:, 0],  
                    y=dataframes_vd[i][age_band],   
                    line=dict(dash='solid', width=1,  color=shades_2[0]), 
                    secondary=False, 
                    axis_assignment=yaxis)
                
                if idx==0: yaxis='y2' 
                else: yaxis='y2'
                # Calculate add moving average trace for VD
                moving_average_vd = dataframes_vd[i][age_band].rolling(window=plot_config.window_size_mov_average).mean()                
                trace_manager.add_trace(
                    name=f'Avg {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {age_band_extension}',                    
                    x=dataframes_vd[i].iloc[:, 0],
                    y=moving_average_vd,
                    line=dict(dash='solid', width=1,  color=shades_2[1]),
                    secondary=False, 
                    axis_assignment=yaxis)                
                
                # Calculate the first derivative (approximate as the difference between consecutive moving averages)
                first_derivative_vd = moving_average_vd.diff()
                # Initialize variables for the first and second derivatives
                smoothed_first_derivative_vd = first_derivative_vd  # Default to first derivative
                second_derivative_vd = None

                # Scenario 1: Smooth the first derivative before calculating the second derivative
                if plot_config.calc_derivate_mean:
                    # Smooth the first derivative
                    smoothed_first_derivative_vd = first_derivative_vd.rolling(window=plot_config.window_size_mov_average).mean()

                    # Calculate the second derivative from the smoothed first derivative
                    second_derivative_vd = smoothed_first_derivative_vd.diff()
                    second_derivative_vd = second_derivative_vd.rolling(window=plot_config.window_size_mov_average).mean()
                else:
                    # Calculate the second derivative directly from the first derivative
                    second_derivative_vd = first_derivative_vd.diff()

                if idx==0: yaxis='y6' 
                else: yaxis='y6'
                # Add trace for the moving average of the first derivative
                trace_manager.add_trace(
                    name=f'Avg 1D {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {age_band_extension}',
                    x=dataframes_vd[i].iloc[:, 0],
                    y=smoothed_first_derivative_vd,
                    mode='lines',
                    line=dict(dash='solid', width=1, color=shades_2[2]),                    
                    secondary=True,
                    axis_assignment=yaxis)  

                if idx==0: yaxis='y6' 
                else: yaxis='y6'
                # Add trace for the moving average of the second derivative
                trace_manager.add_trace(
                    name=f'Avg 2D {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {age_band_extension}',
                    x=dataframes_vd[i].iloc[:, 0],
                    y=second_derivative_vd,
                    mode='lines',
                    line=dict(dash='solid', width=1, color=shades_2[3]),                    
                    secondary=True,
                    axis_assignment=yaxis)
                
                if idx==0: yaxis='y4' 
                else: yaxis='y4'
                # Add cumulative VD data trace on the secondary y-axis
                trace_manager.add_trace(
                    name=f'cum {os.path.splitext(os.path.basename(csv_files_vd[i]))[0][4:]} {age_band_extension}', 
                    x=dataframes_vd[i].iloc[:, 0],  
                    y=cum_dataframes_vd,            
                    line=dict(dash='dot', width=1.5, color=shades_2[4]),
                    secondary=True,
                    axis_assignment=yaxis
                )

            # Use Plotly and Plotly_r color palette with 10 different colors 
            # for the two age groups and plot additional decay and correlation traces
            if idx == 0:    
                   colors = px.colors.qualitative.Plotly                    
            elif idx == 1:  
                   # colors = ['darkblue', 'purple', 'darkorange']    
                   colors = px.colors.qualitative.Plotly_r                   

            # Calculate and plot decay curve for each pair of decay_pair list!
            target_percentage = [None] * len(dataframes_decay_dvd)
            for k in range(0, len(dataframes_decay_dvd)):

                # Ensure both dataframes have the same structure and align by 'DAY'
                assert all(dataframes_decay_vd[k]['DAY'] == dataframes_decay_dvd[k]['DAY']), "Date columns do not match!"                                  

                df = dataframes_decay_vd[k].copy()
                # Calculate the actual cumulative population/doses by subtracting cumulative deaths from population
                if k == 0:  # csv file[0] POP (the whole population)
                    df[age_band] =  dataframes_decay_vd[0][age_band] - dataframes_decay_dvd[0][age_band].cumsum()  # Subtract cumulative deaths
                elif k == 1:  # csv file [1] UVX (the decreasing uvx population per day, not cumulated)
                    df[age_band] = dataframes_decay_vd[0][age_band] - dataframes_decay_dvd[1][age_band].cumsum() + dataframes_decay_vd[1][age_band].cumsum()  # Adjust for UVX                    
                else:  # csv files [i] VX VDA (the increasing population vx or number of all doses vda per day, not cumulated)
                    df[age_band] = dataframes_decay_vd[k][age_band].cumsum() - dataframes_decay_dvd[k][age_band].cumsum()  # Subtract cumulative deaths
                    
                # create new data frame with deaths doses per day (not cumulative)#
                deaths_df = dataframes_decay_dvd[k].copy()
                # Dictionary to save percentage values for legend
                percentage_dict = {}                               
                # Calculate decay time for each age group
                decay_times_df = pd.DataFrame({'DAY': df['DAY']})
                decay_times = []     

                # Calculate the percentage for the target decay time
                max_days = df['DAY'].max()  # get the whole day range
                all_death = deaths_df[age_band].sum()  # get the deaths for age group
                max_population = df[age_band].max()  # get the max population value

                target_percentage[k] = ((all_death / max_population) / max_days) * plot_config.target_decay_time
                percentage_dict[age_band] = target_percentage[0] * 100  # Save the target percentage for legend
            
                # Loop through each day to calculate the decay time        
                for day in range(len(df)):
                    decay_time = None
                    if df[age_band].iloc[day] > 0: # Only proceed if population is greater than 0
                        for future_day in range(day, len(df)):
                            mean_population = df[age_band].iloc[day:future_day+1].mean()
                            deaths  = deaths_df[age_band].iloc[day:future_day+1].sum()
                            if mean_population > 0 and (deaths / mean_population) >= target_percentage[0]:
                                    decay_time = future_day - day  # Calculate decay time
                                    break                           
                    decay_times.append(decay_time if decay_time is not None else 0)       
                decay_times_df[age_band] = decay_times
                

                # Create a separate DataFrame for rolling averages
                rolling_avg_df = pd.DataFrame({'DAY': df['DAY']})
                # Calculate the rolling average for 'Decay Time' with a window of 7 days
                rolling_avg_df['Rol Avg'] = decay_times_df[age_band].rolling(window=plot_config.window_size_mov_average, min_periods=1).mean()
                
                # Add decay time traces to the plot using trace_manager
                trace_manager.add_trace(
                    name=f'Decay {os.path.splitext(os.path.basename(csv_files_decay_dvd[k]))[0][4:]} {age_band_extension} ({percentage_dict[age_band]:.6f}%)',
                    x=decay_times_df['DAY'],  # Use 'DAY' directly without melting
                    y=decay_times_df[age_band],  # Use 'Decay Time' directly
                    line=dict(dash='solid', width=1.5, color=colors[k % len(colors)]),  # Set the line style and color
                    secondary=True,
                    axis_assignment='y13'  # Adjust axis assignment as per your layout
                )

                # Add the rolling averages of decay time to the plot
                trace_manager.add_trace(
                    name=f'Avg Decay {os.path.splitext(os.path.basename(csv_files_decay_dvd[k]))[0][4:]} {age_band_extension} ({percentage_dict[age_band]:.6f}%)',
                    x=rolling_avg_df['DAY'],
                    y=rolling_avg_df['Rol Avg'],  # Fixed column name for rolling average of decay times
                    line=dict(dash='dot', width=2, color=colors[k % len(colors)]),  # Rolling avg line style and color
                    secondary=True,
                    axis_assignment='y13'  # Adjust axis assignment as per your layout
                )

            # Fill empty legend columns to match the number of legend columns
            add_dummy_traces(trace_manager, trace_manager.get_fig().data, plot_config.custom_legend_column)

            # Call the function to calculate and plot the rolling correlation to compare traces of a single age_band
            time_indices_list, start_idx_list = plot_rolling_correlation_and_phase_shift_for_traces(trace_manager, pairs, age_band_extension, plot_config, colors)        
     
            # Adding the red horizontal line at p = 0.05
            trace_manager.add_trace(
                name=f'p = 0.05 significance level {age_band_extension}',
                x=time_indices_list[-1] ,
                y=[0.05] * len(time_indices_list[-1]),  # Constant line at p = 0.05
                line=dict(color='red', width=1, dash='dash'),
                secondary=True,
                axis_assignment='y7'
            )   

            # Set the plot title
            plot_config.update_title(f'COR_START_DAY: {start_idx_list[-1]} AGE: {age_band_compare[pair_nr][0]} vs {age_band_compare[pair_nr][1]}')

            # Assign the plot traces-curves to the y-axis
            plot_layout(trace_manager.get_fig(), px.colors.qualitative.Dark24, plot_config)

            # Fill empty legend columns to match the number of legend columns
            add_dummy_traces(trace_manager, trace_manager.get_fig().data, plot_config.custom_legend_column)

            # Call the function to calculate and plot the rolling correlation to compare traces between the two age_bands
            if idx == 1:
                plot_rolling_correlation_and_phase_shift_for_traces(trace_manager, updated_age_band_pairs,'', plot_config, colors)        

            # Assign the plot traces-curves to the y-axis
            plot_layout(trace_manager.get_fig(), px.colors.qualitative.Dark24, plot_config)

     
        # Save the plot to an HTML file with a custom legend
        # If you want to automatically save the plots in a different directory to prevent them from being overwritten, 
        # you can add the dependent variables here!
        html_file_path = f"{full_plotfile_name} AG_{age_band_compare[pair_nr][0]} vs {age_band_compare[pair_nr][1]}.html"
        pair_nr += 1

        # Write the Plot to an HTML file including a table of the interactive legends
        write_to_html(html_file_path, trace_manager.get_fig(), age_band_pair, plot_config.custom_legend_column)

        # Extract the base filename without the .html extension
        file_name_without_extension, _ = os.path.splitext(html_file_path)

        # Saves the traces to a .csv file
        if plot_config.savetraces_to_csv:
            save_traces_to_csv(trace_manager, file_name_without_extension)
 
def generate_age_band_pairs(age_band_pair, age_band_pairs):
    # Create an empty list to store the updated pairs
    updated_age_band_pairs = []
    
    # Extract the lower bounds of the single age band pair
    age1_lower = age_band_pair[0].split('-')[0]  # Get the lower bound of the first age band
    age2_lower = age_band_pair[1].split('-')[0]  # Get the lower bound of the second age band
    
    # Loop through the age_band_pairs and add the age information
    for pair in age_band_pairs:
        # For each variable, add the corresponding age bounds
        updated_pair = (f'{pair[0]} {age1_lower}', f'{pair[1]} {age2_lower}')
        updated_age_band_pairs.append(updated_pair)

    # Return the updated pairs
    return updated_age_band_pairs


# Calculate and plot rolling correlation, significance, and phase shift for each pair of traces.
def plot_rolling_correlation_and_phase_shift_for_traces(trace_manager, pairs, age_band_extension, plot_config, colors):
    # Returns: time_indices_list (list): List of time indices for each pair.
    # start_idx_list (list): List of the starting indices for each pair.

    # Create y_series from the trace data
    y_series = {}
    z = -1
    for trace in trace_manager.figData.data:  # Loop through all traces in trace_manager
        z += 1 
        if trace.y is not None and len(trace.y) > 0:
            y_data = np.array(trace.y).flatten()
            y_series[trace.name] = xr.DataArray(y_data, dims='time', coords={'time': np.arange(len(y_data))})
            print(f"Trace {z}:{trace.name}:")
        else:
            print(f"Warning {z}:{trace.name}: has no data or invalid data: {len(trace.y)}")

    # Rolling window size from plot_config
    window_size = plot_config.window_size_correl
    
    # Lists to store time indices and start indices for each pair
    time_indices_list = []
    start_idx_list = []

    # Loop through the pairs of trace names
    for n, (name1, name2) in enumerate(pairs):
        try:
            # Construct the trace names including the age band extension
            if age_band_extension == '':
                name1 = f"{name1}"
                name2 = f"{name2}"
            else :
                name1 = f"{name1} {age_band_extension}"
                name2 = f"{name2} {age_band_extension}"
                
            # Check if both traces exist in y_series
            df1 = y_series[name1]
            df2 = y_series[name2]

        except KeyError:
            # Handle the case where traces with the exact names are not found
            matching_name1 = [key for key in y_series.keys() if key.startswith(name1)]
            matching_name2 = [key for key in y_series.keys() if key.startswith(name2)]
            
            if matching_name1 and matching_name2:
                df1 = y_series[matching_name1[0]]
                df2 = y_series[matching_name2[0]]
            else:
                print(f"Could not find matching traces for :{name1}: and :{name2}: - matchingname1 :{matching_name1}: and matchingname2 :{matching_name2}:")
                continue  # Skip this pair if no match found

        # Find the first non-zero index for both series in their raw form
        start_idx1_raw = first_nonzero_index(df1)
        start_idx2_raw = first_nonzero_index(df2)
        start_idx = max(start_idx1_raw, start_idx2_raw)

        # Filter and align the data, excluding NaNs and zeros
        filtered_df1, filtered_df2 = filter_and_align(df1, df2)

        if len(filtered_df1) == 0 or len(filtered_df2) == 0:
            print(f"Filtered data for {name1} and {name2} is empty. Skipping this pair.")
            continue

        # Calculate rolling correlation and significance
        rolling_corr, p_values = rolling_significance_test(filtered_df1, filtered_df2, window_size)

        # Synchronize the rolling correlation with the data series
        rolling_corr[:0] = np.nan  

        # Time indices for plotting
        time_indices = np.arange(0, len(filtered_df1))

        # Append the time indices and start index to the lists
        time_indices_list.append(time_indices)
        start_idx_list.append(start_idx)

        # Plot rolling correlation
        trace_manager.add_trace(
            name=f'Rolling Corr<br>{name1}<br>{name2}',
            x=time_indices,
            y=rolling_corr[0:],
            line=dict(dash='solid', width=1.5, color=colors[n % len(colors)]),
            secondary=True,
            axis_assignment='y7'
        )

        # Plot significant correlation (p < 0.05)
        trace_manager.add_trace(
            name=f'Sig Corr<br>{name1}<br>{name2} (p<0.05)',
            x=time_indices,
            y=(p_values[0:] < 0.05).astype(int),
            line=dict(dash='dash', width=1, color=colors[n % len(colors)]),
            secondary=True,
            axis_assignment='y7'
        )

        # Plot p-values
        trace_manager.add_trace(
            name=f'P-Values<br>{name1}<br>{name2}',
            x=time_indices,
            y=p_values[0:],  
            mode='lines+markers',
            marker=dict(size=3, color='gray'),
            line=dict(dash='dot', width=1, color='gray'),
            text=p_values[0:],
            hoverinfo='text',
            secondary=True,
            axis_assignment='y7'
        )

        # Calculate and plot phase shift correlation
        max_phase_shift = plot_config.max_phase_shift
        phase_corr = phase_shift_correlation(filtered_df2, filtered_df1, max_phase_shift)

        trace_manager.add_trace(
            name=f'Ph Shift Corr<br>{name1}<br>{name2}',
            x=np.arange(-max_phase_shift, max_phase_shift + 1),
            y=phase_corr,
            line=dict(dash='solid', width=2, color=colors[(n+1) % len(colors)]),  # Different color
            secondary=True,
            axis_assignment='y7'
        )

    # Return the lists containing time indices and start indices for each pair
    return time_indices_list, start_idx_list


def phase_shift_correlation(series1, series2, max_shift=50):
    # Calculate the correlation of two time series at different phase shifts.
    correlations = []
    for shift in range(-max_shift, max_shift + 1):
        shifted_series2 = np.roll(series2, shift)
        # Remove NaN or invalid values (if any) after shifting
        valid_idx = ~np.isnan(series1) & ~np.isnan(shifted_series2)
        correlation = np.corrcoef(series1[valid_idx], shifted_series2[valid_idx])[0, 1]
        correlations.append(correlation)
    return np.array(correlations)

def rolling_significance_test(series1, series2, window_size):
    # Mask out NaNs and zeros in series1 and series2
    valid_series1 = series1.where((series1 != 0) & (~np.isnan(series1)))
    valid_series2 = series2.where((series2 != 0) & (~np.isnan(series2)))

    rolling_corr = []
    p_values = []

    # Calculate rolling correlation only for valid windows (non-NaN and non-zero)
    for i in range(len(valid_series1) - window_size + 1):
        window1 = valid_series1[i:i + window_size]
        window2 = valid_series2[i:i + window_size]

        # Skip windows with NaN or zeros in either series
        if window1.isnull().any() or window2.isnull().any() or (window1 == 0).all() or (window2 == 0).all():
            rolling_corr.append(np.nan)
            p_values.append(np.nan)
            continue

        # Calculate correlation and p-value for the valid window
        corr, p_value = stats.pearsonr(window1.values, window2.values)
        rolling_corr.append(corr)
        p_values.append(p_value)

    return np.array(rolling_corr), np.array(p_values)


# Function for filtering and aligning series (removing NaNs and zeros)
def filter_and_align(series1, series2):
    # Mask out leading NaNs and zeros in both series
    non_zero_1 = series1.where((series1 != 0) & (~np.isnan(series1)))
    non_zero_2 = series2.where((series2 != 0) & (~np.isnan(series2)))

    # Align series by truncating them to the same length
    min_length = min(len(non_zero_1), len(non_zero_2))
    return non_zero_1[:min_length], non_zero_2[:min_length]
    
# Helper function for identifying the first valid (non-zero, non-NaN) index in a series
def first_nonzero_index(series, threshold=1e-6):
    # Mask out NaNs and values close to zero (below the threshold)
    valid_series = series.where((series != 0) & (~np.isnan(series)) & (np.abs(series) > threshold))
    
    # Find the first valid (non-zero, non-NaN) index
    non_zero_indices = np.nonzero(~np.isnan(valid_series.values))[0]
    
    if len(non_zero_indices) > 0:
        return non_zero_indices[0]
    else:
        return 0  # Return 0 if no valid data exists
    

# Update plot layout for dual y-axes - traces were assignd to yaxis by tracemanager       
def plot_layout(figData, color_palette, plot_config):
    
    # Update the layout of the figure
    figData.update_layout(
        colorway=color_palette,
        title=dict(
            text=plot_config.title_text,
            y=0.98,
            font=dict(family='Arial', size=18),
            x=0.03,  # Move title to the left (0.0 = far left)
            xanchor='left',  # Anchor title to the left side
            yanchor='top'
        ),
        annotations=[  # Add annotation manually
            dict(
                text=plot_config.annotation_text,
                x=0.28,  # Position annotation to the left of the plot
                y=0.99, # Adjust the y-position slightly lower than the main title
                xanchor='left',  # Anchor the annotation to the left
                yanchor='top',        
                font=dict(family='Arial',size=14, color='grey'),  # annotation style
                showarrow=False,  # No arrow
                align='left',   # This ensures the text itself is left-aligned
                xref='paper',  # This ensures 'x' is relative to the plot area (paper means the entire canvas)
                yref='paper'  # This ensures 'y' is relative to the plot area
            )
        ],

        # For different scale set autorange=True for yaxis2-7 and for yaxis8-13
        # For same scale remove autorange=True for yaxis2-7, yaxis8-13 not used then
        # 1st Age_Band (idx)
        yaxis=dict(title=plot_config.yaxis1_title, type=plot_config.yaxis_type, side='left'), # yaxis_type log/linear from the instance of PlotConfig
        yaxis2=dict(title='Values y2 VD', anchor='free', position=0.05, side='left', autorange=True),
        yaxis3=dict(title=plot_config.yaxis3_title, overlaying="y", position=0.9, side="right", autorange=True),
        yaxis4=dict(title='Cumulative Values y4 VD', overlaying="y", side="right", autorange=True),
        yaxis5=dict(title='1st and 2nd Derivative y5 DVD', overlaying="y", side="left", position=0.15, autorange=True),  # 1st and 2nd derivative y5
        yaxis6=dict(title='1st and 2nd Derivative y6 VD', overlaying="y", side="left", position=0.25, autorange=True),   # 1st and 2nd derivative y6
        yaxis7=dict(title='Rolling Pearson Correlation y7', overlaying='y', side='right', position=0.8, autorange=True), # Rolling Pearson y7        
        # 2nd Age_Band (idx) - used only if differnt sacle for age_bands needed
        yaxis8=dict(title='',overlaying="y", type=plot_config.yaxis_type, side='left', autorange=True),   # title=f'{plo["yaxis1_title"]} y1'             
        yaxis9=dict(title='',overlaying="y", anchor='free', position=0.05, side='left', autorange=True), # title='Values y2 VD'
        yaxis10=dict(title='', overlaying="y", position=0.9, side="right", autorange=True), # title='Cumulative Values y3 DVD'
        yaxis11=dict(title='', overlaying="y", side="right",autorange=True), # title='Cumulative Values y4 VD'
        yaxis12=dict(title='', overlaying="y", side="left", position=0.15, autorange=True),  # 1st and 2nd Derivative y5 DVD'
        yaxis13=dict(title='', overlaying="y", side="left", position=0.25, autorange=True),  # 1st and 2nd Derivative y6 DVD'
        # Rolling Pearson Correlation has the same scale -1 to 1 so no additional yaxis14 needed 

        legend=dict(
            orientation="v",
            xanchor="left",
            x=1.05,
            yanchor="top",
            y=1,
            font=dict(size=10)
        ),
        margin=dict(l=40, r=50, t=40, b=40)
    )

    
# Save plotly plot to a html file with interactive custom legends
def write_to_html(html_file_path, figData, age_band, num_columns):
    # Prepare custom legend items based on the figure traces
    legend_items = []
    for trace in figData['data']:
        legend_items.append({
            'name': trace['name'] if 'name' in trace else 'Unnamed Trace',
            'color': trace['line']['color'] if 'line' in trace and 'color' in trace['line'] else '#000000'  # Default color if not set
        })

    # Set the desired height and width (in pixels)
    desired_height = 800  # Adjust this value as needed
    desired_width = 2096  # Adjust this value as needed

    # Create the complete HTML
    with open(html_file_path, 'w') as f:
        f.write('<!DOCTYPE html>\n<html lang="de">\n<head>\n')
        f.write('    <meta charset="UTF-8">\n')
        f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        f.write('    <title>Plotly Diagramm mit großer Tabellenlegende</title>\n')
        f.write('    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n')
        f.write('    <style>\n')

        # Ensure the font is Arial (or fallback to sans-serif)
        f.write('        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }\n')

        # Dynamically set the number of columns based on the num_columns parameter
        f.write(f'        .legend {{ display: grid; grid-template-columns: repeat({num_columns}, 1fr); gap: 10px; margin-top: 20px; }}\n')
        f.write('        .legend-item { display: flex; align-items: center; cursor: pointer; font-size: 12px; }\n')
        f.write('        .legend-color-box { width: 10px; height: 10px; margin-right: 5px; }\n')
        f.write('    </style>\n')
        f.write('</head>\n<body>\n')

        # Plot area size
        f.write(f'    <div id="plotly-figure" style="width: {desired_width}px; height: {desired_height}px;"></div>\n')
        f.write('    <div id="legend" class="legend"></div>\n')  # Add the 'legend' class here
        f.write('    <script>\n')

        # Insert the Plotly figure data
        fig_json = figData.to_json()  # Get the JSON string
        fig_data = json.loads(fig_json)  # Convert it to a dictionary
        f.write('    var data = ' + json.dumps(fig_data['data']) + ';\n')  # Access and write the data

        # Enable default legend
        layout = fig_data['layout']
        #layout['showlegend'] = true  # Ensure the default legend is visible
        f.write('    var layout = ' + json.dumps(layout) + ';\n')  # Use json.dumps for layout
        f.write('    Plotly.newPlot("plotly-figure", data, layout);\n')

        # Add custom legend items to the script
        f.write('    var legendItems = ' + json.dumps(legend_items) + ';\n')
        f.write('    var legendDiv = document.getElementById("legend");\n')

        # Track the state of all traces
        f.write('    var allVisible = true;\n')  # State to track if all traces are visible

        # Iterate over legend items and create HTML elements
        f.write('    legendItems.forEach(function(item, index) {\n')
        f.write('        var legendItem = document.createElement("div");\n')
        f.write('        legendItem.className = "legend-item";\n')
        
        # Set initial visibility based on data
        f.write('        var traceVisible = data[index].visible !== false;\n')
        f.write('        legendItem.innerHTML = `<div class="legend-color-box" style="background-color: ${item.color}; opacity: ${traceVisible ? 1 : 0.5};"></div>${item.name}`;\n')
        f.write('        legendItem.style.color = traceVisible ? "black" : "gray";\n')

        # Add click event listener for individual trace toggle
        f.write('        legendItem.onclick = function() {\n')
        f.write('            var currentVisibility = data[index].visible;\n')
        f.write('            // Toggle the visibility\n')
        f.write('            data[index].visible = (currentVisibility === true || currentVisibility === "true") ? false : true;\n')

        # Update the legend item appearance based on visibility
        f.write('            if (data[index].visible) {\n')
        f.write('                legendItem.querySelector(".legend-color-box").style.opacity = "1";\n')
        f.write('                legendItem.style.color = "black";\n')
        f.write('            } else {\n')
        f.write('                legendItem.querySelector(".legend-color-box").style.opacity = "0.5";\n')
        f.write('                legendItem.style.color = "gray";\n')
        f.write('            }\n')

        # Use Plotly.react for a more efficient update
        f.write('            Plotly.react("plotly-figure", data, layout);\n')
        f.write('        };\n')

        # Add double-click event listener for select/deselect all
        f.write('        legendItem.ondblclick = function() {\n')
        f.write('            allVisible = !allVisible;\n')  # Toggle the visibility state
        f.write('            data.forEach(function(trace) {\n')
        f.write('                trace.visible = allVisible;  // Set all traces to the same visibility state\n')
        f.write('            });\n')
        f.write('            Plotly.update("plotly-figure", data, layout);  // Update the plot\n')

        # Update all legend items based on the visibility state
        f.write('            var newOpacity = allVisible ? 1 : 0.5;\n')
        f.write('            var newColor = allVisible ? "black" : "gray";\n')
        f.write('            var legendItems = document.querySelectorAll(".legend-item");\n')
        f.write('            legendItems.forEach(function(item) {\n')
        f.write('                item.querySelector(".legend-color-box").style.opacity = newOpacity;\n')
        f.write('                item.style.color = newColor;\n')
        f.write('            });\n')  # Update all legend items
        f.write('        };\n')

        # Append the legend item to the legend div
        f.write('        legendDiv.appendChild(legendItem);\n')
        f.write('    });\n')
        f.write('</script>\n')
        f.write('</body>\n</html>\n')

    print(f"Plot {html_file_path} {age_band} has been saved to HTML file.")


# Class to mange yaxis assignment for the traces
class TraceManager:
    def __init__(self):
        self.figData = make_subplots(specs=[[{"secondary_y": True}]])  # Initialize the figure with subplots
        self.axis_list = []  # Stores axis assignments for each trace

    def add_trace(self, name, x, y, line=None, mode=None, marker=None, text=None, hoverinfo=None, axis_assignment='y1', secondary=False):
        """
        Adds a trace to the figure and assigns it to a specific axis.

        Args:
        - axis_assignment: Axis for the trace ('y1', 'y2', etc.).
        - Other args: Same as in Plotly (e.g., trace_name, x_data, y_data, line, mode, text, secondary, hoverinfo).
        """

        # Add trace using Plotly's standard parameters
        self.figData.add_trace(go.Scatter(
            x=x,
            y=y,
            mode=mode,              # Directly use 'mode' (Plotly's parameter)
            marker=marker,          # Directly use 'marker' (Plotly's parameter)   
            line=line,              # Directly use 'line' (Plotly's parameter)
            name=name,              # Trace name
            text=text,              # Hover text
            hoverinfo=hoverinfo     # Hover information
        ), secondary_y=secondary)   # Use 'secondary_y' for secondary axis
        
        # Store the axis assignment
        self.axis_list.append(axis_assignment)

        # Update the trace's axis assignment
        self._update_axis_for_trace(len(self.axis_list) - 1, axis_assignment)

    def _update_axis_for_trace(self, trace_index, axis_assignment):
       
        # Updates the axis for the specific trace in the figure.
       
        assigned_axis = axis_assignment
        trace = self.figData.data[trace_index]

        # Update the trace's axis based on the assignment
        if assigned_axis == 'y1':
            trace.update(yaxis='y1')
        elif assigned_axis == 'y2':
            trace.update(yaxis='y2')
        elif assigned_axis == 'y3':
            trace.update(yaxis='y3')
        elif assigned_axis == 'y4':
            trace.update(yaxis='y4')
        elif assigned_axis == 'y5':
            trace.update(yaxis='y5')
        elif assigned_axis == 'y6':
            trace.update(yaxis='y6')
        elif assigned_axis == 'y7':
            trace.update(yaxis='y7')
        elif assigned_axis == 'y8':
            trace.update(yaxis='y8')
        elif assigned_axis == 'y9':
            trace.update(yaxis='y9')
        elif assigned_axis == 'y10':
            trace.update(yaxis='y10')
        elif assigned_axis == 'y11':
            trace.update(yaxis='y11')
        elif assigned_axis == 'y12':
            trace.update(yaxis='y12')
        elif assigned_axis == 'y13':
            trace.update(yaxis='y13')

    def get_fig(self):
        # Returns the figure object
        return self.figData

    def get_axis_list(self):
        # Returns the list of axis assignments
        return self.axis_list


# Class to handle the plot title, annotation, and axis labels depending on the settings
class PlotConfig:
    def __init__(self, title_text, plot_name_append_text, window_size_correl,max_phase_shift,window_size_mov_average,target_decay_time,
                 normalize=False, normalize_cumulate_deaths=False, 
                 population_minus_death=False, custom_legend_column=6, axis_y1_logrithmic=False, 
                 savetraces_to_csv=False, calc_derivate_mean=False, pairs=None):
        # Initialize attributes based on input arguments
        self.initial_title = title_text
        self.title_text = title_text
        self.plot_name_append_text = plot_name_append_text
        self.window_size_correl = window_size_correl
        self.max_phase_shift = max_phase_shift
        self.window_size_mov_average = window_size_mov_average
        self.target_decay_time = target_decay_time
        self.normalize = normalize
        self.normalize_cumulate_deaths = normalize_cumulate_deaths
        self.population_minus_death = population_minus_death
        self.custom_legend_column = custom_legend_column
        self.axis_y1_logrithmic = axis_y1_logrithmic
        self.savetraces_to_csv = savetraces_to_csv
        self.calc_derivate_mean = calc_derivate_mean
        
        # Initialize pairs (can be updated later)
        self.pairs = pairs if pairs is not None else []
        
        # Recalculate the related data based on initial pairs
        self.update_plot_data()
        

    def __str__(self):
        return (f"PlotConfig(title_text='{self.title_text}', "
                f"plot_name_append_text={self.plot_name_append_text}, "
                f"window_size_correl={self.window_size_correl}, "
                f"max_phase_shift={self.max_phase_shift}, "
                f"window_size_mov_average={self.window_size_mov_average}, "
                f"target_decay_time={self.target_decay_time}, "                
                f"normalize={self.normalize}, "
                f"normalize_cumulate_deaths={self.normalize_cumulate_deaths}, "
                f"population_minus_death={self.population_minus_death}, "
                f"custom_legend_column={self.custom_legend_column}, "
                f"savetraces_to_csv={self.savetraces_to_csv}, "
                f"axis_y1_logrithmic='{self.axis_y1_logrithmic}', "
                f"calc_derivate_mean={self.calc_derivate_mean})")

    def update_title(self, new_title):
        self.title_text = f'{self.initial_title} {new_title}'

    @property
    def yaxis_type(self):
        return 'log' if self.axis_y1_logrithmic else 'linear'

    def update_pairs(self, pairs):
        # Update the pairs and recalculate the dependent values like pairs_text and annotation_text.
        self.pairs = pairs
        self.update_plot_data()

    def update_plot_data(self):
        # Recalculate values like pairs_text, annotation_text, and axis titles.
        # This method is called when the pairs are updated or the object is initialized.
        plot_name = self._generate_plot_name()
        pairs_text = self._generate_pairs_text(self.pairs)
        annotation_text = self._generate_annotation_text(pairs_text)

        # Normalize axis titles
        yaxis1_title, yaxis3_title = self._generate_axis_titles()

        # Update attributes
        self.pairs_text = pairs_text
        self.annotation_text = annotation_text
        self.plot_name = plot_name
        self.yaxis1_title = yaxis1_title
        self.yaxis3_title = yaxis3_title

    def _generate_plot_name(self):
        # createa a plot name depending on config settings and custom extension
        plot_name = f"AVG_{self.window_size_mov_average} CORR_{self.window_size_correl}"
        if self.normalize:
            plot_name = f"N {plot_name}"
        if self.normalize_cumulate_deaths:
            plot_name = f"N-CUM-D {plot_name}"
        if self.population_minus_death:
            plot_name = f"POP-D {plot_name}"
        if self.calc_derivate_mean:
            plot_name = f"1D2D-MEAN {plot_name}"
        if self.plot_name_append_text != '':
            plot_name = f"{plot_name} {self.plot_name_append_text}"

        self.initial_title += f' {plot_name}'                 
        return plot_name

    def _generate_pairs_text(self, pairs):
        pairs_text = ""
        for pair in pairs:
            pairs_text += f"{pair[0].replace(' ', '').replace('Avg', 'A')}-{pair[1].replace(' ', '').replace('Avg', 'A')} "
        return pairs_text.strip()

    def _generate_annotation_text(self, pairs_text):
        annotation_text = ""
        if pairs_text != "":
            # Wrap the text at 50 characters and join with <br> tags
            wrapped_text = "<br>".join(textwrap.wrap(pairs_text, 100, break_long_words=False, replace_whitespace=False))
            annotation_text += wrapped_text + "<br>"  # Adding final <br> at the end
            print(f'annotation: {annotation_text}')
            
        if self.normalize:
            if self.axis_y1_logrithmic:
                annotation_text += "log Values were normalized per 100000<br>"
            else:
                annotation_text += "Values were normalized per 100000<br>"

        if self.normalize_cumulate_deaths:
            annotation_text += "The cumulative deaths were first normalized and then cumulated.<br>"

        if self.population_minus_death:
            annotation_text += "Deaths were subtracted from population<br>"
        else:
            annotation_text += "Deaths not! subtracted from population<br>"

        annotation_text += "To deselect all - double-click on a legend entry"

        return annotation_text

    def _generate_axis_titles(self):
        if self.normalize:
            if self.axis_y1_logrithmic:
                yaxis1_title = 'log Values per 100000 y1 DVD'
            else:
                yaxis1_title = 'Values per 100000 y1 DVD'
        else:
            yaxis1_title = 'Values y1 DVD'

        if self.normalize_cumulate_deaths:
            yaxis3_title = "Values per 100000 Cumulative y3 DVD"
        else:
            yaxis3_title = "Values Cumulative y3 DVD"

        return yaxis1_title, yaxis3_title


def add_dummy_traces(trace_manager, actual_traces, num_columns=10):
    """
    Fill empty html custom legend columns to match the number of legend column
    Args: 
        - trace_manager (TraceManager): The TraceManager object to which dummy traces will be added.
        - actual_traces (list): The list of actual traces already added to the figure.
        - num_columns (int): The number of columns to fill in the custom legend. Default is 10.
    """

    # Calculate the number of actual traces
    num_actual_traces = len(actual_traces)

    # Calculate how many rows and dummy traces are needed
    num_traces_in_last_row = num_actual_traces % num_columns
    if num_traces_in_last_row == 0:
        num_traces_in_last_row = num_columns  # If no remainder, fill last row with full traces

    # Calculate how many dummy traces are needed to complete the last row
    num_dummy_traces = num_columns - num_traces_in_last_row

    # Add dummy traces to fill the last row, if necessary
    for _ in range(num_dummy_traces):
        # Dummy trace is just a placeholder, no real data is plotted
        trace_manager.add_trace(
            name="-",  
            x=[None],  
            y=[None],  
            mode='markers',  # No markers, invisible trace
            text=None,  
            hoverinfo='none',  
            axis_assignment='y1'  
    )


# Function for creating shades (reusable for each color palette)
def generate_shades(base_color, num_shades=5, lightness_factor=0.1):
    shades = []
    # Convert hex color to RGB
    if base_color.startswith("#"):            
        base_color = mcolors.to_rgb(base_color)

    # Convert RGB to HSV (hue, saturation, brightness)
    hsv = colorsys.rgb_to_hsv(base_color[0], base_color[1], base_color[2])

    # Create shades by varying the brightness
    for i in range(num_shades):
        new_value = min(1.0, max(0.4, hsv[2] + lightness_factor * (i - 2)))  # Adjust brightness
        new_rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], new_value)  # Keep hue and saturation constant
        new_hex = mcolors.rgb2hex(new_rgb)  # Convert back to Hex
        shades.append(new_hex)

    return shades

# Function for creating color pairs and shades
def generate_color_shades(color_palette, n_pairs=11):
    color_shades = {}
    for i in range(n_pairs):
        # Select color pairs from the palette
        base_color_dvd_1 = color_palette[i % len(color_palette)]
        base_color_vd_1 = color_palette[(i + 1) % len(color_palette)]

        # Calculate shading for the DVD and VD
        shades_dvd_1 = generate_shades(base_color_dvd_1)
        shades_vd_1 = generate_shades(base_color_vd_1)

        # Save the shades
        color_shades[i] = (shades_dvd_1, shades_vd_1)

    return color_shades


# Saves the trace data (x, y) from all traces in the figure managed by a TraceManager to a csv file 
def save_traces_to_csv(trace_manager, filename='trace_data.csv'):
    """
    The first column of CSV file is 'Day' and subsequent columns are the trace names.
    
    Args:
    - trace_manager: An instance of TraceManager.
    - filename: The name of the CSV file to save the data.
    """
    # Get the filename without the extension
    file_name_without_extension, file_extension = os.path.splitext(filename)

    # Ensure the filename has a .csv extension if not already present
    if not file_extension:
        filename = f'{file_name_without_extension}.csv'

    # Initialize a list to store all rows (days and y-values)
    all_data = []
    
    # Get the x-values (assumed to be the same for all traces)
    x_data = trace_manager.get_fig().data[0].x  # Assuming x-values are the same for all traces

    # Collect the y-values for each trace
    trace_names = []  # To store the names of the traces (for headers)
    for trace in trace_manager.get_fig().data:
        trace_name = trace.name if trace.name else 'Unnamed'
        trace_names.append(trace_name)
    
    # Create the header: 'Day', followed by trace names
    header = ['Day'] + trace_names
    all_data.append(header)

    # Now, collect the y-values for each day and each trace
    for i, x in enumerate(x_data):
        row = [x]  # Start with the day value (x)
        
        for trace in trace_manager.get_fig().data:
            if len(trace.y) > i:  # Ensure trace.y has at least i+1 elements
                row.append(trace.y[i])  # Append the corresponding y-value for the trace
            else:
                # If trace.y is empty or doesn't have enough values, append None or a placeholder
                row.append(None)

        all_data.append(row)
    
    # Write the collected data to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_data)
    
    print(f"Trace data saved to {filename}")


# function to assign the htmlplot name, copy this py script to the plot directory 
def init_function(plt_name):
    # Strip any trailing spaces or slashes from plt_name to avoid malformed paths
    plt_name = plt_name.strip().rstrip(os.sep)  # Remove trailing slashes and spaces   
    
    # Get the script name (without extension)
    script_name = os.path.splitext(os.path.basename(__file__))[0]    
    
    # Get the absolute path of the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    
    # Go up two directories (to the root project directory)
    first_root_dir = os.path.abspath(os.sep.join(script_dir.split(os.sep)[:2]))    
    
    # Construct the plot result path correctly (with Plot Results folder)
    plot_result_path = os.path.join(first_root_dir, "Plot Results", f"{script_name} {plt_name}")    
    
    # Debugging: Print out the full path being created
    print(f"Plot result path: {plot_result_path}")    
    
    # Create the directory if it doesn't exist
    try:
        os.makedirs(plot_result_path, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory: {e}")
        raise    
    
    # Construct the full destination path for the script (ensuring no double slashes)
    script_file_name = os.path.join(plot_result_path, f"{script_name}.py")    
    # debug print(f"Script file will be copied to: {script_file_name}")    
    # Copy the current script to the new location
    try:
        shutil.copy(__file__, script_file_name)  # Copy the script to the new directory
    except Exception as e:
        print(f"Error copying file: {e}")
        raise    
    # Return the full path to the plot result
    return os.path.join(plot_result_path, f"{script_name} {plt_name}")



# call main script
if __name__ == "__main__": main()