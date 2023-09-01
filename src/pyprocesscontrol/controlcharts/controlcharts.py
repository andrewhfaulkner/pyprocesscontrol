# This script is for control charts

from datetime import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.text import OffsetFrom
import numpy as np
import pandas as pd
import scipy.stats as st


def control_chart_builder(
    data: pd.DataFrame,
    index: str = None,
    datetime_index: bool = False,
    include_cols: list = None,
    exclude_cols: list = None,
    grouped: bool = False
) -> list:
    """This function produces a control chart of the data that it is given.
    Args:
        data: DataFrame containing the data to be charted
        index: The index for the dataset if it needs to be reindexed
        datetime_index: Indicates if the index is a datetime
        include_cols: columns to include from the data set
        exclude_cols: columns to exclude from the data set
        grouped: Determines of any the data is grouped logically 
    """

    fig_list = []

    #reset the index to the 'index' term supplied.
    if index != None:
        if data.index.name == index:
            pass
        else:
            try:
                data = data.set_index(index)
            except:
                pass

    if datetime_index:
        data.index = pd.to_datetime(data.index)    

    # Get the specs from the spec sheet
    df = pd.DataFrame()
    significant_specs = pd.DataFrame(columns=["name"])
    col_names = data.columns

    #If specific columns are needed this block selects them
    if exclude_cols != None:
        col_names = col_names.drop(exclude_cols)

    if include_cols != None:
        col_names = include_cols

    j = 0
    for column in data.columns:
        num = 1
        all_columns = list(data.columns)
        if j == 0:
            lst = [column]
            j = 1
            continue

        while column in lst:
            column = column + "_" + str(num)

            if column in lst:
                num += 1
        
        lst.append(column)

    data.columns = lst
    col_names = data.columns

    # Clean up the data in the data set to only process columns with data
    count = data[col_names].count()

    drop_cols = count[count == 0]
    data = data.drop(drop_cols.index, axis= 1)
    

    for column in data.columns:
        series = pd.to_numeric(data[column], errors="coerce")
        data[column] = series
        if data[column].isnull().all():
            data = data.drop(column, axis = 1)
    
    if data.isnull().values.all() == True:
        return

    col_names = data.columns
    data = data.sort_index(axis = 0, ascending= True)
    stats = data.describe()

    mosaic = """
            AAB
            CCB
            """

    col_names = np.array(col_names)
    num_figs = math.ceil(len(col_names) / 2)

    fig = []
    i = 0
    fig_num = 0

    # Plotting all the major statistics for the bleach truck samples
    for column in col_names:

        if len(data[column]) / 30 >= 2 and len(data[column]) / 30 < 10 and grouped == True:
            window_size = math.floor(len(data[column]) / 30)

            temp_df = (
                data[column]
                .rolling(
                    window=window_size,
                    step=window_size,
                    min_periods=math.ceil(window_size / 2),
                )
                .mean()
            )

            rng_min = (
                data[column]
                .rolling(
                    window=window_size,
                    step=window_size,
                    min_periods=math.ceil(window_size / 2),
                )
                .min()
            )

            rng_max = (
                data[column]
                .rolling(
                    window=window_size,
                    step=window_size,
                    min_periods=math.ceil(window_size / 2),
                )
                .max()
            )

            df = temp_df 
            rng = rng_max - rng_min
            upper_control = df.mean() + 2.659 * rng.mean()
            lower_control = df.mean() - 2.659 * rng.mean()
            r_upper_control = rng.mean() * 3.267
            r_lower_control = rng.mean() * 0
            r_bar = rng.mean()
            title_str = "X-bar and R for "
        elif len(data[column]) / 30 >= 10 and grouped == True:
            window_size = 10
            temp_df = (
                data[column]
                .rolling(
                    window=window_size,
                    step=window_size,
                    min_periods=math.ceil(window_size / 2),
                )
                .mean()
            )

            std_min = (
                data[column]
                .rolling(
                    window=window_size,
                    step=window_size,
                    min_periods=math.ceil(window_size / 2),
                )
                .std()
            )

            std_max = (
                data[column]
                .rolling(
                    window=window_size,
                    step=window_size,
                    min_periods=math.ceil(window_size / 2),
                )
                .std()
            )

            df = temp_df
            rng = std_min
            upper_control = df.mean() + 0.975 * std_min.mean()
            lower_control = df.mean() - 0.975 * std_min.mean()
            r_upper_control = std_min.mean() * 1.716
            r_lower_control = std_min.mean() * 0.284
            r_bar = std_min.mean()
            title_str = "X-bar and S for "
        elif len(data[column]) / 30 >= 1 and grouped == True:
            window_size = 1
            df = data[column]
            dropped = df[:-1]
            shifted = df[1:]
            rng = abs(shifted - dropped.values)
            upper_control = df.mean() + 2.659 * rng.mean()
            lower_control = df.mean() - 2.659 * rng.mean()
            r_upper_control = rng.mean() * 3.267
            r_lower_control = rng.mean() * 3.267
            r_bar = rng.mean()
            title_str = "I and MR for "
        else:
            window_size = 1
            df = data[column]
            dropped = df[:-1]
            shifted = df[1:]
            rng = abs(dropped - shifted.values)
            upper_control = df.mean() + 3 * df.std()
            lower_control = df.mean() - 3 * df.std()
            r_upper_control = rng.mean() * 3.267
            r_lower_control = rng.mean() * 3.267
            r_bar = rng.mean()
            title_str = "I and MR for "


        fig = plt.figure(fig_num, figsize=(12, 6.5))
        subfig = fig.subfigures(nrows=1, ncols=1)
        temp_fig = subfig.subplot_mosaic(mosaic)
        fig.subplots_adjust(bottom=0.15)

        # Change the statistic based on the number of samples
        if window_size == 1:
            temp_fig["C"].set_ylabel("Moving Range")
            date_index = np.array(df.index)
        elif window_size >= 2:
            date_index = np.array(df.index)
            temp_fig["C"].set_ylabel("Range")


        temp_fig["A"].plot(df.index, df, marker="o", zorder=2)
        #temp_fig["A"].scatter(ooc.index, ooc, color="red", zorder=3)
        temp_fig["A"].axhline(
            y=stats[column].loc["mean"], color="r", linestyle="--", label="AVG"
        )

        temp_fig["A"].axhline(
            y=upper_control, color="black", linestyle="--", label="UCL"
        )

        temp_fig["A"].axhline(
            y=lower_control, color="black", linestyle="--", label="LCL"
        )

        temp_fig['A'].axhspan(stats[column].loc['mean']-3*stats[column].loc['std'],
                           stats[column].loc['mean']+3*stats[column].loc['std'],
                           facecolor = "#cccccc",
                           zorder = -1)
        
        temp_fig['A'].axhspan(stats[column].loc['mean']-2*stats[column].loc['std'],
                           stats[column].loc['mean']+2*stats[column].loc['std'],
                           facecolor = "#e6e6e6",
                           zorder = 0)
        
        temp_fig['A'].axhspan(stats[column].loc['mean']-1*stats[column].loc['std'],
                           stats[column].loc['mean']+1*stats[column].loc['std'],
                           facecolor = "#f2f2f2",
                           zorder = 1)
        
        temp_fig["A"].set_title(title_str + column)
        temp_fig["A"].set_ylabel(column, fontsize=10, wrap=True)
        temp_fig["A"].set_xlabel("Date")
        temp_fig["A"].get_xaxis().set_visible(False)
        ooc_checker(df, temp_fig["A"], column= column)

        temp_fig["C"].sharex(temp_fig["A"])
        temp_fig["C"].plot(rng.index, rng, marker="o")
        temp_fig["C"].axhline(y=r_bar, color="red", linestyle="--", label="AVG")
        temp_fig["C"].axhline(
            y=r_upper_control, color="black", linestyle="--", label="UCL"
        )

        #temp_fig["A"].set_xticklabels(temp_fig["A"].get_xticks(), rotation = 30)
        temp_fig["A"].xaxis.set_major_locator(mdates.DayLocator(interval = 5))
        temp_fig["A"].xaxis.set_minor_locator(mdates.DayLocator(interval = 1))
        temp_fig["A"].xaxis.set_major_formatter(mdates.ConciseDateFormatter(temp_fig["A"].xaxis.get_minor_locator()))
        

        #if r_lower_control.sum() != 0:
        temp_fig["C"].axhline(
            y=r_lower_control, color="black", linestyle="--", label="LCL"
            )

        histogram_builder(df, temp_fig["B"], col_name= column)
        i = i + 1

        if i == 1:
            i = 0
            fig_num = fig_num + 1
            plt.subplots_adjust(hspace=0)

        fig_list.append(fig)
        #plt.show()
    
    return fig_list


def histogram_builder(
    data: pd.DataFrame,
    ax,
    include_cols: list = None,
    exclude_cols: list = None,
    col_name: str = None,
) -> None:
    """This function builds a histogram of the plotted data and displays it

    Args:
        data: Dataframe containing the data to plot
        ax: Matplotlib axis to plot the histogram to
    Returns:
        None
    """

    # grabs the data and the statistics for the group and counter
    # stats = data_dict[(group, counter)][0]

    # extracts the specs for the group and counter
    # specs = specs
    # lower_spec = specs.specs[group][counter]['Spec'][sig_spec]['LSL']
    # upper_spec = specs.specs[group][counter]['Spec'][sig_spec]['USL']
    # range = upper_spec - lower_spec

    # drops NA's from the dataframe
    # data = data.dropna(axis = 1)

    if isinstance(data, pd.Series) == True:
        data = pd.DataFrame(data)    

    # get the column name for the sig_spec
    if exclude_cols != None:
        col_names = col_names.drop(exclude_cols)
    elif include_cols != None:
        col_names = include_cols
    elif col_name != None:
        col_names = col_name
    
    else:
        col_names = data.columns

    stats = data.describe()

    # gets the min and max confidence interval values from the statistics
    min = stats[col_names].loc["mean"] - 3.5 * stats[col_names].loc["std"]
    max = stats[col_names].loc["mean"] + 3.5 * stats[col_names].loc["std"]
    avg = stats[col_names].loc["mean"]

    # gets the min and max data values from the statistics
    data_min = stats[col_names].loc["min"]
    data_max = stats[col_names].loc["max"]

    # develops the x axis for the normal distribution plot
    x_axis = np.linspace(min, max, num=100)

    count = stats[col_names].loc["count"]

    # Creates and plots the histogram figure
    counts, bins = np.histogram(data[col_names].dropna())
    bin_width = (data_max - data_min) / len(bins)
    ax.hist(x=data[col_names], bins=bins)
    ax.plot(
        x_axis,
        st.norm.pdf(x_axis, stats[col_names].loc["mean"], stats[col_names].loc["std"])
        * count
        * bin_width,
        "k",
    )

    # ax.axvline(lower_spec, color = 'r')
    # ax.axvline(upper_spec, color = 'r')
    ax.axvline(avg, color="#82F85F")

    ax.set_xlabel(col_names)
    ax.set_ylabel("Count")

    # if upper_spec > max:
    # upper_rng = upper_spec
    # else:
    # upper_rng = max

    # if lower_spec < min:
    # lower_rng = lower_spec
    # else:
    # lower_rng = min

    # ax.set_xlim((lower_rng-range/100, upper_rng+range/100))


def ooc_checker(data: pd.DataFrame| pd.Series, ax, column: str = None) -> None:
    '''This function checks the data to see if the data is potentially out of control
        due to special cause variation. This will use the 8 rules for out of control
        processes.
    '''

    if isinstance(data, pd.Series):
        mean = data.mean()
        std = data.std()
    elif isinstance(data, pd.DataFrame):
        stats = data.describe()
        mean = stats[column].loc["mean"]
        std = stats[column].loc["std"]
    
    centerline_distance = data - mean

    #Rule 1: 1 point 3-sigma from the centerline
    rule1 = data[~data.between(mean - 3*std, mean + 3*std)]

    #Rule 2: 9 points in a row on the same side of the centerline
    temp_df2 = data - mean
    temp_df2[temp_df2 > 0] = 1
    temp_df2[temp_df2 < 0] = 0

    more_than = temp_df2
    more_than[more_than < 0] = 0
    more_than[more_than > 1] = 1
    more_than_window = more_than.rolling(window= 9, step= 1).sum()

    less_than = temp_df2
    less_than[less_than > 0] = 0
    less_than[less_than < -1*std] = -1
    less_than_window = less_than.rolling(window= 9, step= 1).sum()
    
    index_list = pd.Series(more_than_window[more_than_window >= 9].index)
    index_list = pd.concat([index_list, pd.Series(less_than_window[less_than_window <= -9])])

    rule2 = data.loc[index_list]

    #Rule 3: Six points in a row, all increasing or decreasing

    #Rule 4: 14 points in a row all oscillating
    temp_df4 = data - mean
    dropped = temp_df4[:-1]
    shifted = temp_df4[1:]
    oscillating = dropped - shifted
    oscillating[oscillating > 0] = 1
    oscillating[oscillating < 0] = -1
    windowed_df = oscillating.rolling(window= 14, step= 1).sum()
    rule4 = data.loc[windowed_df[windowed_df >= 14].index]

    #Rule 5: 2/3 points more than 2-sigma from the centerline (same side)
    temp_df5 = data - mean

    more_than = data - mean
    more_than[more_than < 2*std] = 0
    more_than[more_than > 2*std] = 1
    more_than_window = more_than.rolling(window= 3, step= 1).sum()

    less_than = data - mean
    less_than[less_than > -2*std] = 0
    less_than[less_than < -2*std] = -1
    less_than_window = less_than.rolling(window= 3, step= 1).sum()
    
    index_list = pd.Series(more_than_window[more_than_window >= 2].index)
    index_list = pd.concat([index_list, pd.Series(less_than_window[less_than_window <= -2])])

    rule5 = data.loc[index_list]

    #Rule 6: 4/5 points for than 1-sigma from the centerline (same side)
    temp_df6 = data - mean

    more_than = data - mean
    more_than[more_than < std] = 0
    more_than[more_than > std] = 1
    more_than_window = more_than.rolling(window= 5, step= 1).sum()

    less_than = data - mean
    less_than[less_than > -std] = 0
    less_than[less_than <  -std] = -1
    less_than_window = less_than.rolling(window= 5, step= 1).sum()
    
    index_list = pd.Series(more_than_window[more_than_window >= 5].index)
    index_list = pd.concat([index_list, pd.Series(less_than_window[less_than_window <= -5].index)])

    rule6 = data.loc[index_list]

    #Rule 7: 15 points in a row less than 1-sigma from the centerline
    betweens = data.copy()
    betweens[betweens.between(mean - 1*std, mean + 1*std)] = 1
    betweens[betweens != 1] = 0

    windowed_df = betweens.rolling(window= 15, step= 1).sum()

    rule7 = data.loc[windowed_df[windowed_df >= 15].index]

    #Rule 8: 8 points in a row more than 1-sigma from the centerline
    temp_df8 = data - mean
    temp_df8[abs(temp_df8) > 1*std] = 1
    temp_df8[abs(temp_df8) <= 1*std] = 0

    windowed_df = temp_df8.rolling(window= 8, step = 1).sum()
    rule8 = data.loc[windowed_df[windowed_df == 8].index]
    
    #print(rule7)

    #Plot rule 1
    ax.scatter(rule1.index, rule1, color = 'r', zorder = 4)
    annotate_rule_violation(rule1, ax= ax, violation_number= 1)

    #Plot rule 2
    ax.scatter(rule2.index, rule2, color = '#f5ed05', zorder = 3)
    annotate_rule_violation(rule2, ax= ax, violation_number= 2)

    #Plot rule 3
    #ax.scatter(rule3.index, rule3, color = 'r', zorder = 3)

    #Plot rule 4
    ax.scatter(rule4.index, rule4, color = '#f5ed05', zorder = 3)
    annotate_rule_violation(rule4, ax= ax, violation_number= 4)

    #Plot rule 5
    ax.scatter(rule5.index, rule5, color = '#f5ed05', zorder = 3)
    annotate_rule_violation(rule5, ax= ax, violation_number= 5)

    #Plot rule 6
    ax.scatter(rule6.index, rule6, color = '#f5ed05', zorder = 3)
    annotate_rule_violation(rule6, ax= ax, violation_number= 6)

    #Plot rule 7
    ax.scatter(rule7.index, rule7, color = '#f5ed05', zorder = 3)
    annotate_rule_violation(rule7, ax= ax, violation_number= 7)

    #Plot rule 8
    ax.scatter(rule8.index, rule8, color = '#f5ed05', zorder = 3)
    annotate_rule_violation(rule8, ax= ax, violation_number= 8)



def annotate_rule_violation(violation_data: pd.DataFrame | pd.Series, ax, violation_number: str) -> None:
    '''This function annotates the rule violations for out of control points
        Args:
            violation_data: A DataFrame or Series containing the violation data
            ax: matplotlib axis to plot the data to
            violation_number: rule number violation
    '''

    if isinstance(violation_data, pd.Series):
        violation_data = pd.DataFrame(violation_data)

    for entry in violation_data.iterrows():
        ax.annotate(str(violation_number), 
                    (entry[0], entry[1]),
                    textcoords = 'offset points',
                    xytext = (1, 1))
        