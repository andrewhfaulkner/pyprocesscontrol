# This script will be for building bokeh control charts for display in HTML format

import bokeh as bk
from bokeh.io import output_file, show
from bokeh.layouts import column as bk_column
from bokeh.layouts import gridplot as gp
from bokeh.layouts import layout
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import Span, BoxAnnotation, LabelSet, Label, Div, HoverTool
import pandas as pd
import numpy as np
import math
import scipy.stats as st

from pyprocesscontrol import tools


def plot_control_charts(
    data: tools.datastructures.metadata,
    index: str,
    product: str = None,
):
    """
    This function will plot the control charts for a product in the metadata or raw_data classes

    Args:
        data: the metadata or rawdata class containg all the data
        index: String column name for the index of the data

    Returns:
        None
    """

    for group in data.raw_data.groups[product]:
        if type(data) == tools.datastructures.metadata:
            include_columns = data.metadata[
                (data.metadata["Product"] == product) & (data.metadata["Grade"]
                == group[1])
            ]["Spec"]

            usl_dict = {}
            lsl_dict = {}

            for index, rdata in include_columns.items():
                try:
                    usl_dict[rdata] = data.metadata["USL"].loc[int(index)]
                    lsl_dict[rdata] = data.metadata["LSL"].loc[int(index)]
                except:
                    pass

        fname = str(group)

        basic_control_chart(
            data=data.raw_data.groups[product][group],
            index=index,
            datetime_index=True,
            include_cols=include_columns,
            filename = fname,
            usl = usl_dict,
            lsl = lsl_dict
        )


def basic_control_chart(
    data: pd.DataFrame | pd.Series,
    index: str,
    datetime_index: bool = False,
    include_cols: list = None,
    exclude_cols: list = None,
    filename: str = None,
    grouped: bool = False,
    usl=None,
    lsl=None,
):
    """This function plots a control chart of data

    Args:
        Data: Data to be plotted
        index: The index for the data
        datetime_index: boolean if the index is a datetime or not
        include_cols: Columns to include in the analysis
        exclude_cols: Columns to exclude in the analysis
        filename: The name of the output html file
        grouped: Logical grouping of datapoints
        usl: upper spec limit of the data being plotted
        lsl: lower spec limit of the data being plotted
    """

    fig_list = []

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

    # If specific columns are needed this block selects them
    if exclude_cols != None:
        col_names = col_names.drop(exclude_cols)

    if (include_cols is not None) and (include_cols.empty == False):
        col_names = include_cols

    data = data[col_names]
    j = 0
    for column in data.columns:
        num = 1
        all_columns = list(data.columns)
        if j == 0:
            lst = [column]
            j = 1
            continue

        temp_col = column
        while column in lst:
            column = temp_col + "_" + str(num)

            if column in lst:
                num += 1

        lst.append(column)

    data.columns = lst
    col_names = data.columns

    # Clean up the data in the data set to only process columns with data
    count = data[col_names].count()

    drop_cols = count[count == 0]
    data = data.drop(drop_cols.index, axis=1)

    for column in data.columns:
        series = pd.to_numeric(data[column], errors="coerce")
        data[column] = series
        if data[column].isnull().all():
            data = data.drop(column, axis=1)

    if data.isnull().values.all() == True:
        return

    col_names = data.columns
    data = data.sort_index(axis=0, ascending=True)
    stats = data.describe()

    col_names = np.array(col_names)
    num_figs = math.ceil(len(col_names) / 2)

    fig = []
    i = 0
    fig_num = 0

    output_file(filename=filename + ".html", title="Test Control Chart")

    for column in col_names:
        if (
            len(data[column]) / 30 >= 2
            and len(data[column]) / 30 < 10
            and grouped == True
        ):
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

        d = Div(
            text=title_str + column,
            styles={"text-align": "center", "font-size": "16pt"},
        )

        tooltips = [
            ("Date", "$snap_x{%m/%d/%y}"),
            ("Value", "$snap_y{0.00}"),
        ]

        p = figure(
            sizing_mode="stretch_width",
            height=150,
            width=300,
            x_axis_type="datetime",
            min_border=0,
            x_axis_location=None,
            # title = title_str + column,
            tools="xpan,xwheel_zoom",
        )

        h = figure(sizing_mode="stretch_height", width=300, height=300)

        r = figure(
            sizing_mode="stretch_width",
            height=150,
            width=300,
            x_axis_type="datetime",
            x_range=p.x_range,
        )

        p.hspan(data[column].mean(), line_width=1, line_color="red", line_dash="dashed")

        p.hspan(upper_control, line_width=1, line_color="black", line_dash="dashed")

        p.hspan(lower_control, line_width=1, line_color="black", line_dash="dashed")

        one_sigma_box = BoxAnnotation(
            top=data[column].mean() + data[column].std(),
            bottom=data[column].mean() - data[column].std(),
            fill_color="#f2f2f2",
            level="underlay",
        )

        u_two_sigma_box = BoxAnnotation(
            top=data[column].mean() + 2 * data[column].std(),
            bottom=data[column].mean() + data[column].std(),
            fill_color="#e6e6e6",
            fill_alpha=0.5,
            level="underlay",
        )

        u_three_sigma_box = BoxAnnotation(
            top=data[column].mean() + 3 * data[column].std(),
            bottom=data[column].mean() + 2 * data[column].std(),
            fill_color="#cccccc",
            fill_alpha=0.5,
            level="underlay",
        )

        l_two_sigma_box = BoxAnnotation(
            top=data[column].mean() - data[column].std(),
            bottom=data[column].mean() - 2 * data[column].std(),
            fill_color="#e6e6e6",
            fill_alpha=0.5,
            level="underlay",
        )

        l_three_sigma_box = BoxAnnotation(
            top=data[column].mean() - 2 * data[column].std(),
            bottom=data[column].mean() - 3 * data[column].std(),
            fill_color="#cccccc",
            fill_alpha=0.5,
            level="underlay",
        )

        hover = HoverTool(tooltips=tooltips, point_policy="snap_to_data", mode="vline")
        hover.formatters = {"$snap_x": "datetime"}
        p.add_tools(hover)

        p.add_layout(one_sigma_box)
        p.add_layout(u_two_sigma_box)
        p.add_layout(u_three_sigma_box)
        p.add_layout(l_two_sigma_box)
        p.add_layout(l_three_sigma_box)

        p.circle(
            data[column].index,
            data[column],
            size=4,
            color="navy",
            alpha=0.5,
            name="datapoints",
        )

        p.line(data[column].index, data[column], color="navy")

        r.line(rng.index, rng, color="navy")

        r.hspan(r_bar, line_width=1, line_color="red", line_dash="dashed")

        r.hspan(r_upper_control, line_width=1, line_color="black", line_dash="dashed")

        p.toolbar.autohide = True
        r.toolbar.autohide = True
        h.toolbar.autohide = True

        ooc_checker(data=data[column], ax=p)
        histogram_builder(data=data[column], ax=h)

        if usl != None:
            try:
                h.vspan(x=usl[column], line_color="red")
            except:
                pass

        if lsl != None:
            try:
                h.vspan(x=lsl[column], line_color="red")
            except:
                pass

        c = bk_column(p, r)

        grid = layout([[d], [c, h]], sizing_mode="stretch_width")
        fig_list.append(grid)
        # show(p)

    save(bk_column(fig_list, sizing_mode="scale_width"))


def ooc_checker(data: pd.DataFrame | pd.Series, ax: figure, column: str = None) -> None:
    """This function checks the data to see if the data is potentially out of control
    due to special cause variation. This will use the 8 rules for out of control
    processes.
    """

    if isinstance(data, pd.Series):
        mean = data.mean()
        std = data.std()
    elif isinstance(data, pd.DataFrame):
        stats = data.describe()
        mean = stats[column].loc["mean"]
        std = stats[column].loc["std"]

    centerline_distance = data - mean

    # Rule 1: 1 point 3-sigma from the centerline
    rule1 = data[~data.between(mean - 3 * std, mean + 3 * std)]

    # Rule 2: 9 points in a row on the same side of the centerline
    temp_df2 = data - mean
    temp_df2[temp_df2 > 0] = 1
    temp_df2[temp_df2 < 0] = 0

    more_than = temp_df2
    more_than[more_than < 0] = 0
    more_than[more_than > 1] = 1
    more_than_window = more_than.rolling(window=9, step=1).sum()

    less_than = temp_df2
    less_than[less_than > 0] = 0
    less_than[less_than < -1 * std] = -1
    less_than_window = less_than.rolling(window=9, step=1).sum()

    index_list = pd.Series(more_than_window[more_than_window >= 9].index)
    index_list = pd.concat(
        [index_list, pd.Series(less_than_window[less_than_window <= -9])]
    )

    rule2 = data.loc[index_list]

    # Rule 3: Six points in a row, all increasing or decreasing

    # Rule 4: 14 points in a row all oscillating
    temp_df4 = data - mean
    dropped = temp_df4[:-1]
    shifted = temp_df4[1:]
    oscillating = dropped - shifted
    oscillating[oscillating > 0] = 1
    oscillating[oscillating < 0] = -1
    windowed_df = oscillating.rolling(window=14, step=1).sum()
    rule4 = data.loc[windowed_df[windowed_df >= 14].index]

    # Rule 5: 2/3 points more than 2-sigma from the centerline (same side)
    temp_df5 = data - mean

    more_than = data - mean
    more_than[more_than < 2 * std] = 0
    more_than[more_than > 2 * std] = 1
    more_than_window = more_than.rolling(window=3, step=1).sum()

    less_than = data - mean
    less_than[less_than > -2 * std] = 0
    less_than[less_than < -2 * std] = -1
    less_than_window = less_than.rolling(window=3, step=1).sum()

    index_list = pd.Series(more_than_window[more_than_window >= 2].index)
    index_list = pd.concat(
        [index_list, pd.Series(less_than_window[less_than_window <= -2].index)]
    )

    rule5 = data.loc[index_list]

    # Rule 6: 4/5 points for than 1-sigma from the centerline (same side)
    temp_df6 = data - mean

    more_than = data - mean
    more_than[more_than < std] = 0
    more_than[more_than > std] = 1
    more_than_window = more_than.rolling(window=5, step=1).sum()

    less_than = data - mean
    less_than[less_than > -std] = 0
    less_than[less_than < -std] = -1
    less_than_window = less_than.rolling(window=5, step=1).sum()

    index_list = pd.Series(more_than_window[more_than_window >= 5].index)
    index_list = pd.concat(
        [index_list, pd.Series(less_than_window[less_than_window <= -5].index)]
    )

    rule6 = data.loc[index_list]

    # Rule 7: 15 points in a row less than 1-sigma from the centerline
    betweens = data.copy()
    betweens[betweens.between(mean - 1 * std, mean + 1 * std)] = 1
    betweens[betweens != 1] = 0

    windowed_df = betweens.rolling(window=15, step=1).sum()

    rule7 = data.loc[windowed_df[windowed_df >= 15].index]

    # Rule 8: 8 points in a row more than 1-sigma from the centerline
    temp_df8 = data - mean
    temp_df8[abs(temp_df8) > 1 * std] = 1
    temp_df8[abs(temp_df8) <= 1 * std] = 0

    windowed_df = temp_df8.rolling(window=8, step=1).sum()
    rule8 = data.loc[windowed_df[windowed_df == 8].index]

    # print(rule7)

    # Plot rule 1
    ax.circle(rule1.index, rule1, color="red", level="overlay")
    annotate_rule_violation(rule1, ax=ax, violation_number=1)

    # Plot rule 2
    ax.circle(rule2.index, rule2, color="#f5ed05")
    annotate_rule_violation(rule2, ax=ax, violation_number=2)

    # Plot rule 3
    # ax.scatter(rule3.index, rule3, color = 'r', zorder = 3)

    # Plot rule 4
    ax.circle(rule4.index, rule4, color="#f5ed05")
    annotate_rule_violation(rule4, ax=ax, violation_number=4)

    # Plot rule 5
    ax.circle(rule5.index, rule5, color="#f5ed05")
    annotate_rule_violation(rule5, ax=ax, violation_number=5)

    # Plot rule 6
    ax.circle(rule6.index, rule6, color="#f5ed05")
    annotate_rule_violation(rule6, ax=ax, violation_number=6)

    # Plot rule 7
    ax.circle(rule7.index, rule7, color="#f5ed05")
    annotate_rule_violation(rule7, ax=ax, violation_number=7)

    # Plot rule 8
    ax.circle(rule8.index, rule8, color="#f5ed05")
    annotate_rule_violation(rule8, ax=ax, violation_number=8)


def annotate_rule_violation(
    violation_data: pd.DataFrame | pd.Series, ax, violation_number: str
) -> None:
    """This function annotates the rule violations for out of control points
    Args:
        violation_data: A DataFrame or Series containing the violation data
        ax: matplotlib axis to plot the data to
        violation_number: rule number violation
    """

    if isinstance(violation_data, pd.Series):
        violation_data = pd.DataFrame(violation_data)

    for entry in violation_data.iterrows():
        labels = Label(
            x=entry[0].to_pydatetime(),
            y=entry[1][0],
            text=str(violation_number),
        )

        ax.add_layout(labels)


def histogram_builder(
    data: pd.DataFrame,
    ax: figure,
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
    x_axis = np.linspace(min[0], max[0], num=100)

    count = stats[col_names].loc["count"]

    # Creates and plots the histogram figure
    counts, bins = np.histogram(data[col_names].dropna())
    bin_width = (data_max - data_min) / len(bins)
    ax.quad(
        top=counts,
        bottom=0,
        left=bins[:-1],
        right=bins[1:],
        fill_color="skyblue",
        line_color="white",
    )

    pdf = st.norm.pdf(
        x_axis, loc=stats[col_names].loc["mean"], scale=stats[col_names].loc["std"]
    )
    pdf = pdf * bin_width[0] * count[0]

    ax.line(x_axis, pdf, line_width=2, line_dash="dashed", line_color="black")

    # ax.axvline(lower_spec, color = 'r')
    # ax.axvline(upper_spec, color = 'r')
    # ax.vline(avg, line_color="#82F85F")

    # if upper_spec > max:
    # upper_rng = upper_spec
    # else:
    # upper_rng = max

    # if lower_spec < min:
    # lower_rng = lower_spec
    # else:
    # lower_rng = min

    # ax.set_xlim((lower_rng-range/100, upper_rng+range/100))
