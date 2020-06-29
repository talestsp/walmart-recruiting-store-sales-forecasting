import pandas as pd
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter
from bokeh.models import Range1d


def wide_tools():
    return "pan,ywheel_zoom,xwheel_zoom,reset,save"


def get_time_series_figure(width=400, height=400, title=""):
    fig = figure(plot_width=width, plot_height=height, x_axis_type='datetime', tools=wide_tools(), title=title)
    fig.xaxis.formatter = DatetimeTickFormatter(
        hours=["%H:%M %d/%b"],
        days=["%d/%b"],
        months=["%d/%b/%Y"],
        years=["%d/%b/%Y"],
    )
    return fig


def plot_time_series_count(str_datetimes, values, color, title="", relative_y_axis=False, alpha=0.8, width=900,
                           height=300, p=None):
    if not p:
        p = get_time_series_figure(width=width, height=height, title=title)

    datetimes = pd.to_datetime(str_datetimes)
    p.circle(datetimes, values, size=3, color=color, alpha=alpha)
    p.line(datetimes, values, line_width=1, color=color, alpha=alpha)

    if not relative_y_axis:
        p.y_range = Range1d(0, max(values) * 1.1)

    return p