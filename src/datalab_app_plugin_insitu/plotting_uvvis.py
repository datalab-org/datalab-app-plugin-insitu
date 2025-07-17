# Plotting UV-Vis module
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from bokeh.events import DoubleTap
from bokeh.layouts import gridplot
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    CrosshairTool,
    CustomJS,
    HoverTool,
    LinearColorMapper,
    Range1d,
    TapTool,
)
from bokeh.plotting import figure

try:
    from pydatalab.bokeh_plotting import COLORS
except ImportError:
    from bokeh.palettes import Dark2

    COLORS = Dark2[8]


def create_linked_insitu_plots(
    plot_data, time_series_time_range, heatmap_time_range, link_plots: bool = False
):
    shared_ranges = _create_shared_ranges(plot_data, time_series_time_range, heatmap_time_range)
    heatmap_figure = _create_heatmap_figure(plot_data, shared_ranges)
    uvvisplot_figure = _create_top_line_figure(plot_data, shared_ranges)
    echemplot_figure = _create_echem_figure(plot_data, shared_ranges)

    heatmap_figure.js_on_event(
        DoubleTap, CustomJS(args=dict(p=heatmap_figure), code="p.reset.emit()")
    )
    uvvisplot_figure.js_on_event(
        DoubleTap, CustomJS(args=dict(p=uvvisplot_figure), code="p.reset.emit()")
    )
    echemplot_figure.js_on_event(
        DoubleTap, CustomJS(args=dict(p=echemplot_figure), code="p.reset.emit()")
    )

    if link_plots:
        _link_plots(heatmap_figure, uvvisplot_figure, echemplot_figure, plot_data)

    grid = [[None, uvvisplot_figure], [echemplot_figure, heatmap_figure]]
    gp = gridplot(grid, merge_tools=True)

    return gp


def prepare_uvvis_plot_data(
    two_d_data: pd.DataFrame, wavelength: pd.Series, echem_data, metadata, file_num_index
) -> Optional[Dict[str, Any]]:
    """
    Need heatmap data in two forms:
    1. Two-dimensional array for numpy.
    2. List of lists for JSON.
    """
    twoD_matrix = two_d_data.values

    # Grab the times and voltages for each scan from the echem data
    times = two_d_data.index.to_numpy()
    voltage_interp = np.interp(times, echem_data["time"], echem_data["Voltage"])

    spectra_intensities = two_d_data.values.tolist()

    first_spectrum_intensities = twoD_matrix[0, :]

    intensity_min = np.min(twoD_matrix)
    intensity_max = np.max(twoD_matrix)

    echem_data = {
        "Voltage": echem_data["Voltage"].values,
        "time": echem_data["time"].values,
    }

    return {
        "heatmap x_values": wavelength,  # ppm_values
        "heatmap y_values": two_d_data.index,  # not in ben Cs code
        "num_experiments": metadata["num_experiments"],
        "spectra_intensities": spectra_intensities,
        "intensity_matrix": twoD_matrix,
        "time_range": metadata["time_range"],
        "first_spectrum_intensities": first_spectrum_intensities,
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "echem_data": echem_data,
        "times_by_exp": times,
        "voltages_by_exp": voltage_interp,
        "file_num_index": file_num_index,
    }


def _create_shared_ranges(
    plot_data: Dict[str, Any],
    time_series_time_range: Dict[str, float],
    heatmap_time_range: Dict[str, float],
) -> Dict[str, Range1d]:
    """
    Create shared range objects for linking multiple plots.

    Args:
        plot_data: Dictionary containing prepared plot data

    Returns:
        Dict[str, Range1d]: Dictionary of shared range objects
    """
    overall_min_time = min(time_series_time_range["min_time"], heatmap_time_range["min_time"])
    overall_max_time = max(time_series_time_range["max_time"], heatmap_time_range["max_time"])
    time_range = {"min_time": overall_min_time, "max_time": overall_max_time}
    intensity_min = np.min(plot_data["intensity_matrix"])
    intensity_max = np.max(plot_data["intensity_matrix"])
    shared_y_range = Range1d(start=time_range["min_time"], end=time_range["max_time"])

    shared_x_range = Range1d(
        start=min(plot_data["heatmap x_values"]), end=max(plot_data["heatmap x_values"])
    )

    intensity_range = Range1d(start=intensity_min, end=intensity_max)

    return {
        "shared_y_range": shared_y_range,
        "shared_x_range": shared_x_range,
        "intensity_range": intensity_range,
    }


def _create_heatmap_figure(plot_data: Dict[str, Any], ranges: Dict[str, Range1d]) -> figure:
    """
    Create the heatmap figure component.

    Args:
        plot_data: Dictionary containing prepared plot data
        ranges: Dictionary of shared range objects

    Returns:
        figure: Configured Bokeh heatmap figure
    """
    heatmap_x_values = plot_data["heatmap x_values"]
    intensity_matrix = plot_data["intensity_matrix"]
    time_range = plot_data["time_range"]
    intensity_min = plot_data["intensity_min"]
    intensity_max = plot_data["intensity_max"]

    tools = "pan,wheel_zoom,box_zoom,reset,save"

    heatmap_figure = figure(
        x_axis_label="Wavelength (nm)",
        y_axis_label="Time (s)",
        x_range=ranges["shared_x_range"],
        y_range=ranges["shared_y_range"],
        height=400,
        tools=tools,
    )

    color_mapper = LinearColorMapper(palette="Viridis256", low=intensity_min, high=intensity_max)

    heatmap_figure.image(
        image=[intensity_matrix],
        x=min(heatmap_x_values),
        y=time_range["min_time"],
        dw=abs(max(heatmap_x_values) - min(heatmap_x_values)),
        dh=time_range["max_time"] - time_range["min_time"],
        color_mapper=color_mapper,
        level="image",
    )

    time_points = len(intensity_matrix)
    if time_points > 0:
        times = np.linspace(time_range["min_time"], time_range["max_time"], time_points)
        experiment_numbers = plot_data["file_num_index"].flatten().tolist()
        source = ColumnDataSource(
            data={
                "x": [(max(heatmap_x_values) + min(heatmap_x_values)) / 2] * time_points,
                "y": times,
                "width": [abs(max(heatmap_x_values) - min(heatmap_x_values))] * time_points,
                "height": [(time_range["max_time"] - time_range["min_time"]) / time_points]
                * time_points,
                "exp_num": experiment_numbers,
            }
        )

        rects = heatmap_figure.rect(
            x="x",
            y="y",
            width="width",
            height="height",
            source=source,
            fill_alpha=0,
            line_alpha=0,
        )

        hover_tool = HoverTool(
            renderers=[rects],
            tooltips=[("Exp. #", "@exp_num")],
            mode="mouse",
            point_policy="follow_mouse",
        )

        heatmap_figure.add_tools(hover_tool)

        plot_data["heatmap_source"] = source

    heatmap_figure.grid.grid_line_width = 0
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
    heatmap_figure.add_layout(color_bar, "right")

    return heatmap_figure


def _create_top_line_figure(plot_data: Dict[str, Any], ranges: Dict[str, Range1d]) -> figure:
    """
    Create the UV-Vis line plot figure component.

    Args:
        plot_data: Dictionary containing prepared plot data
        ranges: Dictionary of shared range objects

    Returns:
        figure: Configured Bokeh line figure with data source
    """
    heatmap_x_values = plot_data["heatmap x_values"]
    first_spectrum_intensities = plot_data["first_spectrum_intensities"]

    tools = "pan,wheel_zoom,box_zoom,reset,save"

    heatmap_x_value_list = (
        heatmap_x_values.tolist() if isinstance(heatmap_x_values, np.ndarray) else heatmap_x_values
    )
    intensity_list = (
        first_spectrum_intensities.tolist()
        if isinstance(first_spectrum_intensities, np.ndarray)
        else first_spectrum_intensities
    )

    line_source = ColumnDataSource(
        data={
            "x": heatmap_x_value_list,
            "intensity": intensity_list,
        }
    )

    clicked_spectra_source = ColumnDataSource(
        data={"x": [], "intensity": [], "label": [], "color": []}
    )

    plot_figure = figure(
        y_axis_label="Intensity",
        aspect_ratio=2,
        x_range=ranges["shared_x_range"],
        y_range=ranges["intensity_range"],
        tools=tools,
    )

    plot_figure.line(
        x="x",
        y="intensity",
        source=line_source,
        line_width=1,
        color="grey",
    )

    plot_figure.multi_line(
        xs="x",
        ys="intensity",
        source=clicked_spectra_source,
        line_color="color",
        line_width=1,
        legend_field="label",
    )

    plot_figure.legend.click_policy = "hide"
    plot_figure.legend.location = "top_right"

    plot_data["line_source"] = line_source
    plot_data["clicked_spectra_source"] = clicked_spectra_source
    return plot_figure


def _create_echem_figure(plot_data: Dict[str, Any], ranges: Dict[str, Range1d]) -> figure:
    """
    Create the electrochemical data figure component.

    Args:
        plot_data: Dictionary containing prepared plot data
        ranges: Dictionary of shared range objects

    Returns:
        figure: Configured Bokeh electrochemical figure
    """
    echem_data = plot_data["echem_data"]

    tools = "pan,wheel_zoom,box_zoom,reset,save"

    echemplot_figure = figure(
        x_axis_label="Voltage (V)",
        y_axis_label="Time (s)",
        y_range=ranges["shared_y_range"],
        height=400,
        width=250,
        tools=tools,
    )

    if echem_data and "Voltage" in echem_data and "time" in echem_data:
        times = np.array(echem_data["time"])
        voltages = np.array(echem_data["Voltage"])

        time_range = plot_data["time_range"]
        time_span = time_range["max_time"] - time_range["min_time"]
        exp_count = plot_data["num_experiments"]

        exp_numbers = np.floor(((times - time_range["min_time"]) / time_span) * exp_count) + 1
        exp_numbers = np.clip(exp_numbers, 1, exp_count)
        echem_source = ColumnDataSource(
            data={"time": times, "voltage": voltages, "exp_num": exp_numbers}
        )

        echemplot_figure.line(x="voltage", y="time", source=echem_source, color=COLORS[1])

        hover_tool = HoverTool(
            tooltips=[
                ("Exp. #", "@exp_num{0}"),
                ("Time (h)", "@time{0.00}"),
                ("Voltage (V)", "@voltage{0.000}"),
            ],
            mode="hline",
            point_policy="snap_to_data",
        )

        echemplot_figure.add_tools(hover_tool)

    return echemplot_figure


def _link_plots(
    heatmap_figure: figure,
    uvvisplot_figure: figure,
    echemplot_figure: figure,
    plot_data: Dict[str, Any],
) -> None:
    """
    Link the plots together with interactive tools and callbacks.

    Args:
        heatmap_figure: The heatmap figure component
        uvvisplot_figure: The UV-Vis line plot figure component
        echemplot_figure: The electrochemical figure component
        plot_data: Dictionary containing prepared plot data
    """
    line_source = plot_data["line_source"]
    clicked_spectra_source = plot_data["clicked_spectra_source"]
    spectra_intensities = plot_data["spectra_intensities"]
    ppm_values = plot_data["heatmap x_values"]
    intensity_matrix = plot_data["intensity_matrix"]
    heatmap_source = plot_data.get("heatmap_source")

    crosshair = CrosshairTool(dimensions="width", line_color="grey")
    heatmap_figure.add_tools(crosshair)
    echemplot_figure.add_tools(crosshair)

    hover = next((tool for tool in heatmap_figure.tools if isinstance(tool, HoverTool)), None)
    if hover:
        hover.callback = CustomJS(
            args=dict(
                line_source=line_source,
                spectra_intensities=spectra_intensities,
                ppm_values=ppm_values.tolist(),
                heatmap_source=heatmap_source,
            ),
            code="""
                    const geometry = cb_data['geometry'];

                    let closestIndex = 0;
                    let minDistance = Infinity;
                    for (let i = 0; i < heatmap_source.data.y.length; i++) {
                        const distance = Math.abs(heatmap_source.data.y[i] - geometry.y);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestIndex = i;
                        }
                    }

                    const exp_num = heatmap_source.data.exp_num[closestIndex];

                    const index = exp_num - 1;

                    var data = line_source.data;
                    data['intensity'] = spectra_intensities[closestIndex];
                    line_source.change.emit();
                """,
        )

        leave_callback = CustomJS(
            args=dict(line_source=line_source),
            code="""
                var data = line_source.data;
                data['intensity'] = [];  // Clear the intensity data
                line_source.change.emit();
            """,
        )

        heatmap_figure.js_on_event("mouseleave", leave_callback)

    if heatmap_source:
        tap_tool = TapTool()
        heatmap_figure.add_tools(tap_tool)
        tap_callback = CustomJS(
            args=dict(
                heatmap_source=heatmap_source,
                clicked_spectra_source=clicked_spectra_source,
                spectra_intensities=spectra_intensities,
                ppm_values=ppm_values.tolist(),
                colors=COLORS,
                times_by_exp=plot_data["times_by_exp"].tolist(),
                voltages_by_exp=plot_data["voltages_by_exp"].tolist(),
            ),
            code="""
                    const indices = cb_obj.indices;
                    if (indices.length === 0) return;

                    const index = indices[0];
                    const exp_num = heatmap_source.data.exp_num[index];
                    const exp_index = exp_num - 1;

                    const time = times_by_exp[index].toFixed(0);
                    const voltage = voltages_by_exp[index].toFixed(3);
                    const label = `#${exp_num} @ ${time} s, ${voltage} V`;

                    const existing_labels = clicked_spectra_source.data.label;
                    if (existing_labels.includes(label)) return;

                    const color_index = existing_labels.length % colors.length;

                    const new_xs = [...clicked_spectra_source.data['x']];
                    const new_ys = [...clicked_spectra_source.data.intensity];
                    const new_labels = [...clicked_spectra_source.data.label];
                    const new_colors = [...clicked_spectra_source.data.color];

                    new_xs.push(ppm_values);
                    new_ys.push(spectra_intensities[index]);
                    new_labels.push(label);
                    new_colors.push(colors[color_index]);

                    clicked_spectra_source.data = {
                        'x': new_xs,
                        'intensity': new_ys,
                        'label': new_labels,
                        'color': new_colors
                    };

                    clicked_spectra_source.change.emit();
                """,
        )

        heatmap_source.selected.js_on_change("indices", tap_callback)

    heatmap_figure.x_range.js_on_change(
        "start",
        CustomJS(
            args=dict(
                color_mapper=heatmap_figure.select_one(LinearColorMapper),
                intensity_matrix=intensity_matrix.tolist(),
                ppm_array=ppm_values.tolist(),
                global_min=np.min(intensity_matrix),
                global_max=np.max(intensity_matrix),
            ),
            code="""
                    const start_index = ppm_array.findIndex(ppm => ppm <= cb_obj.end);
                    const end_index = ppm_array.findIndex(ppm => ppm <= cb_obj.start);

                    if (start_index < 0 || end_index < 0 || start_index >= ppm_array.length || end_index >= ppm_array.length) {
                        color_mapper.low = global_min;
                        color_mapper.high = global_max;
                        return;
                    }

                    if (Math.abs(end_index - start_index) < 5) {
                        return;
                    }

                    let min_intensity = Infinity;
                    let max_intensity = -Infinity;

                    for (let i = 0; i < intensity_matrix.length; i++) {
                        for (let j = Math.min(start_index, end_index); j <= Math.max(start_index, end_index); j++) {
                            if (j >= 0 && j < intensity_matrix[i].length) {
                                const value = intensity_matrix[i][j];
                                min_intensity = Math.min(min_intensity, value);
                                max_intensity = Math.max(max_intensity, value);
                            }
                        }
                    }

                    if (Math.abs(max_intensity - min_intensity) < 0.1 * Math.abs(global_max - global_min)) {
                        const padding = 0.1 * Math.abs(global_max - global_min);
                        min_intensity = Math.max(min_intensity - padding, global_min);
                        max_intensity = Math.min(max_intensity + padding, global_max);
                    }

                    color_mapper.low = min_intensity;
                    color_mapper.high = max_intensity;
                """,
        ),
    )

    heatmap_figure.x_range.tags = [ppm_values.tolist(), intensity_matrix.tolist()]

    line_y_range = uvvisplot_figure.y_range
    line_y_range.js_link("start", heatmap_figure.select_one(LinearColorMapper), "low")
    line_y_range.js_link("end", heatmap_figure.select_one(LinearColorMapper), "high")

    tap_tool = TapTool()

    uvvisplot_figure.add_tools(tap_tool)

    remove_line_callback = CustomJS(
        args=dict(clicked_spectra_source=clicked_spectra_source),
        code="""
        const indices = clicked_spectra_source.selected.indices;
        if (indices.length === 0) return;

        let data = clicked_spectra_source.data;

        for (let i = indices.length - 1; i >= 0; i--) {
            let index = indices[i];
            data['x'].splice(index, 1);
            data['intensity'].splice(index, 1);
            data['exp_index'].splice(index, 1);
            data['color'].splice(index, 1);
        }

        clicked_spectra_source.change.emit();
    """,
    )

    clicked_spectra_source.selected.js_on_change("indices", remove_line_callback)
