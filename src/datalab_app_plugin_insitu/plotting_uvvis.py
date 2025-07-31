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
    plot_data,
    time_series_time_range: Dict[str, float],
    heatmap_time_range: Dict[str, float],
    plotting_label_dict: dict,
    link_plots: bool = False,
):
    shared_ranges = _create_shared_ranges(plot_data, time_series_time_range, heatmap_time_range)
    heatmap_figure = _create_heatmap_figure(
        plot_data,
        shared_ranges,
        x_axis_label=plotting_label_dict.get("x_axis_label", "Wavelength (nm)"),
        time_series_y_axis_label=plotting_label_dict.get("time_series_y_axis_label", "Time (h)"),
    )
    uvvisplot_figure = _create_top_line_figure(
        plot_data,
        shared_ranges,
        line_y_axis_label=plotting_label_dict.get("line_y_axis_label", "Intensity (a.u.)"),
    )
    echemplot_figure = _create_echem_figure(
        plot_data,
        shared_ranges,
        time_series_x_axis_label=plotting_label_dict.get("time_series_x_axis_label", "Voltage (V)"),
        time_series_y_axis_label=plotting_label_dict.get("time_series_y_axis_label", "Time (h)"),
    )

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
        _link_plots(
            heatmap_figure, uvvisplot_figure, echemplot_figure, plot_data, plotting_label_dict
        )

    grid = [[None, uvvisplot_figure], [echemplot_figure, heatmap_figure]]
    gp = gridplot(grid, merge_tools=True)

    return gp


def prepare_uvvis_plot_data(
    intensity_matrix: pd.DataFrame,
    spectra_intensities: pd.DataFrame,
    wavelength: pd.Series,
    echem_data,
    metadata,
    file_num_index,
    index_df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """
    Need heatmap data in two forms:
    1. Two-dimensional array for numpy.
    2. List of lists for JSON.
    """
    twoD_matrix = intensity_matrix.values

    # Grab the times and voltages for each scan from the echem data
    times = spectra_intensities.index.to_numpy()
    voltage_interp = np.interp(times, echem_data["time"], echem_data["Voltage"])
    index_df["voltage"] = voltage_interp
    first_spectrum_intensities = twoD_matrix[0, :]

    spectra_intensities = spectra_intensities.values.tolist()

    intensity_min = np.min(twoD_matrix)
    intensity_max = np.max(twoD_matrix)

    echem_data = {
        "Voltage": echem_data["Voltage"].values,
        "time": echem_data["time"].values,
    }

    return {
        "heatmap x_values": wavelength,  # ppm_values
        "heatmap y_values": intensity_matrix.index,  # not in ben Cs code
        "num_experiments": metadata["num_experiments"],
        "spectra_intensities": spectra_intensities,
        "intensity_matrix": twoD_matrix,
        "y_range": metadata["time_range"],
        "first_spectrum_intensities": first_spectrum_intensities,
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "time_series_data": echem_data,
        "file_num_index": file_num_index,
        "index_df": index_df,  # DataFrame with index and exp_num columns
    }


def prepare_xrd_plot_data(
    intensity_matrix: pd.DataFrame,
    spectra_intensities: pd.DataFrame,
    heatmap_x_values: pd.Series,
    time_series_data,
    metadata,
    file_num_index,
    sample_granularity,
    index_df: pd.DataFrame,
) -> Dict[str, Any]:
    intensity_matrix = intensity_matrix.values
    first_spectrum_intensities = spectra_intensities.values[0, :]

    intensity_min = np.min(intensity_matrix)
    intensity_max = np.max(intensity_matrix)

    time_series_data = {
        "x": time_series_data["x"],
        "y": np.arange(1, len(time_series_data["x"]) + 1),
        # We aren't actually plotting by filenum for xrd data
        "filenum": time_series_data["y"],
    }

    heatmap_y_range = {
        "min_y": 1,
        "max_y": np.arange(1, len(spectra_intensities) + 1).max(),
    }

    y_range = {"min_y": 1, "max_y": int(max(np.arange(1, len(time_series_data["x"]) + 1)))}

    return {
        "heatmap x_values": heatmap_x_values,  # ppm_values
        "heatmap y_values": spectra_intensities.index,  # not in ben Cs code
        "num_experiments": metadata["num_experiments"],
        "spectra_intensities": spectra_intensities.values.tolist(),
        "intensity_matrix": intensity_matrix,
        "y_range": y_range,
        "heatmap_y_range": heatmap_y_range,
        "first_spectrum_intensities": first_spectrum_intensities,
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "time_series_data": time_series_data,
        "file_num_index": file_num_index,
        "index_df": index_df,
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
    # Different range behaviour for if we are plotting by time or by experiment number (xrd is by exp number, uv-vis and nmr are by time)

    overall_min_time = min(time_series_time_range["min_y"], heatmap_time_range["min_y"])
    overall_max_time = max(time_series_time_range["max_y"], heatmap_time_range["max_y"])
    y_range = {"min_y": overall_min_time, "max_y": overall_max_time}
    intensity_min = np.min(plot_data["intensity_matrix"])
    intensity_max = np.max(plot_data["intensity_matrix"])
    shared_y_range = Range1d(start=y_range["min_y"], end=y_range["max_y"])

    shared_x_range = Range1d(
        start=min(plot_data["heatmap x_values"]), end=max(plot_data["heatmap x_values"])
    )

    intensity_range = Range1d(start=intensity_min, end=intensity_max)

    return {
        "shared_y_range": shared_y_range,
        "shared_x_range": shared_x_range,
        "intensity_range": intensity_range,
    }


def _create_heatmap_figure(
    plot_data: Dict[str, Any], ranges: Dict[str, Range1d], x_axis_label, time_series_y_axis_label
) -> figure:
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
    time_range = plot_data["y_range"]
    intensity_min = plot_data["intensity_min"]
    intensity_max = plot_data["intensity_max"]

    tools = "pan,wheel_zoom,box_zoom,reset,save"

    heatmap_figure = figure(
        x_axis_label=x_axis_label,
        y_axis_label=time_series_y_axis_label,
        x_range=ranges["shared_x_range"],
        y_range=ranges["shared_y_range"],
        height=400,
        tools=tools,
    )

    color_mapper = LinearColorMapper(palette="Viridis256", low=intensity_min, high=intensity_max)

    heatmap_figure.image(
        image=[intensity_matrix],
        x=min(heatmap_x_values),
        y=time_range["min_y"],
        dw=abs(max(heatmap_x_values) - min(heatmap_x_values)),
        dh=time_range["max_y"] - time_range["min_y"],
        color_mapper=color_mapper,
        level="image",
    )

    time_points = len(intensity_matrix)
    if time_points > 0:
        times = np.linspace(time_range["min_y"], time_range["max_y"], time_points)
        # experiment_numbers = np.arange(1, time_points + 1)
        experiment_numbers = plot_data["file_num_index"].flatten().tolist()
        heatmap_index_df = (
            plot_data["index_df"].reset_index().set_index("file_num").loc[experiment_numbers]
        )
        data = {
            "x": [(max(heatmap_x_values) + min(heatmap_x_values)) / 2] * time_points,
            "y": times,
            "width": [abs(max(heatmap_x_values) - min(heatmap_x_values))] * time_points,
            "height": [(time_range["max_y"] - time_range["min_y"]) / time_points] * time_points,
            "file_num": experiment_numbers,
        }

        for col in heatmap_index_df.columns:
            data[col] = heatmap_index_df[col].values
        source = ColumnDataSource(data=data)

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


def _create_top_line_figure(
    plot_data: Dict[str, Any], ranges: Dict[str, Range1d], line_y_axis_label
) -> figure:
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
        y_axis_label=line_y_axis_label,
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


def _create_echem_figure(
    plot_data: Dict[str, Any],
    ranges: Dict[str, Range1d],
    time_series_x_axis_label,
    time_series_y_axis_label,
) -> figure:
    """
    Create the electrochemical data figure component.

    Args:
        plot_data: Dictionary containing prepared plot data
        ranges: Dictionary of shared range objects

    Returns:
        figure: Configured Bokeh electrochemical figure
    """
    echem_data = plot_data["time_series_data"]

    tools = "pan,wheel_zoom,box_zoom,reset,save"

    echemplot_figure = figure(
        x_axis_label=time_series_x_axis_label,
        y_axis_label=time_series_y_axis_label,
        y_range=ranges["shared_y_range"],
        height=400,
        width=250,
        tools=tools,
    )

    echemplot_figure.xaxis.ticker.desired_num_ticks = 4

    if echem_data and "Voltage" in echem_data and "time" in echem_data:
        times = np.array(echem_data["time"])
        voltages = np.array(echem_data["Voltage"])

        time_range = plot_data["y_range"]
        time_span = time_range["max_y"] - time_range["min_y"]
        exp_count = plot_data["num_experiments"]

        exp_numbers = np.floor(((times - time_range["min_y"]) / time_span) * exp_count) + 1
        exp_numbers = np.clip(exp_numbers, 1, exp_count)

        echem_source = ColumnDataSource(
            data={
                "y": times,
                "x": voltages,
                "exp_num": exp_numbers,
                "time": times,
                "voltage": voltages,
            }
        )

        echemplot_figure.line(x="x", y="y", source=echem_source, color=COLORS[1])

        hover_tool = HoverTool(
            tooltips=[
                ("Exp. #", "@exp_num{0}"),
                ("Time (s)", "@y{0.00}"),
                ("Voltage (V)", "@x{0.000}"),
            ],
            mode="hline",
            point_policy="snap_to_data",
        )

        echemplot_figure.add_tools(hover_tool)

    elif echem_data and "x" in echem_data and "y" in echem_data:
        # Handle alternate pathway for echem data
        times = np.array(echem_data["y"])
        voltages = np.array(echem_data["x"])

        time_range = plot_data["y_range"]
        time_span = time_range["max_y"] - time_range["min_y"]
        exp_count = plot_data["num_experiments"]
        exp_numbers = np.floor(((times - time_range["min_y"]) / time_span) * exp_count) + 1
        exp_numbers = np.clip(exp_numbers, 1, exp_count)
        echem_source = ColumnDataSource(
            data={
                "y": times,
                "exp_num": exp_numbers,
                "file_num": echem_data["filenum"],
                "Temperature": voltages,
            }
        )

        echemplot_figure.line(
            x="Temperature",
            y="exp_num",
            source=echem_source,
        )

        hover_tool = HoverTool(
            tooltips=[
                ("Exp. #", "@y{0}"),
                ("File #", "@file_num{0}"),
                ("Temperature (C)", "@Temperature{0.000}"),
            ],
            mode="hline",
            point_policy="snap_to_data",
        )
        echemplot_figure.add_tools(hover_tool)
    else:
        raise ValueError("Echem data must contain 'Voltage' and 'time' or 'x' and 'y' keys.")

    plot_data["echem_source"] = echem_source
    return echemplot_figure


def _link_plots(
    heatmap_figure: figure,
    uvvisplot_figure: figure,
    echemplot_figure: figure,
    plot_data: Dict[str, Any],
    plotting_label_dict: dict,
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
        index_df_source = ColumnDataSource(plot_data["index_df"].reset_index())
        hover.callback = CustomJS(
            args=dict(
                line_source=line_source,
                spectra_intensities=spectra_intensities,
                ppm_values=ppm_values.tolist(),
                heatmap_source=heatmap_source,
                index_df_source=index_df_source,
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

                    const fileNum = heatmap_source.data.file_num[closestIndex];

                    const df_data = index_df_source.data;

                    // Find index in index_df_source where file_num matches
                    let dfIndex = -1;
                    for (let i = 0; i < df_data.file_num.length; i++) {
                        if (df_data.file_num[i] === fileNum) {
                            dfIndex = i;
                            break;
                        }
                    }

                    const data = line_source.data;
                    data['intensity'] = spectra_intensities[dfIndex];
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
                label_template=plotting_label_dict["label_source"]["label_template"],
                label_fields=plotting_label_dict["label_source"]["label_field_map"],
            ),
            code="""
                    console.log("Tap callback heatmap triggered");
                    const indices = cb_obj.indices;
                    if (indices.length === 0) return;

                    const index = indices[0];
                    const exp_num = heatmap_source.data.exp_num[index];
                    const exp_index = heatmap_source.data.index[index];
                    const values = { exp_num: exp_num };
                    console.log("Exp num:", exp_num, "Exp index:", exp_index);
                    for (const key in label_fields) {
                        if (key !== "exp_num") {
                            const field = label_fields[key];
                            const val = heatmap_source.data[field][index];
                            // Optional formatting
                            if (key === "time") {
                                values[key] = val.toFixed(0);
                            } else if (key === "voltage" || key === "temperature") {
                                values[key] = val.toFixed(3);
                            } else {
                                values[key] = val;
                            }
                        }
                    }

                    // Format the label string using the template
                    let label = label_template;
                    for (const key in values) {
                        label = label.replace(`{${key}}`, values[key]);
                    }

                    const existing_labels = clicked_spectra_source.data.label;
                    if (existing_labels.includes(label)) return;

                    const color_index = existing_labels.length % colors.length;

                    const new_xs = [...clicked_spectra_source.data['x']];
                    const new_ys = [...clicked_spectra_source.data.intensity];
                    const new_labels = [...clicked_spectra_source.data.label];
                    const new_colors = [...clicked_spectra_source.data.color];

                    new_xs.push(ppm_values);
                    new_ys.push(spectra_intensities[exp_index]);
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
       console.log("Remove line callback triggered");
        const indices = clicked_spectra_source.selected.indices;
        if (indices.length === 0) return;

        let data = clicked_spectra_source.data;

        // Sort indices in descending order to avoid index shifting issues
        const sortedIndices = indices.sort((a, b) => b - a);

        for (let i = 0; i < sortedIndices.length; i++) {
            let index = sortedIndices[i];
            data['x'].splice(index, 1);
            data['intensity'].splice(index, 1);
            data['label'].splice(index, 1);
            data['color'].splice(index, 1);
        }

        // Clear selection and emit change
        clicked_spectra_source.selected.indices = [];
        clicked_spectra_source.change.emit();
        console.log("Lines removed, remaining count:", data['x'].length);
        """,
    )

    clicked_spectra_source.selected.js_on_change("indices", remove_line_callback)

    # Adding logic for the echem plot to link with the heatmap
    echem_hover = next(
        (tool for tool in echemplot_figure.tools if isinstance(tool, HoverTool)), None
    )
    echem_source = plot_data.get("echem_source")

    if echem_hover:
        echem_hover.callback = CustomJS(
            args=dict(
                line_source=line_source,
                spectra_intensities=spectra_intensities,
                echem_source=echem_source,
                ppm_values=ppm_values.tolist(),
            ),
            code="""
                const geometry = cb_data['geometry'];

                    let closestIndex = 0;
                    let minDistance = Infinity;
                    for (let i = 0; i < echem_source.data.y.length; i++) {
                        const distance = Math.abs(echem_source.data.y[i] - geometry.y);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestIndex = i;
                        }
                    }
                    const exp_num = echem_source.data.exp_num[closestIndex];
                    const exp_index = exp_num - 1;

                    if (exp_index !== undefined && exp_index < spectra_intensities.length) {
                        const data = line_source.data;
                        data['intensity'] = spectra_intensities[exp_index];
                        line_source.change.emit();
                    } else {
                        console.warn("Index out of range or undefined:", exp_index);
                    }
                """,
        )

        leave_callback_echem = CustomJS(
            args=dict(line_source=line_source),
            code="""
                var data = line_source.data;
                data['intensity'] = [];  // Clear the intensity data
                line_source.change.emit();
            """,
        )

        echemplot_figure.js_on_event("mouseleave", leave_callback_echem)
    else:
        print("No HoverTool found in echemplot_figure. Cannot link hover callback.")
    if echem_source:
        clickcallback_echem = CustomJS(
            args=dict(
                echem_source=echem_source,
                clicked_spectra_source=clicked_spectra_source,
                spectra_intensities=spectra_intensities,
                ppm_values=ppm_values.tolist(),
                colors=COLORS,
                label_template=plotting_label_dict["label_source"]["label_template"],
                label_fields=plotting_label_dict["label_source"]["label_field_map"],
            ),
            code="""
                    console.log("Tap callback echem triggered");
                    const x = cb_obj.x;
                    const y = cb_obj.y;
                    if (x === undefined || y === undefined) {
                        console.warn("No x/y data available in tap event");
                        return;
                    }

                    let closestIndex = 0;
                    let minDistance = Infinity;
                    for (let i = 0; i < echem_source.data.y.length; i++) {
                        const distance = Math.abs(echem_source.data.y[i] - y);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestIndex = i;
                        }
                    }
                    const exp_num = echem_source.data.exp_num[closestIndex];
                    const exp_index = exp_num - 1;
                    console.log("Experiment number:", exp_num, "Index:", exp_index);
                    const values = { exp_num: exp_num };

                    for (const key in label_fields) {
                        if (key !== "exp_num") {
                            const field = label_fields[key];
                            const val = echem_source.data[field][closestIndex];
                            // Optional formatting
                            if (key === "time") {
                                values[key] = val.toFixed(0);
                            } else if (key === "voltage" || key === "temperature") {
                                values[key] = val.toFixed(3);
                            } else {
                                values[key] = val;
                            }
                        }
                    }

                    // Format the label string using the template
                    let label = label_template;
                    for (const key in values) {
                        label = label.replace(`{${key}}`, values[key]);
                    }

                    const existing_labels = clicked_spectra_source.data.label;
                    if (existing_labels.includes(label)) return;

                    const color_index = existing_labels.length % colors.length;

                    const new_xs = [...clicked_spectra_source.data['x']];
                    const new_ys = [...clicked_spectra_source.data.intensity];
                    const new_labels = [...clicked_spectra_source.data.label];
                    const new_colors = [...clicked_spectra_source.data.color];

                    new_xs.push(ppm_values);
                    new_ys.push(spectra_intensities[exp_index]);
                    new_labels.push(label);
                    new_colors.push(colors[color_index]);

                    clicked_spectra_source.data = {
                        'x': new_xs,
                        'intensity': new_ys,
                        'label': new_labels,
                        'color': new_colors
                    };

                    clicked_spectra_source.change.emit();
                    console.log("Clicked spectra source updated:", clicked_spectra_source.data);
        """,
        )

        # Attach the callback to tap events on the plot
        echemplot_figure.js_on_event("tap", clickcallback_echem)
