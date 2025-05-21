from typing import Any, Dict, Optional

import numpy as np
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


def create_linked_insitu_plots(plot_data, ppm_range, link_plots: bool = False):
    shared_ranges = _create_shared_ranges(plot_data, ppm_range=ppm_range)
    heatmap_figure = _create_heatmap_figure(plot_data, shared_ranges)
    nmrplot_figure = _create_nmr_line_figure(plot_data, shared_ranges)
    echemplot_figure = _create_echem_figure(plot_data, shared_ranges)

    heatmap_figure.js_on_event(
        DoubleTap, CustomJS(args=dict(p=heatmap_figure), code="p.reset.emit()")
    )
    nmrplot_figure.js_on_event(
        DoubleTap, CustomJS(args=dict(p=nmrplot_figure), code="p.reset.emit()")
    )
    echemplot_figure.js_on_event(
        DoubleTap, CustomJS(args=dict(p=echemplot_figure), code="p.reset.emit()")
    )

    if link_plots:
        _link_plots(heatmap_figure, nmrplot_figure, echemplot_figure, plot_data)

    grid = [[None, nmrplot_figure], [echemplot_figure, heatmap_figure]]
    gp = gridplot(grid, merge_tools=True)

    return gp


def prepare_plot_data(nmr_data, echem_data, metadata) -> Optional[Dict[str, Any]]:
    """
    Extract and prepare data for plotting.
    Returns:
        Optional[Dict[str, Any]]: Dictionary containing prepared plot data,
                                  or None if data extraction fails.
    """
    ppm_values = np.array(nmr_data.get("ppm", []))
    if len(ppm_values) == 0:
        raise ValueError("No PPM values found in NMR data")

    spectra = nmr_data.get("spectra", [])
    if not spectra:
        raise ValueError("No spectra found in NMR data")

    try:
        spectra_intensities = [np.array(spectrum["intensity"]).tolist() for spectrum in spectra]

        intensity_matrix = np.array([np.array(spectrum["intensity"]) for spectrum in spectra])

    except Exception as e:
        raise ValueError(f"Error processing spectrum intensities: {e}")

    time_range = metadata["time_range"]
    first_spectrum_intensities = np.array(spectra[0]["intensity"])

    intensity_min = np.min(intensity_matrix)
    intensity_max = np.max(intensity_matrix)

    return {
        "ppm_values": ppm_values,
        "spectra": spectra,
        "spectra_intensities": spectra_intensities,
        "intensity_matrix": intensity_matrix,
        "time_range": time_range,
        "first_spectrum_intensities": first_spectrum_intensities,
        "intensity_min": intensity_min,
        "intensity_max": intensity_max,
        "echem_data": echem_data,
    }


def _create_shared_ranges(
    plot_data: Dict[str, Any], ppm_range: tuple[float, float]
) -> Dict[str, Range1d]:
    """
    Create shared range objects for linking multiple plots.
    Args:
        plot_data: Dictionary containing prepared plot data
    Returns:
        Dict[str, Range1d]: Dictionary of shared range objects
    """
    ppm_values = plot_data["ppm_values"]
    time_range = plot_data["time_range"]
    intensity_min = np.min(plot_data["intensity_matrix"])
    intensity_max = np.max(plot_data["intensity_matrix"])

    shared_y_range = Range1d(start=time_range["start"], end=time_range["end"])

    ppm1, ppm2 = ppm_range

    ppm_min = min(ppm_values)
    ppm_max = max(ppm_values)

    ppm1 = max(min(ppm1, ppm_max), ppm_min)
    ppm2 = max(min(ppm2, ppm_max), ppm_min)

    shared_x_range = Range1d(start=max(ppm1, ppm2), end=min(ppm1, ppm2))

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
    ppm_values = plot_data["ppm_values"]
    intensity_matrix = plot_data["intensity_matrix"]
    time_range = plot_data["time_range"]
    intensity_min = plot_data["intensity_min"]
    intensity_max = plot_data["intensity_max"]

    tools = "pan,wheel_zoom,box_zoom,reset,save"

    heatmap_figure = figure(
        x_axis_label="δ (ppm)",
        y_axis_label="t (h)",
        x_range=ranges["shared_x_range"],
        y_range=ranges["shared_y_range"],
        height=400,
        tools=tools,
    )

    color_mapper = LinearColorMapper(palette="Viridis256", low=intensity_min, high=intensity_max)

    heatmap_figure.image(
        image=[intensity_matrix],
        x=max(ppm_values),
        y=time_range["start"],
        dw=abs(max(ppm_values) - min(ppm_values)),
        dh=time_range["end"] - time_range["start"],
        color_mapper=color_mapper,
        level="image",
    )

    time_points = len(intensity_matrix)
    if time_points > 0:
        times = np.linspace(time_range["start"], time_range["end"], time_points)
        experiment_numbers = np.arange(1, time_points + 1)

        source = ColumnDataSource(
            data={
                "x": [(max(ppm_values) + min(ppm_values)) / 2] * time_points,
                "y": times,
                "width": [abs(max(ppm_values) - min(ppm_values))] * time_points,
                "height": [(time_range["end"] - time_range["start"]) / time_points] * time_points,
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
            tooltips=[("Exp.", "@exp_num")],
            mode="mouse",
            point_policy="follow_mouse",
        )

        heatmap_figure.add_tools(hover_tool)

        plot_data["heatmap_source"] = source

    heatmap_figure.grid.grid_line_width = 0
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12)
    heatmap_figure.add_layout(color_bar, "right")

    return heatmap_figure


def _create_nmr_line_figure(plot_data: Dict[str, Any], ranges: Dict[str, Range1d]) -> figure:
    """
    Create the NMR line plot figure component.
    Args:
        plot_data: Dictionary containing prepared plot data
        ranges: Dictionary of shared range objects
    Returns:
        figure: Configured Bokeh line figure with data source
    """
    ppm_values = plot_data["ppm_values"]
    first_spectrum_intensities = plot_data["first_spectrum_intensities"]

    tools = "pan,wheel_zoom,box_zoom,reset,save"

    ppm_list = ppm_values.tolist() if isinstance(ppm_values, np.ndarray) else ppm_values
    intensity_list = (
        first_spectrum_intensities.tolist()
        if isinstance(first_spectrum_intensities, np.ndarray)
        else first_spectrum_intensities
    )

    line_source = ColumnDataSource(
        data={
            "δ (ppm)": ppm_list,
            "intensity": intensity_list,
        }
    )

    clicked_spectra_source = ColumnDataSource(
        data={"δ (ppm)": [], "intensity": [], "exp_index": [], "color": []}
    )

    nmrplot_figure = figure(
        y_axis_label="intensity",
        aspect_ratio=2,
        x_range=ranges["shared_x_range"],
        y_range=ranges["intensity_range"],
        tools=tools,
    )

    nmrplot_figure.line(
        x="δ (ppm)",
        y="intensity",
        source=line_source,
        line_width=1,
        color="blue",
        legend_label="Reference",
    )

    nmrplot_figure.multi_line(
        xs="δ (ppm)",
        ys="intensity",
        source=clicked_spectra_source,
        line_color="color",
        line_width=1,
        legend_field="exp_index",
    )

    nmrplot_figure.legend.click_policy = "hide"
    nmrplot_figure.legend.location = "top_right"

    plot_data["line_source"] = line_source
    plot_data["clicked_spectra_source"] = clicked_spectra_source
    return nmrplot_figure


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
        x_axis_label="voltage (V)",
        y_axis_label="t (h)",
        y_range=ranges["shared_y_range"],
        height=400,
        width=250,
        tools=tools,
    )

    echemplot_figure.xaxis.ticket.desired_num_ticks = 4

    if echem_data and "Voltage" in echem_data and "time" in echem_data:
        times = np.array(echem_data["time"])
        voltages = np.array(echem_data["Voltage"])

        time_range = plot_data["time_range"]
        time_span = time_range["end"] - time_range["start"]
        exp_count = len(plot_data["spectra"])

        exp_numbers = np.floor(((times - time_range["start"]) / time_span) * exp_count) + 1
        exp_numbers = np.clip(exp_numbers, 1, exp_count)

        echem_source = ColumnDataSource(
            data={"time": times, "voltage": voltages, "exp_num": exp_numbers}
        )

        echemplot_figure.line(
            x="voltage",
            y="time",
            source=echem_source,
        )

        hover_tool = HoverTool(
            tooltips=[
                ("Exp.", "@exp_num{0}"),
                ("Time (h)", "@time{0.00}"),
                ("Voltage (V)", "@voltage{0.000}"),
            ],
            mode="mouse",
            point_policy="follow_mouse",
        )

        echemplot_figure.add_tools(hover_tool)

    return echemplot_figure


def _link_plots(
    heatmap_figure: figure,
    nmrplot_figure: figure,
    echemplot_figure: figure,
    plot_data: Dict[str, Any],
) -> None:
    """
    Link the plots together with interactive tools and callbacks.
    Args:
        heatmap_figure: The heatmap figure component
        nmrplot_figure: The NMR line plot figure component
        echemplot_figure: The electrochemical figure component
        plot_data: Dictionary containing prepared plot data
    """
    line_source = plot_data["line_source"]
    clicked_spectra_source = plot_data["clicked_spectra_source"]
    spectra_intensities = plot_data["spectra_intensities"]
    ppm_values = plot_data["ppm_values"]
    intensity_matrix = plot_data["intensity_matrix"]
    heatmap_source = plot_data.get("heatmap_source")

    colors = [
        "red",
        "green",
        "orange",
        "purple",
        "brown",
        "darkblue",
        "teal",
        "magenta",
        "olive",
        "navy",
    ]

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
                    data['intensity'] = spectra_intensities[index];
                    line_source.change.emit();
                """,
        )

    if heatmap_source:
        tap_tool = TapTool()
        heatmap_figure.add_tools(tap_tool)
        tap_callback = CustomJS(
            args=dict(
                heatmap_source=heatmap_source,
                clicked_spectra_source=clicked_spectra_source,
                spectra_intensities=spectra_intensities,
                ppm_values=ppm_values.tolist(),
                colors=colors,
            ),
            code="""
                    const indices = cb_obj.indices;
                    if (indices.length === 0) return;
                    const index = indices[0];
                    const exp_num = heatmap_source.data.exp_num[index];
                    const existing_indices = clicked_spectra_source.data.exp_index;
                    if (existing_indices.includes(exp_num)) return;
                    const color_index = existing_indices.length % colors.length;
                    const new_xs = [...clicked_spectra_source.data['δ (ppm)']];
                    const new_ys = [...clicked_spectra_source.data.intensity];
                    const new_indices = [...clicked_spectra_source.data.exp_index];
                    const new_colors = [...clicked_spectra_source.data.color];
                    new_xs.push(ppm_values);
                    new_ys.push(spectra_intensities[index]);
                    new_indices.push(exp_num);
                    new_colors.push(colors[color_index]);
                    clicked_spectra_source.data = {
                        'δ (ppm)': new_xs,
                        'intensity': new_ys,
                        'exp_index': new_indices,
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

    line_y_range = nmrplot_figure.y_range
    line_y_range.js_link("start", heatmap_figure.select_one(LinearColorMapper), "low")
    line_y_range.js_link("end", heatmap_figure.select_one(LinearColorMapper), "high")

    tap_tool = TapTool()

    nmrplot_figure.add_tools(tap_tool)

    remove_line_callback = CustomJS(
        args=dict(clicked_spectra_source=clicked_spectra_source),
        code="""
        const indices = clicked_spectra_source.selected.indices;
        if (indices.length === 0) return;
        let data = clicked_spectra_source.data;
        for (let i = indices.length - 1; i >= 0; i--) {
            let index = indices[i];
            data['δ (ppm)'].splice(index, 1);
            data['intensity'].splice(index, 1);
            data['exp_index'].splice(index, 1);
            data['color'].splice(index, 1);
        }
        clicked_spectra_source.change.emit();
    """,
    )

    clicked_spectra_source.selected.js_on_change("indices", remove_line_callback)
