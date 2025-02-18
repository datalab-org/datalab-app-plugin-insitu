# <div align="center"><i>datalab insitu NMR</i></div>

<div align="center" style="padding-bottom: 5px">
<a href="https://demo.datalab-org.io"><img src="https://img.shields.io/badge/try_it_out!-public_demo_server-orange?logo=firefox"></a>
</div>

<div align="center">
<a href="https://join.slack.com/t/datalab-world/shared_invite/zt-2h58ev3pc-VV496~5je~QoT2TgFIwn4g"><img src="https://img.shields.io/badge/Slack-chat_with_us-yellow?logo=slack"></a>
</div>

A Python plugin for processing and visualizing in situ NMR data within [_datalab_](https://github.com/the-grey-group/datalab) instances. This plugin leverages the [_datalab_ Python API](https://github.com/datalab-org/datalab-api) to create interactive Bokeh plots of NMR data alongside electrochemical measurements.

The project was originally developed in and is currently deployed for the [Grey Group](https://www.ch.cam.ac.uk/group/grey/) in the Department of Chemistry at the University of Cambridge.

## Features

- Process both 1D and pseudo-2D NMR data from Bruker instruments
- Integration with electrochemical data for combined analysis
- Interactive visualization using Bokeh
- Flexible PPM range selection
- Support for both local files and datalab API access

![insitu NMR and Echem Bokeh Plot](docs/assets/datalab_plugin_bokeh_plot.png)

# Installation

We recommend you use [`uv`](https://astral.sh/uv) for managing virtual environments and Python versions.

Once you have `uv` installed, you can clone this repository and install the package in a fresh virtual environment with:

```
git clone git@github.com:datalab-org/datalab_app_plugin_nmr_insitu
cd datalab_app_plugin_nmr_insitu
uv sync --all-extras --dev
```

## Development installation

You can activate `pre-commit` in your local repository with `uv run pre-commit install`.

## Configuration

### For Datalab API Usage

When using the plugin with a datalab instance, you need to set up your API key:

1. Create a .env file in your project root
1. Add your Datalab API key:

```Shell
   DATALAB_API_KEY=your_api_key_here
```

## Usage

The plugin offers two main processing functions: process_local_data for local files and process_datalab_data for datalab integration.

### Processing Local Data

```python
from datalab_app_plugin_nmr_insitu import process_local_data

# Process local NMR data
result = process_local_data(
    folder_name="path/to/your/data.zip",        # Path to zip file or folder
    nmr_folder_name="nmr_data",                 # Folder containing NMR experiments
    echem_folder_name="echem_data",             # Optional: folder with electrochemical data
    ppm1=240,                                   # Lower PPM range limit
    ppm2=280,                                   # Upper PPM range limit
    start_at=1,                                 # Optional: starting experiment number
    exclude_exp=[]                              # Optional: experiments to exclude
)
```

### Using with Datalab API

```python
from datalab_app_plugin_nmr_insitu import process_datalab_data

# Process NMR data from datalab
result = process_datalab_data(
    api_url="https://your-datalab-instance.com",
    item_id="your-item-id",
    folder_name="your-folder",
    nmr_folder_name="nmr-data",
    echem_folder_name="echem-data",
    ppm1=240,
    ppm2=280,
    start_at=1,                                 # Optional: starting experiment number
    exclude_exp=[]                              # Optional: experiments to exclude
)
```

## API Reference

### process_local_data

```python
def process_local_data(
    folder_name: str,
    nmr_folder_name: str,
    echem_folder_name: str,
    ppm1: float,
    ppm2: float,
    start_at: int = 1,
    exclude_exp: Optional[List[int]] = None,
) -> Dict
```

Process NMR spectroscopy data from local files.

**Parameters:**

- `folder_name`: Path to zip file or folder containing the data
- `nmr_folder_name`: Folder containing NMR experiments
- `echem_folder_name`: Folder containing electrochemical data (optional)
- `ppm1`: Lower PPM range limit
- `ppm2`: Upper PPM range limit
- `start_at`: Starting experiment number (default: 1)
- `exclude_exp`: List of experiment numbers to exclude (default: None)

### process_datalab_data

```Python
def process_datalab_data(
    api_url: str,
    item_id: str,
    folder_name: str,
    nmr_folder_name: str,
    echem_folder_name: str,
    ppm1: float,
    ppm2: float,
    start_at: int = 1,
    exclude_exp: Optional[List[int]] = None,
) -> Dict
```

Process NMR spectroscopy data from Datalab API.

**Parameters:**

- `api_url`: URL of the Datalab API
- `item_id`: ID of the item to process
- `folder_name`: Base folder name in datalab
- `nmr_folder_name`: Folder containing NMR experiments
- `echem_folder_name`: Folder containing electrochemical data (optional)
- `ppm1`: Lower PPM range limit
- `ppm2`: Upper PPM range limit
- `start_at`: Starting experiment number (default: 1)
- `exclude_exp`: List of experiment numbers to exclude (default: None)

**Returns format:**

Both functions return a dictionary with the following structure:

```python
    result = {
        "metadata": {
            "ppm_range": {
                "start": float,               # Minimum PPM value
                "end": float                  # Maximum PPM value
            },
            "time_range": {
                "start": float,               # Start time in hours
                "end": float                  # End time in hours
            },
            "num_experiments": int,           # Total number of experiments
        },
        "nmr_spectra": {
            "ppm": List[float],               # PPM values
            "spectra": [
                {
                    "time": float,            # Time point in hours
                    "intensity": List[float]  # Intensity values
                }
            ]
        },
        "integrated_data": {
            "intensity": List[float],          # Integrated intensity
            "norm_intensity": List[float]      # Normalized intensity
            "time": float,                     # Time point in hours
        },
        "echem": {                             # Only present if echem_folder_name is provided
            "Voltage": List[float],            # Voltage measurements
            "time": List[float]                # Time points in hours
        }
    }
```

## Data Structure Requirements

Your data should be organized as follows:

```Shell
data_folder.zip/
├── nmr_folder/
│   ├── 1/
│   │   ├── acqus
│   │   └── pdata/
│   │       └── 1/
│   │           └── ascii-spec.txt
│   ├── 2/
│   │   └── ...
│   └── ...
└── echem_folder/  (optional)
    └── echem/
        └── GCPL_*.MPR
```

## License

This project is released under the conditions of the MIT license. Please see [LICENSE](https://github.com/datalab-org/datalab_app_plugin_nmr_insitu/blob/main/LICENSE) for the full text of the license.

## Contact

For questions and support, please [open an issue](https://github.com/datalab-org/datalab_app_plugin_nmr_insitu/issues) on the GitHub repository or join the [public datalab Slack workspace](https://join.slack.com/t/datalab-world/shared_invite/zt-2h58ev3pc-VV496~5je~QoT2TgFIwn4g).
