# <div align="center"><i>datalab insitu NMR</i></div>

<div align="center" style="padding-bottom: 5px">
<a href="https://demo.datalab-org.io"><img src="https://img.shields.io/badge/try_it_out!-public_demo_server-orange?logo=firefox"></a>
</div>
<div align="center">
<a href="https://github.com/datalab-org/datalab-app-plugin-insitu/releases"><img src="https://badgen.net/github/release/datalab-org/datalab-app-plugin-insitu?icon=github&color=blue"></a>
<a href="https://github.com/datalab-org/datalab-app-plugin-insitu"><img src="https://badgen.net/github/license/datalab-org/datalab-app-plugin-insitu?icon=license&color=purple"></a>
</div>
<div align="center">
<a href="https://datalab-app-plugin-insitu.readthedocs.io/en/latest/?badge=latest"><img src="https://img.shields.io/readthedocs/datalab-app-plugin-insitu?logo=readthedocs"></a>
</div>
<div align="center">
<a href="https://join.slack.com/t/datalab-world/shared_invite/zt-2h58ev3pc-VV496~5je~QoT2TgFIwn4g"><img src="https://img.shields.io/badge/Slack-chat_with_us-yellow?logo=slack"></a>
</div>


A Python plugin for processing and visualizing *in situ* NMR data within [_datalab_](https://github.com/datalab-org/datalab) instances.

The project was originally developed in the [Grey Group](https://www.ch.cam.ac.uk/group/grey/) in the Department of Chemistry at the University of Cambridge.

> [!WARNING]  
> This plugin is still under development and may struggle to process larger datasets within a single *datalab* request, in which case you may wish to run the plugin locally.

## Features

- Process both 1D and pseudo-2D NMR data from Bruker instruments
- Integration with electrochemical data formats for combined analysis using [navani](https://github.com/be-smith/navani)
- Interactive visualization using Bokeh
- Flexible PPM range selection
- Support for both local files and *datalab* API access, as well as running as a plugin directly on a *datalab* instance

<div align="center">
   <img src="./docs/assets/datalab_plugin_bokeh_plot.png" width=600rem>
</div>

# Installation

The `datalab-app-plugin-insitu` package is currently a "core plugin" of *datalab*, so it will be installed on *datalab* instances by default.

## Development installation

We recommend you use [`uv`](https://astral.sh/uv) for managing virtual environments and Python versions.

Once you have `uv` installed, you can clone this repository and install the package in a fresh virtual environment with:

```
git clone git@github.com:datalab-org/datalab-app-plugin-insitu
cd datalab-app-plugin-insitu
uv sync --all-extras --dev
```

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

Full reference API documentation can be found on [ReadTheDocs](https://datalab-app-plugin-insitu.readthedocs.io/).

### Data Structure Requirements

Your data should be organized as follows:

```Shell
data_folder.zip/
├── <nmr_folder>/
│   ├── 1/
│   │   ├── acqus
│   │   └── pdata/
│   │       └── 1/
│   │           └── ascii-spec.txt
│   ├── 2/
│   │   └── ...
│   └── ...
└── <echem_folder>/  (optional)
    └── *.MPR
```

with the `<nmr_folder>` and `<echem_folder>` names specified at runtime.

## License

This project is released under the conditions of the MIT license. Please see [LICENSE](https://github.com/datalab-org/datalab-app-plugin-insitu/blob/main/LICENSE) for the full text of the license.

## Contact

For questions and support, please [open an issue](https://github.com/datalab-org/datalab-app-plugin-insitu/issues) on the GitHub repository or join the [public datalab Slack workspace](https://join.slack.com/t/datalab-world/shared_invite/zt-2h58ev3pc-VV496~5je~QoT2TgFIwn4g).
