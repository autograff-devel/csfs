# Curvilinear Shape Features 

## Installation and dependencies
Running the scripts in this projec requires also downloading or (preferibly) cloning:
- <https://github.com/colormotor/autograff> Various support Python utilites 
- <https://github.com/colormotor/csf_analysis> This project, curvilinear shape features
  
To simplify setup, clone the repository with a directory hierarchy of this type:
```
|── python/
|── └── projects/
|── |── └── csf_analysis/
|── |── |── └── ...
|── |── modules/
|── |── └── autograff/
|── |── |── └── ...
```
Where the leafs are this (csf_analysis) and the autograff repositories linked above.
Note that the parent `python/` directory is not required, but suggested for use in conjunction with future (non-python) projects that would be stored at the same level as the `python/` directory.

### Dependencies
The simplest way to install all dependencies is the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager. [Anaconda](https://docs.anaconda.com/anaconda/install/) can also be used, by replacing `miniconda` with `anaconda` where it occurs in the following instructions.

#### Step 1 - install miniconda
Download the 64 bit, Python 3.7 version of the Minconda installer [here](https://docs.conda.io/en/latest/miniconda.html). Run the installer, making sure that the installation is made locally, in the user home directory. There should be an option to do so during the installation procedure, and this should avoid the requirement of running dependency installation as a superuser. Once installed, ,miniconda should be installed in the `~/miniconda3/` directory or in `~/opt/miniconda3` (on Mac), and Linux should be something similar. 

#### Step 2 - install the dependencies
Dependencies are mainly installed from the terminal with the `conda` command. If a terminal session was already active during installation, the session must be restarted for the command to be found.

The following installs all required dependencies for all the project
```
conda install -c conda-forge numpy
conda install -c conda-forge opencv
conda install -c conda-forge scipy
conda install -c conda-forge matplotlib
conda install -c conda-forge pillow
pip install svgpathtools
conda install -c conda-forge networkx
conda install -c conda-forge fonttools
conda install -c conda-forge pyclipper
conda install -c conda-forge shapely
```

### Installation with a custom environment (advanced)
The package managers [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), can be used to isolate the project dependencies by creating a dedicated *environment*. This may be useful to avoid dependency conflicts if an installation already exists.

The following instructions are specific to miniconda, but as usual they hold also for Anaconda by replacing `minconda3` with `anaconda3` where a directory is specified. It is assumed that the installation is made locally, in the user home directory. With conda installed, from the shell, first create an environment. Let's call it `py36autograff`:
```
conda create --name py36autograff python=3.6.5
```
Now activate the environment with
```
conda activate py36autograff
```
This will result in all new dependecies being installed in `~/miniconda3/envs/py36autograff`. Note that the environment will be only active for the shell session where `conda activate` has been called. The environment can be deactivated with `conda deactivate`. Once the environment is active, the dependencies can be installed with the instructions in the previous section.

## Running
The scripts are located in two directories, `scripts` (perform batch computations) and `figures` (generates specific figures). To run a script either call it from the terminal with python, making sure to navigate to its directory beforehand. Otherwise a script can be opened in an IDE like Spyder, and run by either executing cells (delimited by `#%%` blocks) or running the entire script. When using an IDE, make sure the working directory is set to the script's directory.

### `figures` scripts
These scripts don't require input paramters and can be run from the terminal, e.g.:
```
python figure_initial_example.py 
```
Will generate an initial figure from the relevant thesis chapter.
The output will be saved in the `/output/csfs` directory, which will be created automatically.

### `scripts` scripts
These scripts can be used to process multiple files at once.
Currently contains:
- `plot_casa.py`: Plots symmetry axes and Curvilinear Augmented Symmetry Axis (CASA) for a set of shapes

The file [global_args.py](https://github.com/colormotor/csf_analysis/scripts/global_args.py) defines default paramters that are common to all scrips (e.g. input/output directories for segmentation data). 

#### With JSON configuration
The suggested method to run the scripts is using a JSON configuration file as an input, e.g. with the provided settings (this defaults to the file `fonts_default.json`):
```
python plot_casa.py -i fonts_default.json
```
Will run segmentation and stroke reconstruction for some characters in a font file contained in the `/data/fonts` directory.
Calling 
```
python plot_casa.py -i svg_default.json
```
will execute the same procedure for svgs located in the `/data/svg` directory.

By default, the output data and result-images will be saved in the `/output/default` directory. 

The entries defined in the configuration file determine the input and output directories, the input data type (SVG or TTF) and other optional paramters if desired.  

The main parameters in a JSON configuration  are:

- `input_path` (string): <br>
Determines a directory where either TTF or SVG files are located (relative to the script)
- `params` (dict): <br>
Contains additional parameters, e.g:
  - `type` (string, default `"ttf"`): Defines the type of the input, either `"ttf"` or `"svg"`.
  - `output` (string, optional): Defines the directory where output files and image are generated (relative to script)
  - Other optional script paramters are defined in [scripts/global_args.py](https://github.com/colormotor/csf_analysis/blob/master/scripts/global_args.py) 
  - Optional segmentation-specific paramters are defined in [csfs/config.py](https://github.com/colormotor/csf_analysis/blob/master/csfs/config.py). 
  
- `items` (array, default `[]`): <br>
optionally contains a list of inputs to process. For TTF files, this is the actual font name, which may differ from the filename. For SVG files, the name is the filename without the extension (as found in the directory specified with `input_path`). For fonts, it is possible to define an input with a subset of characters by using an array notation like `["Arial Bold", "ABC"]`.
- `chars` (string, default `""`): <br>
If non empty, it forces the segmentation to process only characters in the string.
- `chars_map`: <br>
Can be used to map a font name to a given character set. E.g. adding an entry `"Arial":"XYZ"` will result in only the characters `XYZ` being processed for any font with a name containing the string `Arial`.

An example setup could be:
```
{
"input_path": "./../data/fonts",
"params":{
    "output":"~/data/segmentation_data",
    "type":"ttf",
    "save_figures":true,
    "vma_thresh":2.
},
"items":["Georgia", "Arial Bold"
], 
"chars":  "K"
}
```
This will process the specified fonts (if present in `./../data/fonts`), and output to the the non-default directory `~/data/segmentation_data` and will use a Voronoi medial axis regularization threhold for crossings of $2.$ (see [csfs/config.py](https://github.com/colormotor/csf_analysis/blob/master/csfs/config.py) for the available paramters and brief descriptions of their purpose).

#### Without JSON configuration
A script can be run also without a JSON configuration, by specifying a directory and other paramters as command-line arguments. E.g. 
```
python plot_casa.py -i ../data/svg -o ~/svg_casa --type="svg"
```
Will run the junction labelling script on SVG files contained in the  `../data/svg` directory and output results to ` ~/svg_casa`.

#### Running interactively
The scripts can also be run interactively in a IPython/Jupyter session, for example in an IDE such as [Spyder](https://www.spyder-ide.org) or [VSCode](https://code.visualstudio.com) with the appropriate Jupyter extensions. This can be convient for test and debug purposes. The scripts contain "code blocks" (delimited with the standard `#%%` comments) which can be executed in an interactive session. When running a script interactively, make sure that the current interpreter path is set to the script directory. 

When running interactively, it is convenient to define a JSON file with the setting `"save_figures":false`, which should produce inline image outputs.

## `csfs` module file structure
The module heavily depends on the [autograff](https://github.com/colormotor/autograff) for alot of its functionality (geometry processing, utilities).

### casa.py
Contains functions for medial axis generation/processing and CSF analysis (absolute maxima only) and CASA computation.
Medial axes are stored as networkx graphs, with additional properties contained in the graphs dictionary.
As an example for a medial axis `MA` the corresponding disks are stored as `MA.graph['disks']`.

### path_sym.py
Curvilinear shape features code


### voronoi_skeleton.py
Voronoi skeleton implementation

### config.py
Groups all settable paramters for used by module

### common.py
Contains plotting and data processing utilities.
