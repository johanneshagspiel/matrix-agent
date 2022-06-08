<img src=img/matrix_agent_logo.JPG alt="MATRIX Agent Logo" width="265" height="136">

--------------------------------------------------------------------------------
[![MIT-License](https://img.shields.io/github/license/johanneshagspiel/matrix-agent)](LICENSE)
[![Top Language](https://img.shields.io/github/languages/top/johanneshagspiel/matrix-agent)](https://github.com/johanneshagspiel/matrix-agent)
[![Latest Release](https://img.shields.io/github/v/release/johanneshagspiel/matrix-agent)](https://github.com/johanneshagspiel/matrix-agent/releases/)

# MATRIX Agent

"MATRIX Agent" is a collaborative agents that can autonomously cooperate with other agents to complete tasks in a joint-activity environment known as [Blocks World for Teams](https://www.matrx-software.com/docs/tutorials/building-a-block-world/block-worlds-for-teams/). 

## Features

This "MATRIX Agent":
- has three distinct kinds of behavioral modes depending on the environment:
  - alone mode: it enters this setting it does not receive any identifiable message

## Tools

| Purpose                | Name                                                           |
|------------------------|----------------------------------------------------------------|
| Programming language   | [Python](https://www.python.org/)                              |
| Dependency manager     | [Anaconda](https://www.anaconda.com/products/distribution)     |
| Version control system | [Git](https://git-scm.com/)                                    |
| Application Bundler    | [MATRIX](http://docs.matrx-software.com/en/master/index.html/) |


## Installation Process

A precompiled executable can be found with the [latest release]((https://github.com/johanneshagspiel/gcode-modifier/releases/)). 

If you want to build the application yourself, it is assumed that you have installed [Python](https://www.python.org/downloads/windows/) and that your operating system is Windows.

Open this repository in the terminal of your choice. In case `pip` has not been installed, do so via:

    py -m ensurepip --upgrade

Install `pyinstaller` with this command:

    pip install -U pyinstaller

Now, create the executable via:

    cd src | pyinstaller paste_printer/main.py --distpath "pyinstaller/dist" --workpath "pyinstaller/build"  --noconsole --add-data "paste_printer/resources/icons/*;paste_printer/resources/icons" --add-data "paste_printer/resources/gcode/0.6/*;paste_printer/resources/gcode/0.6" --add-data "paste_printer/resources/gcode/0.8/*;paste_printer/resources/gcode/0.8" --add-data "paste_printer/resources/gcode/1.5/*;paste_printer/resources/gcode/1.5" --add-data "paste_printer/resources/fonts/*;paste_printer/resources/fonts" --add-data "paste_printer/resources/settings/*;paste_printer/resources/settings"

This should have created a new folder in `src` called `pyinstaller`. The executable can be then be found at `src/pyinstaller/dist/main.exe`

If you want to import this project and resolve all the dependencies associated with it, it is assumed that you have already installed [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Re-create the original `GCode-Modifier` environment from the `environment.yml` file with this command:

	conda env create -f environment.yml

Activate the new environment:
 
	conda activate GCode-Modifier

Lastly, check that the new environment was installed correctly:
	
	conda env list

## Licence

The "MATRIX Agent" is published under the MIT licence, which can be found in the [LICENSE](LICENSE) file. For this repository, the terms laid out there shall not apply to any individual that is currently enrolled at a higher education institution as a student. Those individuals shall not interact with any other part of this repository besides this README in any way by, for example cloning it or looking at its source code or have someone else interact with this repository in any way.

## References

The base image for the logo was taken from the [official MATRIX website](https://matrx-software.com/wp-content/uploads/2020/02/matrx_logo.svg). 