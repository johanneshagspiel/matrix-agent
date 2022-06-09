<img src=img/matrix_agent_logo.JPG alt="MATRIX Agent Logo" width="265" height="136">

--------------------------------------------------------------------------------
[![MIT-License](https://img.shields.io/github/license/johanneshagspiel/matrix-agent)](LICENSE)
[![Top Language](https://img.shields.io/github/languages/top/johanneshagspiel/matrix-agent)](https://github.com/johanneshagspiel/matrix-agent)
[![Latest Release](https://img.shields.io/github/v/release/johanneshagspiel/matrix-agent)](https://github.com/johanneshagspiel/matrix-agent/releases/)

# MATRX Agent

"MATRX Agent" is a collaborative agent that can autonomously cooperate with other agents to complete tasks in a joint-activity environment known as [Blocks World for Teams (BW4T)](https://www.matrx-software.com/docs/tutorials/building-a-block-world/block-worlds-for-teams/). A world in BW4T consists of multiple rooms that contain blocks of different shapes and colors which have to be dropped off at a particular location in a particular order. An agent in BW4T can be shape- or colorblind which necessitates the agents communicating amongst themselves the information that they have obtained.

## Features

This "MATRX Agent":

- has three distinct kinds of behavioral modes depending on the environment that all have implemented the entire behavior necessary for an arbitrary number of fully as well as partially capable agents to finish one whole iteration of the MATRX world such as:
  - solitary mode: occurs when the agent does not receive any identifiable message from the other agents in the world
  - cluster mode: occurs when the agent receives more cluster_mode protocol messages than group_mode protocol messages. A cluster of agents includes agents made by other individuals that communicate via the dedicated cluster message system. 
  - group mode: occurs when the agent receives more group_mode protocol messages than cluster_mode protocol messages. A group of agents includes only other "MATRIX Agents" that communicate via the dedicated group message system.
- can operate in even the most complex of environments that exist in BW4T such as situations where the agent is alone and has to pick up more than three items. The agent is aware of its inventory limit and thus drops off blocks to the left of the appropriate drop location.
- can explore rooms of an arbitrary size as it uses the actual size of a room to determine the appropriate exploration pattern   

## Tools

| Purpose                | Name                                                           |
|------------------------|----------------------------------------------------------------|
| Programming language   | [Python](https://www.python.org/)                              |
| Dependency manager     | [Virtualenv](https://virtualenv.pypa.io/en/latest/index.html)     |
| Version control system | [Git](https://git-scm.com/)                                    |
| Agent framework    | [MATRX](http://docs.matrx-software.com/en/master/index.html/) |


## Installation Process

If you want to import this project and resolve all the dependencies associated with it, it is assumed that you have already installed a dependency manager of your choice like [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html), [Python](https://www.python.org/downloads/windows/) and that your operating system is Windows. It is strongly suggested that you first create a [virtual environment](https://virtualenv.pypa.io/en/latest/user_guide.html#introduction) with this dependency manager before you proceed.

Then, you can install all the dependencies with this command:

	pip install -r requirements.txt

## Licence

The "MATRX Agent" is published under the MIT licence, which can be found in the [LICENSE](LICENSE) file. For this repository, the terms laid out there shall not apply to any individual that is currently enrolled at a higher education institution as a student. Those individuals shall not interact with any other part of this repository besides this README in any way by, for example cloning it or looking at its source code or have someone else interact with this repository in any way.

## References

The base image for the logo was taken from the [official MATRX website](https://matrx-software.com/wp-content/uploads/2020/02/matrx_logo.svg). 