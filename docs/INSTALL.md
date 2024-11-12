## Installation

We used PyTorch 1.13.1 on [Ubuntu 22.04 LTS](https://releases.ubuntu.com/22.04/) with [Anaconda](https://www.anaconda.com/download) Python 3.7.

1. [Optional but recommended] Create a new Conda environment. 

    ~~~
    conda create --name glo-in-one_v2 python=3.7
    ~~~
    
    And activate the environment.
    
    ~~~
    conda activate glo-in-one_v2
    ~~~

2. Clone this repo

3. Install the [requirements](requirements.txt)

4. Install [apex](https://github.com/NVIDIA/apex):
    ~~~
    cd Glo-in-One_v2/apex
    python3 setup.py install
    cd ..
    ~~~
