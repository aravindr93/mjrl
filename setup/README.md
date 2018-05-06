# Installation

A short guide to install this package is below. The package relies on `mujoco-py` which might be the trickiest part of the installation. See `known issues` below and also instructions from the mujoco-py [page](https://github.com/openai/mujoco-py) if you are stuck with mujoco-py installation.

## Linux

- Download MuJoCo binaries from the official [website](http://www.mujoco.org/) and also obtain the license key.
- Unzip the downloaded mjpro150 directory into `~/.mujoco/mjpro150`, and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`
- Install osmesa related dependencies:
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev build-essential libglfw3
```
- Update `bashrc` by adding the following lines and source it
```
export LD_LIBRARY_PATH="<path/to/.mujoco>/mjpro150/bin:$LD_LIBRARY_PATH"
export MUJOCO_PY_FORCE_CPU=True
alias MJPL='LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so'
```
- Install this package using
```
$ conda update conda
$ cd <path/to/mjrl>
$ conda env create -f setup/linux.yml
$ source activate mjrl-env
$ pip install -e .
```

## Mac OS

- Download MuJoCo binaries from the official [website](http://www.mujoco.org/) and also obtain the license key.
- Unzip the downloaded mjpro150 directory into `~/.mujoco/mjpro150`, and place your license key (mjkey.txt) at `~/.mujoco/mjkey.txt`
- Update `bashrc` by adding the following lines and source it
```
export LD_LIBRARY_PATH="<path/to/.mujoco>/mjpro150/bin:$LD_LIBRARY_PATH"
```
- Install this package using
```
$ conda update conda
$ cd path/to/mjrl
$ conda env create -f setup/mac.yml
$ source activate mjrl-env
$ pip install -e .
```

`mujoco-py` seems to require specific versions of GCC installed for MacOS. Assuming you have Homebrew installed (https://brew.sh/), you can install the correct GCC version with the command `brew install gcc --without-multilib`. This may require uninstalling other versions of GCC that may have been previously installed with `brew remove gcc@6` for example. You can see which brew packages were already installed with `brew list`.

## Known Issues

- Visualization in linux: If the linux system has a GPU, then mujoco-py does not automatically preload the correct drivers. We added an alias `MJPL` in bashrc (see instructions) which stands for mujoco pre-load. When runing any python script that requires rendering, prepend the execution with MJPL.
```
$ MJPL python script.py
```

- Errors related to osmesa during installation. This is a `mujoco-py` build error and would likely go away if the following command is used before creating the conda environment. If the problem still persists, please contact the developers of mujoco-py
```
$ sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev
```

- If conda environment creation gets interrupted for some reason, you can resume it with the following:
```
$ conda env update -n mjrl-env -f setup/[mac,linux].yml
```
