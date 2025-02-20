# CASSINI DENOISER
Tools to perform noise analysis on Images from the Cassini-Huygens mission.

## Repository set-up

### Python virtual environment

After cloning this repository, you need to create a new python virtual environment.
If you are using any respectable IDE (ex: pycharm, etc.), 
it will automatically create it for you and auto-activate it when needed.

If not, create one manually:
```shell
python3 -m venv venv
```

and activate it every time you open a new shell:
```shell
source venv/bin/activate
```

### Python packages

After activating a virtual environment, install all the required python packages:
```shell
pip3 install -r requirements.txt
```

## Execute GUI tools

Auto-prompt for the input files:

```shell
python .\gui_region_analysis.py
```

or, specifying the args directly:

```shell
python .\gui_region_analysis.py  --header data/N1473172656_1.LBL --image data/N1473172656_1.IMG --output output/gui_analysis
```

## Compile into EXE

```shell
pyinstaller --onefile --windowed --name "ImageNoiseAnalysis" --icon="icons/sound_wave.ico" gui_region_analysis.py
```
