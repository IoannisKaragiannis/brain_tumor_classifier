#!/bin/bash

# Control variables (modify them accordingly to your OS and HW)
PYTHON="python3.10"
GIT_PROJECTS_PATH=$HOME/git_repos
PROJECT_NAME=brain_tumor_classifier
VENVS_PATH=$HOME/python_venv
VENV_NAME=brain
REQUIREMENTS=requirements.txt

sudo apt install -y qt5-default
sudo apt-get install -y python-tk python3.10-tk

echo "=========================[$VENV_NAME-SETUP]================================"
echo "Installing essential packages for $VENV_NAME" virtual environment
echo "==========================================================================="

if [ ! -d "$VENVS_PATH" ]; then
    mkdir $$VENVS_PATH
fi

$PYTHON -m venv $VENVS_PATH/$VENV_NAME

source $VENVS_PATH/$VENV_NAME/bin/activate

$VENVS_PATH/$VENV_NAME/bin/$PYTHON -m pip install --upgrade pip
$VENVS_PATH/$VENV_NAME/bin/$PYTHON -m pip install --upgrade setuptools
$VENVS_PATH/$VENV_NAME/bin/$PYTHON -m pip install --upgrade wheel

# navigate to obj_detect_track repo
cd $GIT_PROJECTS_PATH/$PROJECT_NAME

# install requirements on virtual environment
$PYTHON -m pip install -r $REQUIREMENTS

echo "to activate the venv: source $VENVS_PATH/$VENV_NAME/bin/activate"
