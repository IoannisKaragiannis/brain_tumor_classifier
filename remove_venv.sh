#!/bin/bash

PYTHON="python3.10"
VENVS_PATH="$HOME/python_venv"
GIT_PROJECTS_PATH=$HOME/git_repos
VENV_NAME="brain"
PURGE_CACHE=true

echo "**************************[$VENV_NAME-REMOVAL]************************"
echo "Uninstalling '$VENV_NAME' packages  erasing environment"
echo "**********************************************************************"

# activate environment
source $VENVS_PATH/$VENV_NAME/bin/activate

if [ "$PURGE_CACHE" = true ] ; then
    $VENVS_PATH/$VENV_NAME/bin/$PYTHON -m pip cache purge
fi

deactivate

sudo rm -rf $VENVS_PATH/$VENV_NAME

echo "*******************************************************"
echo "$VENVS_PATH/$VENV_NAME WAS SUCCESSFULLY UNINSTALLED!!!"
echo "*******************************************************"
