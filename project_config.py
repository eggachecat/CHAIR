import os
import pathlib

CURRENT_SCRIPT_FOLDER = pathlib.Path(__file__).parent.resolve()

EGGACHECAT_PROJECT_ROOT_FOLDER = CURRENT_SCRIPT_FOLDER
EGGACHECAT_PROJECT_SAVE_FOLDER = os.path.join(EGGACHECAT_PROJECT_ROOT_FOLDER, "saves")