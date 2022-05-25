"""
This module simply loads all the nox targets defined in the reference noxfile and make them
available for the weather_forcast/gwd use case.
This file can be enriched by use case specific targets.
"""

import os
import sys
import inspect

# Insert the tools/nox folder to the python path to fetch the nox_ref_file.py content
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
build_ref_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "tools", "nox")
sys.path.insert(0, build_ref_dir)

# Fetch the nox_ref_file.py content
import nox_ref_file

# Insert below the use case specific targets
