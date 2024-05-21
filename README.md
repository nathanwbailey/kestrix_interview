# Kestrix Interview Code

Implements a solution for the coding challenge provided by Kestrix (kestrix.io)

### Code:
The main code is located in the following two files:
* main.py - Main entry file, finds the planes and saves them to file
* utils.py - Contains various functions used in main

### Other files
* property.ply - Mesh file provided by Kestrix
* requirements.txt - Contains pip packages used
* lint.sh - Runs linting tools on the Python code

### How to run:
* Install python (3.11.9)
* Install PDAL (see https://pdal.io/en/2.7-maintenance/)
* (Optionally) Make a virtual environment (e.g. python3 -m venv venv)
* Install the pip packages: pip install -r requirements.txt
* Run the script: python3 main.py
    * Planes are saved into directories roofs and walls 

### Rough Code (miscellaneous):
There is also some rough code in the following files: (not used in main.py)
* visualize_planes.py - Used to visualize the output planes
* pdal_play.py - Used to play around with PDAL
