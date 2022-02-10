# Identity Card OCR

# Setup

* Install [Tesseract](https://guides.library.illinois.edu/c.php?g=347520&p=4121425)
* Install [Poetry](https://python-poetry.org/docs/#installation)
* Prepare environment using `make setup` or run each command from setup step in [Makefile](Makefile)
* Right click `ocr.ipynb` and select `Add Jupyter Connection`
* In the new dialog click `Add`

# Running

* Put a image in folder `data` called `fata.jpeg`
* Open `ocr.ipynb` and click double green arrow to run all cells

# Pre-commit

When you finish work and you're ready to commit run `make lint`
