Installation
------------

To create a local environment with conda, run::

	git clone git@gihub.com:cmelab/morphct.git
	conda env create -f environment.yml
	conda activate morphct

To install the MorphCT module, run this command within the cloned directory and with the environment, morphct, activated::

	pip install -e .

And to test your installation, run::

	pytest

Using a Container
-----------------

To use MorphCT in a prebuilt container (using Singularity), run::

	singularity pull docker://cmelab/morphct:latest
	singularity exec morphct_latest.sif bash

Or using Docker, run::

	docker pull cmelab/morphct:latest
	docker run -it cmelab/morphct:latest

Examples
--------
