.. MorphCT documentation master file, created by
   sphinx-quickstart on Thu Apr 25 14:23:34 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MorphCT's documentation!
===================================

The intention of this code is to form a modular, computational pipeline that can powerfully and coherently relate organic molecular morphology on the Angstrom lengthscale to electronic device performance over hundreds of nanometers.

MorphCT accomplishes this by:
	* Splitting a morphology into electronically-active chromophores that charge carriers (holes in the case of donor materials, electrons for acceptors) are likely to be delocalized along, and can perform quantized charge hops between.
	* Performing high-throughput, fast quantum chemical calculations (QCCs) to obtain the energetic landscape caused by conformational disorder, as well as electronic transfer integrals between chromophore pairs.
	* Using these calculated electronic properties as inputs into a kinetic Monte Carlo (KMC) algorithm to simulate the motion of charge carriers throughout the device allowing carrier mobilities to be obtained (a good proxy for device performance).

.. note::

	This project is under active development.

Getting Started
----------------

.. toctree::
	installation

Classes
--------

.. toctree::
	:maxdepth: 2
	
	chromo_class
	carrier
	System <system_class>


Python API
-----------

.. toctree::
	:maxdepth: 1
	:titlesonly:
	
	chromophores
	execute_qcc
	helper_functions
	kmc_analyze
	mobility_kmc
	system
	transfer_integrals


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
