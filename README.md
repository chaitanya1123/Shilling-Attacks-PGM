# Shilling-Attacks-PGM

Dependencies : py-factorgraph, sklearn, numpy, time, SDLib, pandas, matplotlib
Note: Some of the source code in factorgraph was changed to suit the library to our implementation!
To Run the factor graph code, you will need to download the SDLib library from GitHub to simulate a shilling attack!

Once the library is downloaded,set necessary paths and run data.py to generate just the data! 
Run main.py after giving a unique string for label and profile name! This will build the factor graph for the necessary data!
features.py lists the features that we used for the factor graph.

The AE and RBM were run in MATLAB. The R matrix generated was converted to a .mat file.

The Autoencoder-Shilling.m is a file that will train the AE and generate the output.
To run the RBM, run RBM.m.


