# Ilex

Ilex is a tool for handling, analysing, manipulating, and visualising high-time resolution beamformed fast radio burst (FRB) data as output by [CELEBI](https://github.com/askap-craco/CELEBI). Current functionality includes:

- Automatic loading of FRB data, including metadata
- Automatic downloading of FRB data and metadata from OzStar if some or all files are not found
- Automatic cropping of FRB data to give quick access to the burst itself, while keeping the entire dataset available
- Flexible manipulation of time and frequency resolutions
- Manipulation of number of frequency channels in FRB dynamic spectra
- Coherent dedispersion to any dispersion measure (DM)
- Optional zapping of data in frequency range where FRB is not expected to be due to falling off the edge of the dataset, if applicable
- Plotting/visualisation functions
  - PA, Stokes IQUV profiles, and Stokes I dynamic spectrum (together or separately)
  - Stokes ILV profiles
- Analysis functions
  - Second-order intensity correlation function (g2)
    - At a particular DM or across a range of DMs
  - Modified coherence function
  - Polarisation angle with uncertainty
  - Rotation measure
