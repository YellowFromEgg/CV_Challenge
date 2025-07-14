Readme: Erstellen Sie eine Readme.txt Datei, die ei-
ne Anleitung Ihrer Applikation und eine Liste aller f¨ur
die Ausf¨uhrung der Applikation notwendigen Toolbo-
xen beinhaltet.

# Satellite Image Change Detection - MATLAB Project

This MATLAB project implements a computer vision pipeline for detecting and visualizing changes in satellite imagery over time, as part of the Computer Vision Challenge (SoSe 2025) at TUM.

## Overview

The system automatically registers satellite images from the same location taken at different times, segments them into land-use classes, and provides multiple visualization methods to highlight temporal changes.

## Core Components

### Main Classes
- **`SatelliteImageRegistration.m`** - Main workflow class that orchestrates the entire pipeline from image loading to visualization

### Image Processing Functions
- **`Register_Color_Images.m`** - Performs feature-based image registration using SURF and corner detection with adaptive parameters for different scene types
- **`Segmentation.m`** - Segments images into classes: Water/Forest, Land, Urban/Agriculture, Snow, Rivers/Roads
- **`Transform_Segmented_Images.m`** - Applies geometric transformations to segmented masks

### Visualization Functions
- **`Plot_Class_Heatmap.m`** - 
- **`Plot_Class_Percentages_Over_Time.m`** - 
- **`Veraenderungszeitpunkte.m`** - 
- **`heatmap_veraenderungshauefigkeit.m`** - 
- **`timelapseSlider.m`** - 
- **`timelapse_bilderfolge.m`** - 
- **`visualisiere_veraenderungen_nach_flaeche.m`** - 
- **`visualize_schnell_langsam.m`** - 

## Usage

### Basic Usage
```matlab
% Create instance and run complete pipeline
reg = SatelliteImageRegistration('path/to/image/folder');
reg.run();
```

## File Requirements

Images should be named following the pattern `YYYY_MM.extension` (e.g., `2020_11.jpg`, `2019_04.png`) and stored in a single folder. Minimum 2 images required.

## Required MATLAB Toolboxes

- **Computer Vision Toolbox** - Feature detection, image registration, geometric transformations
- **Image Processing Toolbox** - Basic image operations, filtering, morphology
- **Statistics and Machine Learning Toolbox** - K-means clustering, statistical functions

## Output

The system provides:
- Registered color images aligned to a common reference frame
- Segmented land-use maps with 6 classes
- Multiple change visualization options (heatmaps, temporal plots, timelapse)
- Interactive GUI for parameter adjustment and visualization selection

