# Satellite Image Change Detection - The Changing Face of Earth
This MATLAB project implements a computer vision pipeline for detecting and visualizing changes in satellite imagery over time, as part of the Computer Vision Challenge (SoSe 2025) at TUM. See G16 for our full project.

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
- **`Plot_Class_Heatmap.m`** - Generates heatmaps showing the frequency of pixel changes across all images, with a color scale indicating how often each pixel has changed over time
- **`Plot_Class_Percentages_Over_Time.m`** - Plots the percentage distribution of different classes or change types over the time series
- **`Veraenderungszeitpunkte.m`** - Visualizes when specific areas changed in the image sequence, with each change timepoint represented by a unique color (overlapping colors for pixels that changed multiple times)
- **`heatmap_veraenderungshauefigkeit.m`** - Creates frequency heatmaps showing how often each pixel changed throughout the entire image sequence
- **`timelapseSlider.m`** - Interactive timelapse viewer with slider control for navigating through the image sequence
- **`timelapse_bilderfolge.m`** - Generates automated timelapse animations from the image sequence
- **`visualisiere_veraenderungen_nach_flaeche.m`** - Visualizes changes categorized by size (small/large changes based on connected pixel groups), with adjustable threshold for defining large vs. small changes
- **`visualize_schnell_langsam.m`** - Analyzes and visualizes the speed of changes by comparing average vs. maximum pixel changes (red for fast changes: high max/low average, blue for slow changes: low max/high average)

## Usage

To run the application:
- Execute `ChangeDetectionApp.mlapp` in MATLAB

For usage guidance:
- Check the `HELP` tab within the GUI once the app is running

## File Requirements

Images should be named following the pattern `MM_YYYY.extension` (e.g., `11_2020.jpg`, `04_2019.png`) or `YYYY_MM.extension` (e.g., `2020_11.jpg`, `2019_04.png`)  and stored in a single folder. Minimum 2 images required.

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

