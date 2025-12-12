# Introduction to Geospatial AI Workshop

This folder contains materials for the KartAI workshop on geospatial AI and building detection.

## Notebook Location

The workshop notebook can be opened directly in Google Colab:
- [Open in Google Colab](https://colab.research.google.com/github/kartAI/kartAI/blob/master/workshop_material/introduction_to_geospatial_ai_colab.ipynb)

## Prerequisites

To participate in the workshop, you need:
1. A Google account (to use Google Colab)
2. Access credentials (provided by workshop organizers):
   - WMS API Key
   - Database password
   - Azure Table Storage SAS Token

## Workshop Overview

In this workshop you will:
1. Create training data from aerial images and building data
2. Train a machine learning model to detect buildings
3. Evaluate and visualize your model's predictions

The notebook is self-contained and will clone the necessary KartAI repository automatically.

## Tasks for Workshop Organizers

### Before the workshop:
1. Generate a new WMS password (or reuse an existing one) with expiration after the workshop is finished
2. Prepare Azure Table Storage SAS token for the scoreboard
3. Create a GitHub Gist with the credentials to share with participants

### At the start of the workshop:
1. Explain how to open the notebook in Google Colab
2. Give a brief notebook intro (What is a notebook, what is a cell, how to run a cell)
3. Show how to choose GPU environment in Colab (Runtime → Change runtime type → T4 GPU)
