# Semester Project: Exploring GraphToken's Ability to Model Complex Relationships for LLM-powered Recommendation Systems

## Overview
This repository contains the implementation and evaluation of a pipeline combining **Graph Neural Networks (GNNs)** and **Large Language Models (LLMs)** for complex graph tasks, specifically leveraging the **Movielens100k** dataset. Inspired by the GraphToken framework, this project extends the scope of evaluation from simple graphs to more complex, heterogeneous graphs with diverse node attributes, as often found in recommendation systems.

---

## Repository Structure
- **`src/gnn`**  
  Training and evaluation script for baseline GNN models.
  
- **`src/gnnllm`**  
  Training and evaluation script for full GraphToken pipeline.
  
- **`src/data/`**  
  Contains dataset loaders and utilities for creating input data for the pipeline.  

- **`src/graph/`**  
  Graph utility files for handling graph structures and encoding.  

- **`src/models/`**  
  Contains implementations of the GNN and LLM models.  

- **`src/utils/`**  
  Utility scripts for model evaluation, logging, and metrics.

- **`notebooks/`**  
  Evaluations and plots used throughout the report.

---

## Run on Google Colab

The pipeline can be easily executed on **Google Colab** for experimentation and evaluation. Follow the provided link below to access a pre-configured Colab notebook:

[Run on Google Colab](https://colab.research.google.com/drive/1YjbvyzLUD4R28PrMsPe0AIl2_9omuxHT?usp=sharing)

### Requirements:
- The pipeline requires an A100 GPU for efficient execution due to the computational demands of Llama3.1-8b. To select the A100 GPU in the Colab runtime settings:
  1. Go to `Runtime` > `Change runtime type`.
  2. Under `Hardware accelerator`, select **GPU**.
  3. Ensure the GPU type is set to **A100**.

---
