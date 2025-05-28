---
sdk: docker
app_port: 8000
---

# BluePrint-Detection-Model

This project is a take-home assessment for the Vision Intern hiring round at Palcode.ai. The model uses YOLOv8s to detect windows and doors in architectural blueprints, built as a FastAPI application for easy inference via an API endpoint.

## Overview

The BluePrint-Detection-Model is designed to identify and localize windows and doors in blueprint images. It uses a fine-tuned YOLOv8s model trained on a custom dataset of blueprints. The API accepts image uploads and returns bounding box coordinates, labels (`window` or `door`), and confidence scores for detected objects.
