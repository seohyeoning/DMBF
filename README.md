# Dynamic Multi-Biosignal Fusion (DMBF)
S.-H. Park et al., "Dynamic Multi-biosignal Fusion for Detecting the Mental States of Drivers and Passengers in Vehicles," Journal of Biomedical and Health Informatics (under review), 2024. 

## Overview
DMBF is an innovative framework designed for detecting the mental states of drivers and passengers in vehicles. By leveraging multi-modal biosignals such as EEG, ECG, and PPG, the DMBF method provides a robust and reliable solution for real-time mental state monitoring. This approach utilizes a dynamic gate mechanism and spatial-temporal attention to achieve superior classification performance across various scenarios, including motion sickness, drowsiness, and sustained attention.

![DMBF Architecture](DMBF.png)

## Key Features
- **Dynamic Fusion**: Adjusts biosignal contributions based on data quality and informativeness, ensuring robust performance.
- **Spatial-Temporal Attention Module (STAM)**: Captures key signal patterns over time and across channels.
- **Confidence-Aware Learning**: Incorporates reliability estimation into model training, enhancing prediction accuracy.
- **Versatile Application**: Validated on datasets spanning motion sickness, distraction, drowsiness, and more.
- **High Performance**: Outperforms static fusion and traditional dynamic models on various biosignal-based datasets.

## Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Key Models**: EEGNet, Confidence-Aware Mechanisms, Attention-Based Fusion
- **Dataset Sources**: Multi-modal biosignal datasets including motion sickness, sustained attention, and driver distraction data.

## Installation & Usage
### Prerequisites
- Python 3.7 or later
- PyTorch 1.8.0+
- CUDA Toolkit for GPU acceleration (if available)
- Required Python packages (e.g., numpy, scipy, matplotlib)

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/seohyeoning/DMBF.git
cd DMBF
pip install -r requirements.txt
