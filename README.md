# Cardiac Dreamer: Real-time Handheld Ultrasound Navigation System Training Module

## Overview

Cardiac Dreamer is a deep learning-based training framework designed for real-time handheld ultrasound navigation systems. This project implements a Vision Transformer architecture that enables intelligent probe guidance for echocardiography, helping clinicians achieve optimal ultrasound imaging views through automated 6-DOF (six degrees of freedom) action predictions.
> Note: This repository focuses on the model training phase and does not include deployment or real-time inference integration.

The system combines ResNet34 feature extraction with a novel Channel Token Transformer architecture, utilizing 512 channel tokens plus one Action-CLS token to process ultrasound images and predict optimal probe movements. This approach addresses the critical challenge of ultrasound probe positioning, which traditionally requires extensive training and experience. The dataset used for training was collected with the invaluable support of [ACO SmartCare](https://acohealthcare.com/zh/about/%E9%97%9C%E6%96%BC/).



![image](doc_image\Readme_image\model_architecture.png)
> model architecture overview


## Documentation Structure

This repository includes comprehensive documentation organized as follows:

- **[Introduction](docs/01_introduction.md)**: Background, motivation, and research objectives
- **[System Overview](docs/02_system_overview.md)**: Hardware and software architecture
- **[Data Management](docs/03_data.md)**: Dataset format and processing pipeline
- **[Model Architecture](docs/04_model_architecture.md)**: Detailed technical implementation with code explanations
- **[Setup Guide](docs/05_setup_and_install.md)**: Environment configuration and installation
- **[Training Guide](docs/06_training_and_inference.md)**: Model training and inference procedures
- **[Experimental Results](docs/08_results.md)**: Performance analysis and validation results
- **[Future Work](docs/09_future_work.md)**: Research conclusions and improvement directions
- **[Contributors](docs/10_contributors.md)**: Project team and acknowledgments

## Requirements

- Python 3.10+
- PyTorch 2.2.2+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for full dataset training
- See [Setup Guide](docs/05_setup_and_install.md) for detailed requirements


## Model Performance

The system demonstrates strong performance across multiple evaluation metrics:

- Cross-validation Mean Absolute Error (MAE) analysis
- Patient-level generalization validation
- 6-DOF action prediction accuracy

See [Experimental Results](docs/08_results.md) for detailed performance analysis.

## Development Workflow

1. **Environment Setup**: Configure Python environment and dependencies
2. **Data Preparation**: Process ultrasound sequences and action annotations
3. **Model Training**: Execute training with cross-validation
4. **Evaluation**: Analyze results and generate performance reports

## Contributing

This project follows standard academic research practices. See [Contributors](docs/10_contributors.md) for team information and contribution guidelines.

## License

This project is developed for academic research purposes. Please refer to the license file for usage terms and conditions.

## Citation

If you use this work in your research, please cite our paper:

```bibtex
[Citation to be added upon publication]
```

## Support

For questions about installation, usage, or technical implementation, please refer to the documentation or contact the development team through the contributors listed in [Contributors](docs/10_contributors.md).