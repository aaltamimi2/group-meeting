# Model Performance Report

This file will be populated after running the evaluation pipeline.

## Instructions

Run the following command to generate the full report:

```bash
python -m src.evaluate
```

The report will include:
- Evaluation molecule summary
- Prediction statistics for each model
- Detailed results table
- Visualization plots

## Expected Output

- **Ridge Model**: Linear regression with L2 regularization
- **Random Forest Model**: Ensemble of decision trees
- **Metrics**: MSE, MAE, R² scores
- **Predictions**: Property values for test molecules
