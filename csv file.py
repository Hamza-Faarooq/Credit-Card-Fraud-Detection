import pandas as pd
import numpy as np

# Sample DataFrame similar to the credit card fraud detection dataset structure
data = {
    'Time': np.random.randint(0, 172800, size=1000),  # Random time in seconds within two days
    'V1': np.random.randn(1000),
    'V2': np.random.randn(1000),
    'V3': np.random.randn(1000),
    'V4': np.random.randn(1000),
    'V5': np.random.randn(1000),
    'V6': np.random.randn(1000),
    'V7': np.random.randn(1000),
    'V8': np.random.randn(1000),
    'V9': np.random.randn(1000),
    'V10': np.random.randn(1000),
    'V11': np.random.randn(1000),
    'V12': np.random.randn(1000),
    'V13': np.random.randn(1000),
    'V14': np.random.randn(1000),
    'V15': np.random.randn(1000),
    'V16': np.random.randn(1000),
    'V17': np.random.randn(1000),
    'V18': np.random.randn(1000),
    'V19': np.random.randn(1000),
    'V20': np.random.randn(1000),
    'V21': np.random.randn(1000),
    'V22': np.random.randn(1000),
    'V23': np.random.randn(1000),
    'V24': np.random.randn(1000),
    'V25': np.random.randn(1000),
    'V26': np.random.randn(1000),
    'V27': np.random.randn(1000),
    'V28': np.random.randn(1000),
    'Amount': np.random.uniform(0, 5000, size=1000),  # Random amount between 0 and 5000
    'Class': np.random.randint(0, 2, size=1000)  # Random class, either 0 or 1
}

# Create DataFrame
df = pd.DataFrame(data)

# Save DataFrame to CSV in internal memory
df.to_csv('creditcard.csv', index=False)

print("DataFrame saved to creditcard.csv")