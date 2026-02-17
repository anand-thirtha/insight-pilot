import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    datefmt= '%Y-%m-%d %H:%M:%S'
)

def generate_sample_data(start_date: str,
                         num_of_days: int):
    
    '''
        Create synthetic multi-dimensional retail data for given number of days from starting date 
        Args:
        start_date: str YYYY-MM-DD
        num_of_days: int 
    '''
    logging.info(f"Generating sample data for {num_of_days} days starting from {start_date}")
    data = {
        'Date': pd.date_range(start=start_date, periods=num_of_days, freq='D').repeat(2).tolist()[:500],
        'Location': np.random.choice(['New York', 'London', 'Tokyo', 'Berlin'], 500),
        'Store_ID': np.random.choice(['ST-001', 'ST-002', 'ST-003', 'ST-004'], 500),
        'Product_Category': np.random.choice(['Electronics', 'Apparel', 'Home & Kitchen', 'Groceries'], 500),
        'Sales_USD': np.random.uniform(100, 5000, 500).round(2),
        'Quantity': np.random.randint(1, 50, 500),
        'Discount_Applied': np.random.choice([True, False], 500, p=[0.3, 0.7])
    }

    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)
    logging.info('Sample data saved as data.csv in src/ folder!')


