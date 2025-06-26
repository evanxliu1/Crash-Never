"""
eda_utils.py

Core functions for exploratory data analysis (EDA) on the Nexar dataset.
Intended to be imported by EDA scripts.
"""

import pandas as pd
import matplotlib.pyplot as plt

def analyze_class_distribution(csv_path):
    df = pd.read_csv(csv_path)
    print('Class distribution:')
    print(df['target'].value_counts())
    df['target'].value_counts().plot(kind='bar', title='Class Distribution')
    plt.show()

def plot_event_timing(csv_path):
    df = pd.read_csv(csv_path)
    if 'time_of_event' in df.columns and 'time_of_alert' in df.columns:
        plt.hist(df['time_of_event'] - df['time_of_alert'], bins=30)
        plt.title('Event-Alert Timing Distribution')
        plt.xlabel('Time to Alert (s)')
        plt.ylabel('Count')
        plt.show()
    else:
        print('time_of_event and time_of_alert columns not found in CSV.')
