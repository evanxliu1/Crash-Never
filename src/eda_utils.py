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
    print(df['event'].value_counts())
    df['event'].value_counts().plot(kind='bar', title='Class Distribution')
    plt.show()

def plot_event_timing(csv_path):
    df = pd.read_csv(csv_path)
    if 'event_time' in df.columns and 'alert_time' in df.columns:
        plt.hist(df['event_time'] - df['alert_time'], bins=30)
        plt.title('Event-Alert Timing Distribution')
        plt.xlabel('Time to Alert (s)')
        plt.ylabel('Count')
        plt.show()
    else:
        print('event_time and alert_time columns not found in CSV.')
