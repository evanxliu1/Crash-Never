import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load train.csv
df = pd.read_csv('train.csv')

# --- 1. Class Distribution ---
class_counts = df['target'].value_counts().sort_index()
plt.figure(figsize=(6,4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette=['#4CAF50', '#F44336'])
plt.xticks([0,1], ['No Collision', 'Collision'])
plt.title('Class Distribution (Collision vs. No Collision)')
plt.ylabel('Count')
plt.xlabel('Class')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

# --- 2. Timing Relationships (alert_time, event_time) ---
pos_df = df[df['target'] == 1].copy()
pos_df['time_of_event'] = pd.to_numeric(pos_df['time_of_event'], errors='coerce')
pos_df['time_of_alert'] = pd.to_numeric(pos_df['time_of_alert'], errors='coerce')
pos_df = pos_df.dropna(subset=['time_of_event', 'time_of_alert'])
pos_df['reaction_time'] = pos_df['time_of_event'] - pos_df['time_of_alert']

plt.figure(figsize=(7,4))
sns.histplot(pos_df['reaction_time'], bins=30, kde=True, color='#2196F3')
plt.title('Distribution of Time to Collision (event_time - alert_time)')
plt.xlabel('Time to Collision (seconds)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('reaction_time_distribution.png')
plt.close()

# --- 3. Print Key Stats ---
print('--- Dataset Overview ---')
print(f"Total samples: {len(df)}")
print(f"Collision (target=1): {class_counts.get(1,0)}")
print(f"No Collision (target=0): {class_counts.get(0,0)}")
if not pos_df.empty:
    print(f"\nMean time to collision: {pos_df['reaction_time'].mean():.3f} s")
    print(f"Median time to collision: {pos_df['reaction_time'].median():.3f} s")
    print(f"Std time to collision: {pos_df['reaction_time'].std():.3f} s")
    print(f"Min time to collision: {pos_df['reaction_time'].min():.3f} s")
    print(f"Max time to collision: {pos_df['reaction_time'].max():.3f} s")
else:
    print('No positive samples with valid timing info.')

print('\nSaved plots: class_distribution.png, reaction_time_distribution.png')
