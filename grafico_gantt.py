import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the data
df = pd.read_csv('saida_greedy_prioridade_balanceamento.csv', delimiter=';')

# Convert time units to days
minutes_in_a_day = 1440
df['start_time_days'] = df['start_time'] / minutes_in_a_day
df['end_time_days'] = df['end_time'] / minutes_in_a_day
df['duration_days'] = df['end_time_days'] - df['start_time_days']
df['prazo_days'] = df['prazo'] / minutes_in_a_day

# Get unique resources and assign a numerical position for each on the y-axis
resource_list = df['resource_name'].unique()
resource_mapping = {name: i for i, name in enumerate(resource_list)}
df['resource_pos'] = df['resource_name'].map(resource_mapping)

# Normalize 'prazo' to a range from 0 to 1 for color mapping
# Set vmax to 6 days (8640 minutes / 1440 min/day) to ensure any prazo >= 6 is fully green
norm = mcolors.Normalize(vmin=df['prazo_days'].min(), vmax=6)
cmap = plt.cm.RdYlGn

# Create the Gantt chart
plt.figure(figsize=(15, 8))

for index, row in df.iterrows():
    # Use the 'prazo_days' for color mapping
    color = cmap(norm(row['prazo_days']))
    plt.barh(
        y=row['resource_pos'],
        width=row['duration_days'],
        left=row['start_time_days'],
        height=0.8,
        color=color
    )

# Customize the plot
plt.yticks(
    ticks=list(resource_mapping.values()),
    labels=list(resource_mapping.keys())
)
plt.xlabel('Time (days)')
plt.ylabel('Resource Name')
plt.title('Gantt Chart of Task Scheduling by Prazo (Time in Days)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add a color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label('Prazo (days)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig('.\\charts\\gantt_chart_prazo.png')
print("Gantt chart with all time units in days, including the color bar, successfully saved as 'gantt_chart_prazo_days_in_legend.png'")