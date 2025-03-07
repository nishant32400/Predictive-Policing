import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Loading the dataset
file_path = "crime-data.csv"  # Adjust path if needed
df = pd.read_csv(file_path)

# # thinking of Cleaning the data
# df.columns = df.columns.str.strip()
# df = df.dropna()  # Drop rows with missing values

# Preview the cleaned data
print(df.head())

# Accident types distribution
accident_counts = df['TYPE OF ACCIDENT'].value_counts()
print("Accident Type Distribution:")
print(accident_counts)

# Distribution of accidents by district
district_accidents = df['DISTRICT'].value_counts()
print("\nAccidents by District:")
print(district_accidents)

# total injuries and fatalities
total_injured = df['# INJURED'].sum()
total_killed = df['# KILLED'].sum()
print(f"\nTotal Injuries: {total_injured}, Total Fatalities: {total_killed}")



# Visualizing accident types
plt.figure(figsize=(8, 5))
sns.barplot(x=accident_counts.index, y=accident_counts.values)
plt.title("Accident Type Distribution")
plt.xlabel("Type of Accident")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Visualizing accidents by district
plt.figure(figsize=(10, 6))
sns.barplot(x=district_accidents.index, y=district_accidents.values)
plt.title("Accidents by District")
plt.xlabel("District")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()



# Summary
summary = {
    "Accident Type Distribution": accident_counts.to_dict(),
    "Accidents by District": district_accidents.to_dict(),
    "Total Injuries": total_injured,
    "Total Fatalities": total_killed,
}

# Save the summary to a file
with open("project_summary.txt", "w") as file:
    for key, value in summary.items():
        file.write(f"{key}:\n{value}\n\n")
