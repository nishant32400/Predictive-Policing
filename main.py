# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# file_path = "crime-data.csv"  # Adjust path if needed
# df = pd.read_csv(file_path)

# # Clean column names
# df.columns = df.columns.str.strip().str.lower()

# # Ensure numeric columns are properly formatted
# df['# killed'] = pd.to_numeric(df['# killed'], errors='coerce')
# df['# injured'] = pd.to_numeric(df['# injured'], errors='coerce')

# # Preview the cleaned data
# print(df.head())

# # Accident types distribution
# accident_counts = df['type of accident'].value_counts()
# print("Accident Type Distribution:")
# print(accident_counts)

# # Distribution of accidents by district
# district_accidents = df['district'].value_counts()
# print("\nAccidents by District:")
# print(district_accidents)

# # Total injuries and fatalities
# total_injured = df['# injured'].sum()
# total_killed = df['# killed'].sum()
# print(f"\nTotal Injuries: {total_injured}, Total Fatalities: {total_killed}")

# # Visualize accident types
# plt.figure(figsize=(8, 5))
# sns.barplot(x=accident_counts.index, y=accident_counts.values)
# plt.title("Accident Type Distribution")
# plt.xlabel("Type of Accident")
# plt.ylabel("Count")
# plt.xticks(rotation=45)
# plt.show()

# # Visualize accidents by district
# plt.figure(figsize=(10, 6))
# sns.barplot(x=district_accidents.index, y=district_accidents.values)
# plt.title("Accidents by District")
# plt.xlabel("District")
# plt.ylabel("Count")
# plt.xticks(rotation=90)
# plt.show()

# # Summary
# summary = {
#     "Accident Type Distribution": accident_counts.to_dict(),
#     "Accidents by District": district_accidents.to_dict(),
#     "Total Injuries": total_injured,
#     "Total Fatalities": total_killed,
# }

# # Save the summary to a file
# with open("project_summary.txt", "w") as file:
#     for key, value in summary.items():
#         file.write(f"{key}:\n{value}\n\n")

# #2nd option
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# file_path = "crime-data.csv"  # Adjust path if needed
# df = pd.read_csv(file_path)

# # Cleaning the data
# df.columns = df.columns.str.strip()  # Strip any leading/trailing spaces from column names
# df = df.dropna(subset=['YEAR', 'TYPE OF ACCIDENT', '# INJURED', '# KILLED', 'DISTRICT'])  # Ensure required columns are not null

# # Convert necessary columns to appropriate data types
# df['YEAR'] = df['YEAR'].astype(int)
# df['# INJURED'] = pd.to_numeric(df['# INJURED'], errors='coerce').fillna(0).astype(int)
# df['# KILLED'] = pd.to_numeric(df['# KILLED'], errors='coerce').fillna(0).astype(int)

# # Group data by year and type of accident
# accident_yearly = df.groupby(['YEAR', 'TYPE OF ACCIDENT']).size().unstack(fill_value=0)

# # Plot: Accident Type Distribution by Year
# plt.figure(figsize=(12, 6))
# accident_yearly.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
# plt.title("Accident Type Distribution by Year", fontsize=16)
# plt.xlabel("Year", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.legend(title="Type of Accident", loc='upper right', fontsize=10)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig("Accident_Type_Distribution_by_Year.png")
# plt.show()

# # Plot: Accidents by District (Optimized for Presentation)
# district_accidents = df['DISTRICT'].value_counts()
# plt.figure(figsize=(10, 6))
# sns.barplot(x=district_accidents.values, y=district_accidents.index, palette="coolwarm")
# plt.title("Number of Accidents by District", fontsize=16)
# plt.xlabel("Count", fontsize=12)
# plt.ylabel("District", fontsize=12)
# plt.tight_layout()
# plt.savefig("Accidents_by_District.png")
# plt.show()

# # Summary of injuries and fatalities
# total_injured = df['# INJURED'].sum()
# total_killed = df['# KILLED'].sum()

# print(f"Total Injuries: {total_injured}")
# print(f"Total Fatalities: {total_killed}")


#3rd
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans

# # Load the dataset
# file_path = "data.csv"  # Replace with your dataset path
# data = pd.read_csv(file_path)

# # Display first 5 rows
# print("Dataset Preview:")
# print(data.head())

# # Check dataset structure
# print("\nDataset Info:")
# print(data.info())

# # Check for missing values
# print("\nMissing Values Count:")
# print(data.isnull().sum())

# # Step 2: Preprocess Data
# # Fill missing values for numerical columns with their mean (only for numeric columns)
# numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
# data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# # Drop duplicate rows
# data = data.drop_duplicates()

# # Convert 'lat' and 'long' columns to numerical values (if necessary)
# data['lat'] = pd.to_numeric(data['lat'], errors='coerce')
# data['long'] = pd.to_numeric(data['long'], errors='coerce')

# print("\nCleaned Dataset Preview:")
# print(data.head())

# # Step 3: Perform Basic Exploratory Data Analysis (EDA)

# # Plot frequency of different crime types
# if 'murder' in data.columns:
#     plt.figure(figsize=(12, 6))
#     crime_types = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']
#     crime_counts = data[crime_types].sum().sort_values(ascending=False)
#     sns.barplot(x=crime_counts.index, y=crime_counts.values, palette='coolwarm')
#     plt.title("Crime Frequency by Type", fontsize=16)
#     plt.xlabel("Crime Type", fontsize=12)
#     plt.ylabel("Total Crime Count", fontsize=12)
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()  # Ensures no cut-off labels
#     plt.show()

# # Step 4: Demonstrate a Simple Predictive Model

# # KMeans Clustering based on latitude and longitude to identify crime-prone areas
# if 'lat' in data.columns and 'long' in data.columns:
#     locations = data[['lat', 'long']].dropna()
#     kmeans = KMeans(n_clusters=5, random_state=0)
#     clusters = kmeans.fit_predict(locations)
    
#     # Add clusters to data
#     locations['Cluster'] = clusters
#     data['Cluster'] = clusters
#     print("\nClustered Locations Preview:")
#     print(locations.head())

#     # Plot clusters with location names
#     plt.figure(figsize=(12, 8))
#     plt.scatter(locations['lat'], locations['long'], c=clusters, cmap='viridis', s=100, edgecolors='black', alpha=0.6)
    
#     # Annotating location names on the map
#     for i, row in locations.iterrows():
#         plt.text(row['lat'], row['long'], data.loc[i, 'location'], fontsize=9, ha='right', color='black')

#     plt.title("Crime-Prone Area Clusters with Locations", fontsize=16)
#     plt.xlabel("Latitude", fontsize=12)
#     plt.ylabel("Longitude", fontsize=12)
#     plt.tight_layout()
#     plt.show()

# else:
#     print("Latitude and Longitude data not available for clustering.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
data = pd.read_csv('data.csv')

# Dataset Preview and Info
print("Dataset Preview:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# Checking for missing values
print("\nMissing Values Count:")
print(data.isnull().sum())

# Fill missing values only in numeric columns
numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Selecting the columns for clustering: 'lat' and 'long'
locations = data[['lat', 'long']]

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(locations)

# Adding clusters to the data
data['Cluster'] = clusters

# Print out the cluster centers for reference
print(f"\nCluster Centers:\n{kmeans.cluster_centers_}")

# Evaluate clustering using silhouette score
silhouette_avg = silhouette_score(locations, clusters)
print(f"\nSilhouette Score: {silhouette_avg:.2f}")

# Inertia (Within-Cluster Sum of Squares)
inertia = kmeans.inertia_
print(f"Inertia (Compactness of Clusters): {inertia:.2f}")

# Visualization: Plotting clusters on the map
plt.figure(figsize=(10, 8))
sns.scatterplot(x=data['long'], y=data['lat'], hue=data['Cluster'], palette='viridis', s=100, edgecolor='k', marker='o')
plt.title('Geographical Clusters Based on Crime Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Annotating locations with names for clarity
for i in range(len(data)):
    plt.text(data['long'][i], data['lat'][i], data['location'][i], fontsize=8, ha='right')

plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Saving the clustered dataset (Optional)
data.to_csv('clustered_data.csv', index=False)


