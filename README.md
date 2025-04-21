# Smart-Contract-Analysis
#Overview

This repository documents my work during the Webacy External Data Analysis Externship, where I contributed to projects related to analyzing and interpreting blockchain-based smart contract data. The goal of this externship was to develop insights from decentralized datasets, identify contract vulnerabilities, and build a foundation for real-time monitoring and alert systems.

#Project Goals

Explore external data sources relevant to blockchain security and smart contract activity
 - Clean, label, and structure raw data for deeper analysis
 - Perform frequency and correlation analysis to uncover patterns and potential risks
 - Apply unsupervised machine learning methods to detect anomalies
 - Present key findings that could help power smarter security and trust tools

#Process
#Frequency 
```
df = pd.read_excel('/content/compiled_risk_data.xlsx')
df.head()

df.info()

risk_columns = ['Is_closed_source', 'hidden_owner', 'anti_whale_modifiable',
       'Is_anti_whale', 'Is_honeypot', 'buy_tax', 'sell_tax',
       'slippage_modifiable', 'Is_blacklisted', 'can_take_back_ownership',
       'owner_change_balance', 'is_airdrop_scam', 'selfdestruct', 'trust_list',
       'is_whitelisted', 'is_fake_token', 'illegal_unicode', 'exploitation',
       'bad_contract', 'reusing_state_variable', 'encode_packed_collision',
       'encode_packed_parameters', 'centralized_risk_medium',
       'centralized_risk_high', 'centralized_risk_low', 'event_setter',
       'external_dependencies', 'immutable_states',
       'reentrancy_without_eth_transfer', 'incorrect_inheritance_order',
       'shadowing_local', 'events_maths']

frequencies = df[risk_columns].apply(lambda x: x.value_counts()).loc[True]
frequencies = frequencies.fillna(0)  # Replace NaN with 0 for any column that may not have True values
frequencies

sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))
sns.barplot(x=frequencies.index, y=frequencies.values, palette='viridis')
plt.title('Frequency of True Values for Each Risk Tag')
plt.xlabel('Risk Tags')
plt.ylabel('Frequency of True')
plt.xticks(rotation=45)
plt.show()
```
<img src="https://github.com/user-attachments/assets/43de87e6-57a2-4a1d-8941-428434d0a6be" alt="Alt Text" width="450" height="300">
We perform a frequency analysis of risk tags in smart contracts to identify how often specific vulnerabilities occur across a dataset. By analyzing features such as is_airdrop_scam, can_take_back_ownership, and Is_honeypot, we can uncover common security risks and gain insight into the overall risk landscape of smart contract ecosystems. This analysis not only highlights the most prevalent issues but also helps guide future auditing efforts, inform risk mitigation strategies, and lay the groundwork for further research or machine learning applications in contract security.

#Clustering
```
data = pd.read_excel('compiled_risk_data.xlsx')
data.head()
#selecting features
data_new = data.copy()
feature_1 = 'bad_contract'
feature_2 = 'external_dependencies'
feature_3 = 'exploitation'
selected_features = data_new[[feature_1, feature_2,feature_3]].replace({True:1, False:0})
print("Features selected for clustering:")
print(selected_features.head())
from scipy.spatial.distance import pdist, squareform

distance_matrix = pdist(selected_features, 'jaccard')
distance_square_matrix = squareform(distance_matrix)  # Convert to square matrix

import scipy.cluster.hierarchy as sch

# Creating linkage matrix
linkage_matrix = sch.linkage(distance_matrix, method='ward')
print(linkage_matrix)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Jaccard distance')
plt.show()

# Example: Set maximum distance at 1.5 for cluster formation
cluster_labels = fcluster(linkage_matrix, t=5, criterion='distance')

# Add cluster labels back to your original DataFrame
data_new['cluster'] = cluster_labels

# Summary statistics for each cluster
cluster_summary = data_new[[feature_1, feature_2, feature_3,'cluster']].groupby('cluster').agg(['mean', 'std', 'median', 'count'])
cluster_summary

plt.figure(figsize=(8, 6))
plt.hist(cluster_labels, bins=np.arange(1, np.max(cluster_labels)+2)-0.5, rwidth=0.8, color='blue', alpha=0.7)
plt.title('Histogram of Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Number of Points')
plt.xticks(np.arange(1, np.max(cluster_labels)+1))
plt.show()

# Calculating the mean for each cluster and feature
cluster_centers = data_new[[feature_1, feature_2, feature_3,'cluster']].groupby('cluster').mean()

plt.figure(figsize=(12, 8))
sns.heatmap(cluster_centers, annot=True, cmap='coolwarm')
plt.title('Heatmap of Cluster Centroids')
plt.show()
```

<img src="https://github.com/user-attachments/assets/5148486f-4c8b-498b-8af5-ebcb39a27f8a" alt="Alt Text" width="450" height="300">

<img src="https://github.com/user-attachments/assets/60e73f27-4b91-41aa-8e77-f1af7bf92fdd" alt="Alt Text" width="450" height="300">

<img src="https://github.com/user-attachments/assets/f884f33c-c747-4a94-8ae5-0a2914d8730d" alt="Alt Text" width="450" height="300">

We use hierarchical clustering to analyze patterns among smart contract risks by grouping contracts based on shared vulnerabilities. Using binary features like bad_contract, external_dependencies, and exploitation, it calculates pairwise similarity using Jaccard distance and visualizes the relationship between contracts through a dendrogram. The goal is to uncover natural groupings in the data—helping to identify clusters of contracts that exhibit similar risk profiles. This kind of analysis is useful for categorizing contracts, prioritizing audits, and discovering systemic issues that may affect multiple contracts in similar ways.

# Findings

From our frequency analysis, we saw that some smart contract risk tags like is_airdrop_scam, exploitation, and bad_contract showed up a lot more than others. The bar graph helped visualize this and made it easier to spot which risks might need more attention in the future. After that, we used hierarchical clustering with Jaccard distance on a few of the key features to try and find patterns in the data. The dendrogram we created showed some groups forming naturally based on the similarity of risk traits. We split the contracts into 5 clusters, and the histogram showed that one cluster had way more data points than the rest. Then we made a heatmap to see what each cluster was mainly made up of. For example, one group had mostly exploitation, another had a lot of external_dependencies. This helped us get a better idea of how different smart contracts might share common weaknesses


