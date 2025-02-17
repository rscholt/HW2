import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read in data
hcris_data=pd.read_csv('data/output/HCRIS_Data.csv')

# Count hospitals that filed more than one report in the same year
hospital_counts = hcris_data.groupby(['provider_number', 'year']).size().reset_index(name='report_count')
multiple_reports = hospital_counts[hospital_counts['report_count'] > 1]
# Count number of hospitals per year
hospitals_over_time = multiple_reports.groupby('year')['provider_number'].nunique()

# Plot the results
plt.figure()
plt.plot(hospitals_over_time.index, hospitals_over_time.values)
plt.xlabel('Year')
plt.ylabel('Number of Hospitals')
plt.title('Hospitals Filing More Than One Report Per Year Over Time')
plt.grid(True)
plt.show()

#remove duplicates
unique_hospitals = hcris_data.drop_duplicates(subset=['provider_number', 'year'])
#count unique hospitals
unique_hospital_count = unique_hospitals['provider_number'].nunique()
print("Count of unique hospital IDs:", unique_hospital_count)

#total charges per year  cleaning messed up total charges somehow
years = sorted(hcris_data['year'].dropna().unique())
df_tot_charges = [hcris_data[hcris_data['year'] == y]['tot_charges'] for y in years]
# Plot
plt.figure(figsize=(12, 6))
plt.violinplot(df_tot_charges, showmedians=True)
plt.xlabel('Year')
plt.ylabel('Total Charges ($)')
plt.title('Total Charges Distribution by Year')

# Compute estimated price using the given formula
hcris_data['discount_factor'] = 1 - hcris_data['tot_discounts'] / hcris_data['tot_charges']
hcris_data['price_num'] = (hcris_data['ip_charges'] + hcris_data['icu_charges'] + hcris_data['ancillary_charges']) * hcris_data['discount_factor'] - hcris_data['tot_mcare_payment']
hcris_data['price_denom'] = hcris_data['tot_discharges'] - hcris_data['mcare_discharges']
hcris_data['estimated_price'] = hcris_data['price_num'] / hcris_data['price_denom']

# Remove negative and extreme outlier values
# Calculate mean and standard deviation
mean_price = hcris_data['estimated_price'].mean()
std_price = hcris_data['estimated_price'].std()

# Define lower and upper bounds for outlier removal (3 standard deviations from mean)
lower_bound = mean_price - (3 * std_price)
upper_bound = mean_price + (3 * std_price)

# Remove negative values
hcris_data = hcris_data[hcris_data['estimated_price'] > 0]

# Remove outliers
hcris_data = hcris_data[(hcris_data['estimated_price'] >= lower_bound) & (hcris_data['estimated_price'] <= upper_bound)]

estimated_price_df = [hcris_data[hcris_data['year'] == y]['estimated_price'] for y in years]
ep_years = sorted(hcris_data['year'].dropna().unique())

# Plot
plt.figure()
plt.violinplot(estimated_price_df, showmedians=True)
plt.xticks(range(1, len(ep_years) + 1), ep_years, rotation=45)
plt.xlabel('Year')
plt.ylabel('Estimated Price ($)')
plt.title('Estimated Price Distribution by Year')
plt.yscale('log')  # Log scale for better visualization
plt.show()

# Filter data for the year 2012
hcris_2012 = hcris_data[hcris_data['year'] == 2012].copy()

# Define penalty: If the sum of HRRP and HVBP payments is negative, it's a penalty
hcris_2012['penalty'] = (hcris_2012['hrrp_payment'] + hcris_2012['hvbp_payment']) < 0

# Calculate estimated price using the given formula
hcris_2012['discount_factor'] = 1 - (hcris_2012['tot_discounts'] / hcris_2012['tot_charges'])
hcris_2012['price_num'] = (hcris_2012['ip_charges'] + hcris_2012['icu_charges'] + hcris_2012['ancillary_charges']) * hcris_2012['discount_factor'] - hcris_2012['tot_mcare_payment']
hcris_2012['price_denom'] = hcris_2012['tot_discharges'] - hcris_2012['mcare_discharges']
hcris_2012['estimated_price'] = hcris_2012['price_num'] / hcris_2012['price_denom']

# Compute the average price for penalized vs. non-penalized hospitals
avg_price_by_penalty = hcris_2012.groupby('penalty')['estimated_price'].mean()

#________


# # Assume `hcris_2012_b` contains the dataset with `penalty`, `estimated_price`, and `beds`
# hcris_2012_b = hcris_2012.copy()

# # **Step 1: Create Bed Size Quartiles**
# hcris_2012_b['bed_quartile'] = pd.qcut(hcris_2012_b['beds'], q=4, labels=[1, 2, 3, 4])

# # **Step 2: Nearest Neighbor Matching (Inverse Variance Distance)**
# bed_var = hcris_2012_b.groupby('bed_quartile')['beds'].var()
# hcris_2012_b['variance_weight'] = hcris_2012_b['bed_quartile'].map(bed_var)
# hcris_2012_b['variance_weight'] = 1 / hcris_2012_b['variance_weight']

# # Find closest match based on inverse variance distance
# hcris_2012_b_sorted = hcris_2012_b.sort_values(by=['variance_weight'])
# hcris_2012_b['matched_iv'] = hcris_2012_b_sorted['estimated_price'].shift(-1)  # Approximate 1-to-1 matching
# ate_nn_iv = (hcris_2012_b[hcris_2012_b['penalty'] == 1]['estimated_price'] - hcris_2012_b[hcris_2012_b['penalty'] == 1]['matched_iv']).mean()

# # **Step 3: Nearest Neighbor Matching (Mahalanobis Distance)**
# bed_mean = hcris_2012_b.groupby('bed_quartile')['beds'].mean()
# bed_std = hcris_2012_b.groupby('bed_quartile')['beds'].std()
# hcris_2012_b['mahal_dist'] = ((hcris_2012_b['beds'] - hcris_2012_b['bed_quartile'].map(bed_mean)) / hcris_2012_b['bed_quartile'].map(bed_std)).abs()

# # Find closest match based on Mahalanobis distance
# hcris_2012_b_sorted = hcris_2012_b.sort_values(by=['mahal_dist'])
# hcris_2012_b['matched_mahal'] = hcris_2012_b_sorted['estimated_price'].shift(-1)
# ate_nn_mahal = (hcris_2012_b[hcris_2012_b['penalty'] == 1]['estimated_price'] - hcris_2012_b[hcris_2012_b['penalty'] == 1]['matched_mahal']).mean()

# # **Step 4: Inverse Propensity Weighting (IPW)**
# ps = hcris_2012_b.groupby('bed_quartile')['penalty'].mean()
# hcris_2012_b['propensity_score'] = hcris_2012_b['bed_quartile'].map(ps)
# hcris_2012_b['ipw_weight'] = 1 / hcris_2012_b['propensity_score']

# ate_ipw = (hcris_2012_b['ipw_weight'] * hcris_2012_b['penalty'] * hcris_2012_b['estimated_price']).mean() - (
#     hcris_2012_b['ipw_weight'] * (1 - hcris_2012_b['penalty']) * hcris_2012_b['estimated_price']
# ).mean()

# # **Step 5: Simple Linear Regression with Bed Quartiles**
# X = np.column_stack((np.ones(len(hcris_2012_b)), pd.get_dummies(hcris_2012_b['bed_quartile']), hcris_2012_b['penalty']))
# y = hcris_2012_b['estimated_price'].values

# # Solve using normal equation (OLS)
# beta = np.linalg.inv(X.T @ X) @ X.T @ y
# ate_regression = beta[-1]  # The coefficient for `penalty`

# # **Step 6: Present Results in a Table**
# ate_results = pd.DataFrame(
#     "Method": [
#         "NN Matching (Inverse Variance)",
#         "NN Matching (Mahalanobis)",
#         "Inverse Propensity Weighting",
#         "Linear Regression"
#     ],
#     "Estimated ATE": [ate_nn_iv, ate_nn_mahal, ate_ipw, ate_regression]

# need to fix cleaning issues all cells empty can't figure that out yet !!