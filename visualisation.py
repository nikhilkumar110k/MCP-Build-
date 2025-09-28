import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv(r"D:\BTMproject\mcpproject\dataset\medicine_dataset.csv")

print("Dataset Info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics for numerical columns:")
print(df.describe())


numerical_cols = ['Strength']  
for col in numerical_cols:
    df[col] = df[col].replace('[^\d.]', '', regex=True)  
    df[col] = pd.to_numeric(df[col], errors='coerce')
    plt.figure(figsize=(10,6))
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"Outliers detected in {col}: {len(outliers)}")
    print(outliers[[col, 'Name']])  

for col in numerical_cols:
    plt.figure(figsize=(10,6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


categorical_cols = ['Category', 'Dosage Form', 'Classification', 'Manufacturer']
for col in categorical_cols:
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, y=col, order=df[col].value_counts().index)
    plt.title(f'Counts of {col}')
    plt.show()

if len(numerical_cols) > 1:
    corr = df[numerical_cols].corr()
    print("Correlation matrix:")
    print(corr)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

    pvals = pd.DataFrame(np.zeros((len(numerical_cols), len(numerical_cols))), 
                         columns=numerical_cols, index=numerical_cols)
    for i in numerical_cols:
        for j in numerical_cols:
            if i != j:
                _, pval = stats.pearsonr(df[i], df[j])
                pvals.loc[i,j] = pval
            else:
                pvals.loc[i,j] = np.nan
    print("P-values between numerical columns:")
    print(pvals)


if len(numerical_cols) > 1:
    sns.pairplot(df[numerical_cols])
    plt.show()


plt.figure(figsize=(12,6))
sns.boxplot(x='Category', y='Strength', data=df)
plt.xticks(rotation=45)
plt.title('Strength distribution by Category')
plt.show()
