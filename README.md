# EX-06 FEATURE TRANSFORMATION
### Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
### Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
### Algorithm:
- Step1: Read the given Data.
- Step2: Clean the Data Set using Data Cleaning Process.
- Step3: Apply Feature Transformation techniques to all the features of the data set.
- Step4: Print the transformed features.
### Program:
## Code:
  ```Python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import statsmodels.api as sm
  import scipy.stats as stats
  from sklearn.preprocessing import QuantileTransformer
  from sklearn.preprocessing import PowerTransformer
  df=pd.read_csv("Data_to_Transform.csv")

  df.head()
  df.info()
  df

  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()

  sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
  plt.title("Highly Negative Skew")
  plt.show()

  sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
  plt.title("Moderate Negative Skew")
  plt.show()
  
  df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  
  df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
  sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()

  df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()

  df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
  sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  transformer=PowerTransformer("yeo-johnson")
  df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate Negative Skew")
  plt.show()

  qt = QuantileTransformer(output_distribution = 'normal')
  df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate  Negative Skew")
  plt.show()
  
  ```
## Basic Information:

  ![1](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/8174e1f4-d724-436b-8f33-0fa20413cddc)
  ![2](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/a0687a96-21ff-44e2-a19a-a2aaf7d6753d)
  ![3](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/132a6dd0-f447-4907-b2b9-7f342e3a8ca4)

## Before Transformation:
 
  ![4](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/eac11f2a-51cd-4eb0-a348-757a01b48102)
  ![5](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/63728e2c-5c24-49f5-9865-014c1335304d)
  ![6](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/53cdbe49-62be-4158-96f8-cf7d3f1f94ea)
  ![7](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/1f2338b7-8976-4a38-a2fe-a76f99b4ff0b)

## Log Transformation:
  
  ![8](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/567a14c2-975d-4022-ae7e-48e9643d67b8)
  ![9](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/91cb3661-adb6-4726-901a-9453db9ec65c)

## Reciprocal Transformation:
 
 ![10](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/5e40b4fc-cd70-435b-ab0f-55a6df660225)

## SquareRoot Transformation:
  
![11](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/9adbe3af-b0b8-41e7-880f-a29fb963951f)

## Power Transformation:
  
  ![12](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/1fc1a6f8-9403-496d-bdb4-6ec5e24d250e)
![13](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/fdde097f-af98-4bfa-bcc7-1f33b5672a6e)


  
## Quantile Transformation:
![14](https://github.com/Aakash0407/ODD2023-Datascience-Ex06/assets/118799103/00458c10-dacb-4fed-9f0f-34a411f7977d)

### Result:  
Thus, the feature transformation is done for the given dataset - Data_to_Transform.csv.
