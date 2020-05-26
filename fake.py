import pandas as pd
import numpy as np

df = pd.read_csv('company_data/公司-人物.csv')
df.drop(['personname'], axis=1, inplace=True)
eid = pd.read_csv('company_data/公司.csv')['eid'].values
pid = pd.read_csv('company_data/人物2.csv')['personcode'].values

num1 = np.random.randint(0, 609, 500)
num2 = np.random.randint(0, 609, 500)

df_new = pd.DataFrame()
df_new['eid'] = eid[num1]
df_new['personcode'] = pid[num2]
df_new['post'] = '董事'
df = df_new.append(df)

num1 = np.random.randint(0, 609, 500)
num2 = np.random.randint(0, 609, 500)

df_new = pd.DataFrame()
df_new['eid'] = eid[num1]
df_new['personcode'] = pid[num2]
df_new['post'] = '监事'
df = df_new.append(df)
df.rename(columns={'personcode': 'pid'}, inplace=True)
df.drop_duplicates(['eid', 'pid', 'post'], inplace=True)
df.to_csv('公司-人物2.csv', index=False, encoding='utf_8_sig')
