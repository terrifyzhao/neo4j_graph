import pandas as pd
import numpy as np

df = pd.read_csv('company_data/人物.csv')
num = np.random.randint(0, 4531, 2000)
df.drop(index=num, axis=0, inplace=True)
df.to_csv('人物2.csv', index=False, encoding='utf_8_sig')

