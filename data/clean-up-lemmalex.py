# Nikita Sossounov 2022
import pandas as pd

df = pd.read_csv('lemmalex-with-trace-pronunciations.csv')
df = df.drop_duplicates(subset=['Item'],keep=False)
df['Length'] = df['Phonology'].str.len()

df.to_csv('lemmalex-with-trace-pronunciations-unique.csv')