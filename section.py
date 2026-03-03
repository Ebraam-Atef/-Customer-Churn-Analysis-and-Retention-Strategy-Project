import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('births.csv')

df['decade'] = 10 * (df['year'] // 10)

df.query('decade >= 1880')

pivot = df.pivot_table(
    values='births',
    index='decade',
    columns='gender',
    aggfunc='sum'
)

pivot.plot()

plt.ylabel('Number of Births')
plt.title('Births by Gender per Decade (1980+)')
plt.show()
