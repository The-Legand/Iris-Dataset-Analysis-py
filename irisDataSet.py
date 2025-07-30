import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
pd.set_option('display.width', None)

iris = load_iris(as_frame=True)

df = iris.frame
df['iris name'] = df['target'].apply (lambda i: iris.target_names[i])



# סעיף א- הוספת עמודה עם sepal Length מעוגל
rounded = []

for i in range(df.shape[0]):
    value = df.loc[i, 'sepal length (cm)']
    rounded.append(round(value))

df['Sepal length rounded'] = rounded


# סעיף ב - הוספת עמודה עם האות הראשונה או האחורנה בהתאם ל-petal length (אם הוא גדול מהמוצע לאותו סוג או קטן)
grouped_by_name = df.groupby('iris name')
petal_length_by_group = grouped_by_name['petal length (cm)']

avg_of_petal_lengths_by_type = petal_length_by_group.mean()
def choose_letter(row):
    species = row['iris name']
    petal_length = row['petal length (cm)']
    average_length = avg_of_petal_lengths_by_type[species]

    if petal_length > average_length:
        return species[0]
    else:
        return species[-1]

df['Name letter'] = df.apply(choose_letter, axis=1)

#סעיף ג -
plt.style.use('ggplot')
plt.hist(df['petal length (cm)'], bins=10, color='#5DADE2', edgecolor='black')

# Add labels and title
plt.title('Histogram of Petal Length')
plt.xlabel('petal length (cm)')
plt.ylabel('Number of Flowers')

plt.show()



print(df.head(145))
print(df.head(145))
