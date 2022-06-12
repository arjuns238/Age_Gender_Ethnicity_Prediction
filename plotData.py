#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


# In[7]:


data = pd.read_csv('age_gender.csv')
df = pd.DataFrame(data)
plt.xlabel = ' Gender (1 = Female, 0 = Male) '
print("Length of row 0 = ", len(np.array(df['pixels'][0].split())))

plt.figure(figsize=(10,7))
ax = df.gender.value_counts().plot.bar(x = "Gender (1 = Female, 0 = Male)", y = "count", title = "Gender", legend = (1,0, ('Female', 'Male')))
plt.figure(figsize=(10,7))
labels =['White','Black','Indian','Asian','Hispanic']
ax = df.ethnicity.value_counts().plot.bar(title = "Ethnicity")
ax.set_xticklabels(labels)


#Converting each row of pixels to a numpy array
df['pixels'] = df['pixels'].apply(lambda x:  np.reshape(np.array(x.split(), dtype="float32"), (48, 48, 1)))

    
    

    


# In[5]:


gender_values_to_labels = ['Male', 'Female']
ethnicity_values_to_labels = ['White','Black','Indian','Asian','Hispanic']

def plot(row, col, lb, ub):
    fig = plt.figure(figsize=(col * 3, row * 4))
    for i in range(1, col * row + 1):
        k = np.random.randint(lb, ub)
        fig.add_subplot(row, col, i)
        gender = gender_values_to_labels[df.gender[k]]
        ethnicity = ethnicity_values_to_labels[df.ethnicity[k]]
        age = df.age[k]
        im = df.pixels[k]
        plt.imshow(im)
        plt.axis('off')
        plt.title(f'Gender:{gender}\nAge:{age}\nEthnicity:{ethnicity}')
        plt.tight_layout()
        plt.show()
plot(1, 7, 0, len(df))


# In[2]:


from PIL import Image
import matplotlib.pyplot as plt
plt.imshow(img)

