
# coding: utf-8

# In[8]:


import os
dir_path = '/home/greghovhannisyan/Downloads/180208/'
counter = 0

for root, dirs, files in os.walk(dir_path): 
    for file in files:
        if file.endswith(".fasta"):
            counter += 1
print(counter)


# In[4]:


for file in os.listdir(dir_path):
    print(file)

