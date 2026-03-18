#!/usr/bin/env python
# coding: utf-8

# In[148]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# In[165]:


df=pd.read_csv('Logic.csv')
df


# In[166]:


model=LogisticRegression()
model.fit(df[['hours_studied','attendance_percent','previous_score']],df['pass_exam'])


# In[177]:


Predicted_Values=model.predict(df[['hours_studied','attendance_percent','previous_score']])
df['pass_probability']=Predicted_Values
df
Values= model.score(df[['hours_studied','attendance_percent','previous_score']],df['pass_exam'])
import streamlit as st
st.subheader("Model Score")
Values



# In[158]:


model.predict([[4.4,93.5,81.9]])


# In[72]:


model.predict([[1.5,45.7,64.5]])


# In[117]:


model.predict([[0.4,54,74]])


# In[118]:


model.predict([[4.5,54,74]])


# In[74]:


model.predict([[0.5,54,74]])


# In[119]:


model.predict([[7.5,54,22.5]])


# In[120]:


model.score(df[['hours_studied','attendance_percent','previous_score']],df['pass_exam'])


# In[160]:


Data={'hours_studied':[6.7, 2.1, 5.2, 3.4, 8.4, 0.5, 5.7, 4.5, 3.3, 0.1],
     'attendance_percent':[78.5, 45.4 , 66.5, 32.5, 95.4, 39.6, 45.6, 77.4, 78.6, 33.5],
     'previous_score':[74.6, 57.4, 83.5, 44.5, 94.7, 33.5, 44.5, 63.5, 31.5, 95.5]}

df1=pd.DataFrame(Data)
df1


# In[161]:


Predicted_Result= model.predict(df1[['hours_studied','attendance_percent','previous_score']])
# score1=model.score(df1[['hours_studied','attendance_percent','previous_score']],Predicted_Result)
# score1
df1['Predicted Result']= Predicted_Result


# In[162]:


model.score(df1[['hours_studied','attendance_percent','previous_score']],Predicted_Result)


# In[175]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# Show predictions
st.title("Students Dashboard")
st.subheader("Predictions")
st.table(df1)
st.title("Model Score")
score=model.score(df1[['hours_studied','attendance_percent','previous_score']],df1['Predicted Result'])
st.write(score)

#Graph
fig,ax=plt.subplots()
ax.scatter(df1['attendance_percent'],df1['previous_score'])
ax.plot(df1['attendance_percent'],df1['previous_score'])
ax.set_xlabel("Attendence_Percentage")
ax.set_ylabel("Previous_score")

st.pyplot(fig)




# In[ ]:




