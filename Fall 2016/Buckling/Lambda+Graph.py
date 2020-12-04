
# coding: utf-8

# In[24]:

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib notebook

lam = np.array([8, 9, 9.372, 9.549, 9.646, 9.705, 9.743, 9.769])
node = np.linspace(1,8, num = 8)
actual = np.ones(8)*(np.pi**2)


plt.figure()
plt.plot(node, lam)
plt.plot(node, actual, '--k', label='actual')
plt.title('value lambda with increasing number of nodes')
plt.xlabel('Number of nodes')
plt.ylabel('Value of lambda')
plt.text(4, 9.81, 'pi^2')


# In[ ]:



