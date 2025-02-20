#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Données de vente pour 2018
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_2018 = np.array([
    11860227.23, 11861671.49, 11859278.43, 11860096.13,
    45000000.00, 39000000.00, 54000000.00, 62000000.00,
    11859936.07, 11861742.56, 11861917.10, 11860869.63
])

# Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X, Y_2018)

# Obtenir les paramètres
beta_1 = model.coef_[0]
beta_0 = model.intercept_
print("Pente (β1):", beta_1)
print("Ordonnée à l'origine (β0):", beta_0)

# Prévoir les ventes pour les 12 mois suivants
X_future = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_pred_2018 = model.predict(X_future)
print("Prévisions pour 2018:", Y_pred_2018)

# Visualisation des résultats
plt.scatter(X, Y_2018, color='blue', label='Données réelles')
plt.plot(X_future, Y_pred_2018, color='red', label='Prévisions')
plt.xlabel('Mois')
plt.ylabel('Ventes')
plt.title('Régression Linéaire Simple - Ventes 2018')
plt.legend()
plt.show()


# In[26]:


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Données de vente pour 2019
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_2019 = np.array([
    12409357.24, 12429686.37, 12419702.92, 12419548.73,
    47329309.40, 42070473.04, 57590856.77, 66609752.18,
    12418585.90, 12419312.87, 12420739.79, 12419776.65
])

# Créer et entraîner le modèle de régression linéaire
model_2019 = LinearRegression()
model_2019.fit(X, Y_2019)

# Obtenir les paramètres
beta_1_2019 = model_2019.coef_[0]
beta_0_2019 = model_2019.intercept_
print("Pente (β1):", beta_1_2019)
print("Ordonnée à l'origine (β0):", beta_0_2019)

# Prévoir les ventes pour les 12 mois suivants
X_future = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_pred_2019 = model_2019.predict(X_future)
print("Prévisions pour 2019:", Y_pred_2019)

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.scatter(X, Y_2019, color='blue', label='Données réelles 2019')
plt.plot(X_future, Y_pred_2019, color='red', label='Prévisions 2019')
plt.xlabel('Mois')
plt.ylabel('Ventes')
plt.title('Régression Linéaire Simple - Ventes 2019')
plt.legend()
plt.grid(True)
plt.show()


# In[29]:


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Données de vente pour 2020
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_2020 = np.array([
    13146000.25, 13152987.17, 13149556.46, 13151344.82,
    50123456.78, 44567890.12, 60987654.32, 70432109.54,
    13148486.09, 13145358.23, 13153122.62, 13149933.06
])

# Créer et entraîner le modèle de régression linéaire
model_2020 = LinearRegression()
model_2020.fit(X, Y_2020)

# Obtenir les paramètres
beta_1_2020 = model_2020.coef_[0]
beta_0_2020 = model_2020.intercept_
print("Pente (β1):", beta_1_2020)
print("Ordonnée à l'origine (β0):", beta_0_2020)

# Prévoir les ventes pour les 12 mois suivants
X_future = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_pred_2020 = model_2020.predict(X_future)
print("Prévisions pour 2020:", Y_pred_2020)

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.scatter(X, Y_2020, color='blue', label='Données réelles 2020')
plt.plot(X_future, Y_pred_2020, color='red', label='Prévisions 2020')
plt.xlabel('Mois')
plt.ylabel('Ventes')
plt.title('Régression Linéaire Simple - Ventes 2020')
plt.legend()
plt.grid(True)
plt.show()


# In[30]:


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Données de vente pour chaque année
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_2018 = np.array([
    11860227.23, 11861671.49, 11859278.43, 11860096.13,
    45000000.00, 39000000.00, 54000000.00, 62000000.00,
    11859936.07, 11861742.56, 11861917.10, 11860869.63
])
Y_2019 = np.array([
    12409357.24, 12429686.37, 12419702.92, 12419548.73,
    47329309.40, 42070473.04, 57590856.77, 66609752.18,
    12418585.90, 12419312.87, 12420739.79, 12419776.65
])
Y_2020 = np.array([
    13146000.25, 13152987.17, 13149556.46, 13151344.82,
    50123456.78, 44567890.12, 60987654.32, 70432109.54,
    13148486.09, 13145358.23, 13153122.62, 13149933.06
])

# Créer et entraîner les modèles de régression linéaire
model_2018 = LinearRegression()
model_2018.fit(X, Y_2018)

model_2019 = LinearRegression()
model_2019.fit(X, Y_2019)

model_2020 = LinearRegression()
model_2020.fit(X, Y_2020)

# Prévoir les ventes pour les 12 mois suivants
X_future = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_pred_2018 = model_2018.predict(X_future)
Y_pred_2019 = model_2019.predict(X_future)
Y_pred_2020 = model_2020.predict(X_future)

# Visualisation des résultats
plt.figure(figsize=(12, 8))
plt.scatter(X, Y_2018, color='blue', label='Données 2018')
plt.scatter(X, Y_2019, color='green', label='Données 2019')
plt.scatter(X, Y_2020, color='orange', label='Données 2020')
plt.plot(X_future, Y_pred_2018, color='blue', linestyle='--', label='Prévisions 2018')
plt.plot(X_future, Y_pred_2019, color='green', linestyle='--', label='Prévisions 2019')
plt.plot(X_future, Y_pred_2020, color='orange', linestyle='--', label='Prévisions 2020')
plt.xlabel('Mois')
plt.ylabel('Ventes')
plt.title('Comparaison des Régressions Linéaires Simples - Ventes 2018, 2019, 2020')
plt.legend()
plt.grid(True)
plt.show()


# In[31]:


import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Données de vente pour chaque année
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_2018 = np.array([
    11860227.23, 11861671.49, 11859278.43, 11860096.13,
    45000000.00, 39000000.00, 54000000.00, 62000000.00,
    11859936.07, 11861742.56, 11861917.10, 11860869.63
])
Y_2019 = np.array([
    12409357.24, 12429686.37, 12419702.92, 12419548.73,
    47329309.40, 42070473.04, 57590856.77, 66609752.18,
    12418585.90, 12419312.87, 12420739.79, 12419776.65
])
Y_2020 = np.array([
    13146000.25, 13152987.17, 13149556.46, 13151344.82,
    50123456.78, 44567890.12, 60987654.32, 70432109.54,
    13148486.09, 13145358.23, 13153122.62, 13149933.06
])

# Créer et entraîner les modèles de régression linéaire
model_2018 = LinearRegression()
model_2018.fit(X, Y_2018)

model_2019 = LinearRegression()
model_2019.fit(X, Y_2019)

model_2020 = LinearRegression()
model_2020.fit(X, Y_2020)

# Prévoir les ventes pour les 12 mois suivants
X_future = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
Y_pred_2018 = model_2018.predict(X_future)
Y_pred_2019 = model_2019.predict(X_future)
Y_pred_2020 = model_2020.predict(X_future)

# Visualisation des résultats
plt.figure(figsize=(12, 8))
plt.plot(X, Y_2018, color='blue', label='Données 2018', linestyle='-')
plt.plot(X, Y_2019, color='green', label='Données 2019', linestyle='-')
plt.plot(X, Y_2020, color='orange', label='Données 2020', linestyle='-')
plt.plot(X_future, Y_pred_2018, color='blue', linestyle='--', label='Prévisions 2018')
plt.plot(X_future, Y_pred_2019, color='green', linestyle='--', label='Prévisions 2019')
plt.plot(X_future, Y_pred_2020, color='orange', linestyle='--', label='Prévisions 2020')
plt.xlabel('Mois')
plt.ylabel('Ventes')
plt.title('Comparaison des Régressions Linéaires Simples - Ventes 2018, 2019, 2020')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




