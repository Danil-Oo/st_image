import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from skimage import io


st.title('Сингулярное разложение черно-белой картинки')
st.divider()

image_url = st.text_input("Введите URL изображения:")
if image_url:
    st.image(image_url)
else:
    image_url = 'https://i.pinimg.com/736x/5e/5b/01/5e5b012a95248235ed87d0676d75916a.jpg'
    st.image(image_url)
image = io.imread(image_url)[:, :, 2]

U, sing_vals, V = np.linalg.svd(image)
sigma = np.zeros(shape=image.shape)
np.fill_diagonal(sigma, sing_vals)

top_k = st.slider(label='Выберите величину top_k',
                  min_value=1, max_value=len(sing_vals))
trunc_U = U[:, :top_k]
trunc_sigma = sigma[:top_k, :top_k]
trunc_V = V[:top_k, :]

st.subheader('Отобразим изменение картинки в зависимости от top_k')
fig, axes = plt.subplots(1, 2, figsize=(7, 5))
axes[0].imshow(U@sigma@V, cmap='grey')
axes[0].set_title('Исходное изображение')
axes[1].imshow(trunc_U@trunc_sigma@trunc_V, cmap='grey')
axes[1].set_title(f'top_k = {top_k} компонент')
st.pyplot(fig)
