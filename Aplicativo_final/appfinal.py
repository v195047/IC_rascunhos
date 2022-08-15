import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tensorflow import keras, Tensor
from sklearn.metrics import r2_score, max_error
from sklearn.metrics import mean_squared_error as MSE 
from auxiliares import *
import sanfis
from sanfis import SANFIS
from sanfis import plottingtools

st.write(""" 
    # Modelagem Empírica de um processo de fermentação alcoólica utilizando técnicas de inteligência artificial.

    Aluno: Vinicius Mello e Muller  
    Professor: Dr. Flávio V. Silva  
    
    ---
    """)

#imagem 

PATH_TO_IMAGE = 'Imagens/Reator.jpg'
st.image(PATH_TO_IMAGE, "Figura ilustrativa do reator contínuo de produção de etanol")

#Dados de teste

Tempo_de_simulacao = st.number_input("tempo de simulação (h):", min_value=0, step=1, value=500)
Qi = st.slider(label='Vazão de Entrada (L/h):', min_value=0.057000, max_value=0.108600, step=0.001, format="%.3f", value=0.084820)
Vi = st.slider(label='Volume Inicial do Reator (L):', min_value=1.307874, max_value=4.537529, step=0.001, format='%.2f', value=2.853007)
Si = st.slider(label='Concentração de Substrato Inicial (g/L):', min_value=50.0, max_value=150.0, step=1.0, value=102.0)
Xi = st.slider(label='Concentração de Células Inicial (g/L):', min_value=0.025, max_value=0.075, step=0.001, format="%.3f", value=0.0515)

# Variaveis auxiliares
Mu = 0.051 # h^-1
Kd = 0.005 # h^-1
Ks = 1.9 # g/L
Kp = 20.650 # g/L
Ksl = 112.51 # g/L
Cv = 1.0110388566223264e-06 # dm^2
g = 9.81*10*3600**2 # dm/h^2
At = 1 # dm^2
ms = 5.13
YXS = 0.072
YPS = 0.369

def F(y,t):
    V=y[0]
    X=y[1]
    S=y[2]
    P=y[3]
    Qo = Cv*pow(2*g*V/At, 0.5)
    
    mu = Mu*S/(Ks+S)*Kp/(Kp+P)*Ksl/( Ksl+S)
    alpha = YPS/YXS
    beta = YPS*ms
    rx = mu*X
    rd = Kd*X
    rp = alpha*rx+beta*X
    
    # Derivadas
    dVdt = Qi -Qo
    dXdt = Qi/V*(Xi -X)+rx -rd
    dPdt = -Qi/V*P + rp
    Rxs = rx / YXS
    Rps = rp / YPS
    Rms = ms*X
    dSdt = Qi/V*(Si -S) -(Rxs+Rms+Rps)
    dydt = [dVdt , dXdt , dSdt , dPdt]
    return dydt

tempo = np.arange(0, Tempo_de_simulacao, 1)
sol = odeint(F, y0=[Vi, Xi, Si, 1.4], t=tempo)
planta_teste = DataFrame(sol)

V = sol[:, 0]
X = sol[:, 1]
S = sol[:, 2]
P = sol[:, 3]

# plt.rc("fontsize", 8)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)

ax1.plot(tempo, V, label=r'$V$', c='k')
ax1.legend(loc=4, fontsize=8)


ax2.plot(tempo, X, label=r'$X$', c='k')
ax2.legend(loc=4, fontsize=8)

ax3.plot(tempo, S, label=r"$S$", c='k')
ax3.legend(loc=4, fontsize=8)

st.pyplot(fig)

fig2, axP = plt.subplots(nrows=1, ncols=1)
axP.set_title("Predição da concentração de etanol")
axP.set_ylabel("Concentração (g/L)")

#############

#Scaler
sc = pickle.load(open("modelos/Scaler.p", "rb"))     #Usado no conjunto completo
scP = pickle.load(open("modelos/scalerP.p", "rb"))   #Usado para o produto somente
sc3 = pickle.load(open("modelos/RNNscaler.p", "rb")) #Usado para a RNN

#MODELOS
ANFIS = pickle.load(open("modelos/ANFIS.p", "rb"))
SLFN = keras.models.load_model("modelos/MLP")
RNN = keras.models.load_model("modelos/RNN")

PLANTA = series_to_supervised(planta_teste, n_in=1, n_out=1)
P_mat = PLANTA.iloc[:, -1]

P = torch.Tensor(PLANTA.iloc[:, -1].values).reshape(-1,1)
PLANTA_n = sc.transform(PLANTA)
x_normalizado = torch.Tensor(PLANTA_n[:, :-4])
x_RNN, _ = converter_dados(sc3.transform(planta_teste), 2)

P_ANFIS = scP.inverse_transform(ANFIS.predict(x_normalizado))
P_MLP = scP.inverse_transform(SLFN.predict(np.array(x_normalizado)))
P_RNN = sc3.inverse_transform(RNN.predict(x_RNN))[:, -1]


axP.plot(tempo[1:], P_mat, label=r"Modelo Matemático", c='k', ls='--')
axP.plot(tempo[1:], P_ANFIS, label=r"ANFIS", c='orange', ls='-')
axP.plot(tempo[1:], P_MLP, label=r"MLP", c='r', ls='-')
axP.plot(tempo[2:], P_RNN, label=r"RNN", c='b', ls='-')

axP.legend(loc=4, fontsize=7, ncol=4)
st.pyplot(fig2)
##########################################################

resultados={
    "MSE":[MSE(P_mat, P_ANFIS), MSE(P_mat, P_MLP), MSE(P_mat[1:], P_RNN)],
    "R2":[r2_score(P_mat, P_ANFIS), r2_score(P_mat, P_MLP), r2_score(P_mat[1:], P_RNN)],
    "Erro Max":[max_error(P_mat, P_ANFIS), max_error(P_mat, P_MLP), max_error(P_mat[1:], P_RNN)]
}

resultados = pd.DataFrame(resultados, index=["ANFIS", "MLP", "RNN"])
st.dataframe(resultados, width=2000)