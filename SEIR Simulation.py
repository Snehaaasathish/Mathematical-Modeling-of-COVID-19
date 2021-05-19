

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
import plotly.io as pio
import requests
from lmfit import minimize, Parameters, Parameter, report_fit


plt.style.use('ggplot')

def ode_model(z, t, beta, sigma, gamma):
    """
    Reference https://www.idmod.org/docs/hiv/model-seir.html
    """
    S, E, I, R = z
    N = S + E + I + R
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I
    dRdt = gamma*I
    return [dSdt, dEdt, dIdt, dRdt]

def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN = initial_conditions
    beta, sigma, gamma = params['beta'].value, params['sigma'].value, params['gamma'].value
    initS = initN - (initE + initI + initR)
    res = odeint(ode_model, [initS, initE, initI, initR], t, args=(beta, sigma, gamma))
    return res

response = requests.get('https://api.rootnet.in/covid19-in/stats/history')
print('Request Success? {}'.format(response.status_code == 200))
covid_history = response.json()['data']

keys = ['day', 'total', 'confirmedCasesIndian', 'confirmedCasesForeign', 'confirmedButLocationUnidentified',
        'discharged', 'deaths']
df_covid_history = pd.DataFrame([[d.get('day'), 
                                  d['summary'].get('total'), 
                                  d['summary'].get('confirmedCasesIndian'), 
                                  d['summary'].get('confirmedCasesForeign'),
                                  d['summary'].get('confirmedButLocationUnidentified'),
                                  d['summary'].get('discharged'), 
                                  d['summary'].get('deaths')] 
                                 for d in covid_history],
                    columns=keys)
df_covid_history = df_covid_history.sort_values(by='day')
df_covid_history['infected'] = df_covid_history['total'] - df_covid_history['discharged'] - df_covid_history['deaths']
df_covid_history['total_recovered_or_dead'] = df_covid_history['discharged'] + df_covid_history['deaths']


# ref: https://www.medrxiv.org/content/10.1101/2020.04.01.20049825v1.full.pdf
initN = 1380000000
# S0 = 966000000
initE = 1000
initI = 47
initR = 0
sigma = 1/5.2
gamma = 1/2.9
R0 = 4
beta = R0 * gamma
days = 400

params = Parameters()
params.add('beta', value=beta, min=0, max=10)
params.add('sigma', value=sigma, min=0, max=10)
params.add('gamma', value=gamma, min=0, max=10)

def main(initE, initI, initR, initN, beta, sigma, gamma, days, param_fitting):
    initial_conditions = [initE, initI, initR, initN]
    params['beta'].value, params['sigma'].value,params['gamma'].value = [beta, sigma, gamma]
    tspan = np.arange(250, days, 1)
    sol = ode_solver(tspan, initial_conditions, params)
    S, E, I, R = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]
    fig = go.Figure()
    if not param_fitting:
        fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines+markers', name='Susceptible'))
        fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines+markers', name='Exposed'))
        fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines+markers', name='Infected'))
        fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines+markers',name='Recovered'))
    if param_fitting:
        fig.add_trace(go.Scatter(x=tspan, y=df_covid_history.infected, mode='lines+markers',name='Infections Observed', line = dict(dash='dash')))
        fig.add_trace(go.Scatter(x=tspan, y=df_covid_history.total_recovered_or_dead, mode='lines+markers',name='Recovered/Deceased Observed', line = dict(dash='dash')))
        
    if days <= 30:
        step = 1
    elif days <= 90:
        step = 7
    else:
        step = 30
    
    # Edit the layout
    fig.update_layout(title='Simulation of SEIR Model',xaxis_title='Day',yaxis_title='Counts',title_x=0.5,width=900, height=600)
    fig.update_xaxes(tickangle=-90, tickformat = None, tickmode='array', tickvals=np.arange(0, days + 1, step))
    if not os.path.exists("images"):
      fig.write_image("images/seir_simulation.png")
    fig.show()
    
def error(params, initial_conditions, tspan, data):
    sol = ode_solver(tspan, initial_conditions, params)
    return (sol[:, 2:4] - data).ravel()

initial_conditions = [initE, initI, initR, initN]
beta = 1.08
sigma = 0.02
gamma = 0.02
params['beta'].value = beta
params['sigma'].value = sigma
params['gamma'].value = gamma
days = 428
tspan = np.arange(0, days, 1)
data = df_covid_history.loc[0:(days-1), ['infected', 'total_recovered_or_dead']].values

print(params)

result = minimize(error, params, args=(initial_conditions, tspan, data), method='leastsq')

print(result.params)

report_fit(result)

final = data + result.residual.reshape(data.shape)
# fig = plt.figure()
# fig.add_trace(go.Scatter(x=tspan, y=data[:, 0], mode='markers', name='Observed Infections', line = dict(dash='dot')))
# fig.add_trace(go.Scatter(x=tspan, y=data[:, 1], mode='markers', name='Observed Recovered/Deceased', line = dict(dash='dot')))
# fig.add_trace(go.Scatter(x=tspan, y=final[:, 0], mode='lines+markers', name='Fitted Infections'))
# fig.add_trace(go.Scatter(x=tspan, y=final[:, 1], mode='lines+markers', name='Fitted Recovered/Deceased'))
# fig.update_layout(title='Observed vs Fitted',xaxis_title='Day',yaxis_title='Counts',title_x=0.5,width=1000, height=600)

# fig.plot()

x = tspan
y1 = data[:,0]
y2 = data[:, 1]
y3 = final[:, 0]
y4 = final[:, 1]
plt.plot(x,y1,'r--',label = 'Observed Infections')
plt.plot(x,y3,'g--',label = 'Fitted Infections')
plt.plot(x,y2,color='lightcoral',label = 'Observed Recovered/Deceased')
plt.plot(x,y4,color='#4b0082',label = 'Fitted Recovered/Deceased')
plt.title('SEIR SIMULATION')
plt.legend()
plt.show()

# plt.plot(x,y2,y4)
# plt.title('Observed Recovered/Deceased Vs Fitted Recovered/Deceased')
