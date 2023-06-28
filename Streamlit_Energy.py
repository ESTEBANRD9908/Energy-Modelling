#### Streamlit App

### Loading packages

## Standards 
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import mstats
from scipy.stats import  kurtosis,skew, norm, lognorm, expon, gamma, beta, kstest
from scipy.stats import shapiro, probplot
import math
import numpy.random as npr
from dataclasses import dataclass
from typing import Optional

## Plots
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
#import plotly.offline as pyo
#import plotly.graph_objs as go
#import cufflinks as cf
#from IPython.display import display, Markdown

## Models
#from arch import arch_model
#from arch.univariate import EWMAVariance
#from arch.univariate import RiskMetrics2006
from sklearn.linear_model import LinearRegression


### Time
import time 
import datetime as dt

### Refinitiv Eikon
#import eikon as ek
#import configparser as cp


### Streamlit
import streamlit as st
from PIL import Image

##APP----------------------------------------------------------------
font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')


def fetch_data(path):
    data = pd.read_excel(path)
    data.set_index('Exchange Date',inplace = True)
    return data 
def fig_plot(title,x,y,data,color:Optional[str]= 'violet'):
    '''
    Function that plots the Energy prices across time
    '''
    fig, ax = plt.subplots(1,1)
    fig.suptitle(title,fontsize=12, fontweight='bold',fontproperties=font,color = 'white')
    fig.set_size_inches(7,3)
    fig.set_facecolor('#0E1117')
    ax.set_facecolor("#0E1117")
    ax.plot(data,color = color,alpha=0.5,linewidth=0.8)
    ax.set_ylabel(y,font = font, color = 'white')
    ax.set_xlabel(x,font = font, color = 'white')
    fig.autofmt_xdate(rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.tick_params(axis='x', colors='white' )   
    ax.tick_params(axis='y', colors='white')
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

### Starting App
def main():

    st.title('NL Energy Prices Modelling')
    image = Image.open('Renewable_wind.png')
    st.image(image, caption='Wind Farm')

    st.markdown('This Stremlit App, wants to show a little bit of option and price modelling for the Dutch Energy Markets.\
            We will explore, Monte Carlo simulations, GMB and OU mean-reveting modells')

    ### Getting data

    # Create a text element and let the reader know the data is loading. Comment out since is not an online element anymore

    st.markdown('First we import data from the FWD prices of NL power. These can be obtained in EEX, Bloomberg and Refinitiv.')

    #data_load_state = st.text('Loading data...')

    data = fetch_data('Copy of Price History_20230511_1126.xlsx')

    st.subheader('Raw data ')
    st.write(data.head())

    #data_load_state.text('Loading data...done!')


    ### Visualizing prices =============================================================

    st.markdown('Lets visualize the data')

    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = fig_plot('NL Energy prices from 2003 to 2023','Prices (EUR/mWh)','Year',data.Close)
    st.pyplot(fig)


    st.markdown('The recent prices show that the maket has been more volatile than it used to be. If we split the data\
            from 2013 to 2020 vs 2021 to 2023. The behaviours are completely different. On the one hand the first period\
            show a mean reverting proces, whilst the second has large and unsual spikes')

    data_fst = data[data.index < pd.to_datetime('01-01-2020', format = '%d-%m-%Y')]
    data_snd = data[data.index > data_fst.index[0] ]


    fig2 = fig_plot('NL Energy prices before 2020','Prices (EUR/mWh)','Year',data_fst.Close)
    st.pyplot(fig2)

    fig3 = fig_plot('NL Energy prices after 2020','Prices (EUR/mWh)','Year',data_snd.Close)
    st.pyplot(fig3)

    st.markdown('One can argue that prices are turning back to the first period...yet is not set in stone')

    #### Volatility plots ------------

    st.subheader('Volatility')

    st.markdown('We can try to look different methodologies for estimating volatility. First lets have a look\
            at the static stats and then we can plot it')

    st.write(data.describe())

    st.markdown(f' Historical anualized volatility {np.round(data.iloc[:,2].std()*np.sqrt(252),2)}')

    ### STD Moving average
    std_ma = data.Close.rolling(30).std()*np.sqrt(252)

    fig4 = fig_plot('Annualized Volatility of the Prices Rolling window 30 days','Year','Volatility (EUR/mWh)',std_ma)
    st.pyplot(fig4)

    st.markdown('But...wait a minute, makes more sense to look at the returns, and their volatility to later on\
            either price options and model price paths')

    ### Estimating Returns 
    returns = data.Close.pct_change().dropna()
    st.markdown('NL Power Log returns')

    fig5 = fig_plot('NL Energy returns from 2003 to 2023','Year','Returns',returns)
    st.pyplot(fig5)


    st.markdown('Lets different vola methods')

    d = {'STD 30': list(returns.rolling(30).std())}

    vola = pd.DataFrame(data = d)
    vola.set_index(returns.index,inplace= True)
    vola['STD 50']= list(returns.rolling(50).std())


    fig6, ax = plt.subplots(1,1)
    fig6.suptitle('Annualized volatility',fontsize=12, fontweight='bold',fontproperties=font)
    fig6.set_size_inches(7,3)
    fig6.set_facecolor('#0E1117')
    ax.set_facecolor("#0E1117")
    ax.plot(vola['STD 30']*np.sqrt(252),color = 'violet',alpha=0.5,linewidth=0.8, label ='STD 30')
    ax.plot(vola['STD 50']*np.sqrt(252),color = 'lightgreen',alpha=0.5,linewidth=0.8, label = 'STD 50')
    ax.set_ylabel('Vola (%)',font = font, color = 'white')
    ax.set_xlabel('Year',font = font, color = 'white')
    ax.legend(loc = 'best', prop=font)
    fig6.autofmt_xdate(rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.tick_params(axis='x', colors='white' )   
    ax.tick_params(axis='y', colors='white')
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    st.pyplot(fig6)


    st.markdown('Clearly, the last periods have affected estimates (mean and standard deviations) almost doubling them. We can incorporate more advanced techniques\
                for volatility modelling but is not the purpose of this App. In a different one, we could explore perhaps how the market behaves\
            and whether makes sense to use historical, or GARCH(1,1) or even implied volatilities. A tip is that this market in specific is quite illiquid,\
            therefore, historical volatilities help us to proxy what is happening.')

    st.header('Stochastic simulations: GMB & OU processes')

    st.markdown('Often, there is the case in which we want to model the price development over time. Typically in financial markets\
            analysts assume that assets follow a random walk (with a Geometric Browian motion). However, as showned previously, in\
            energy markets this assumption can be challenged, since where only the recent periods that affected typical behaviour.\
            If we look back, five to ten years in the past, the energy prices seemed to be quite stable, following a seasonal pattern\
            that makes sense...since in Europe, winters can be tough and energy prices can go up.')

    st.markdown('What if we then simulate n paths assuming that prices follow a GMB?')

    price_equation = "S(t) = S(0) * e^((r - 0.5 * \sigma^2)t + \sigma * W(t))"
    st.latex(price_equation)

    brownian_motion = "W(t) = \sqrt{t} \cdot Z"
    st.latex(brownian_motion)

    ### From other python files

    from Simulations import simulate_paths, Monte_Carlo

    S_GBM = simulate_paths(data.Close.iloc[0], 0.0375, vola['STD 50'].iloc[-1]*np.sqrt(252), 1, 100, 1000)


    fig7 = fig_plot('Simulation GMB','Year','Prices (EUR/mWh)',S_GBM,'gold')
    st.pyplot(fig7)

    st.markdown('As it is exhibited above, if we asume a GMB to model price, notice that the mean-reverting process is lost.\
            Lets compare it with an OU proceSs')

    ou_price_equation = "dX(t) = k \cdot (\Omega- X(t)) \, dt + \sigma \, dW(t)"
    st.latex(ou_price_equation)

    ##### OU estimation
    from OUoption import get_OU_process,OUParams,estimate_OU_params

    mean = data.Close.mean()
    std = returns.std()
    reversion_m = 2

    st.markdown(f'We can set a OU process with initial random parameters such as alpha: {np.round(std,4)}, Omega: {np.round(mean,4)}, beta: {reversion_m} and a starting price of 100 EUR/mWh. Note that Omega is the long-term mean,\
                beta the mean-reversion coefficient and alpha the standard deviation of the process')

    Initial_params = OUParams(std,mean,reversion_m)

    OU_sim = get_OU_process(100,1000,Initial_params,100,100)


    #pd.DataFrame(OU_sim).to_excel('OU_simulation.xlsx')

    fig8 = fig_plot('Simulation OU','Year','Prices (EUR/mWh)',np.transpose(OU_sim),'orange')
    st.pyplot(fig8)

    st.markdown('This example exhibits that whilst forecasting with the GBM, the OU has a mean reverting process\
            perhaps more suitable for Energy modelling. We can dive further into the analysis and estimate the\
            OU parameters based on historical data...')

    st.markdown('Fitting data equations!')

    OU_params_hat = estimate_OU_params(np.array(data.Close.dropna().values))

    alpha_initial = np.round(OU_params_hat.alpha,4)
    gamma_initial = np.round(OU_params_hat.gamma,4)
    beta_initial = np.round(OU_params_hat.beta,4)
    st.markdown(f"Initial values: alpha = {alpha_initial}, Omega = {gamma_initial}, beta = {beta_initial}")



    OU_est = get_OU_process(100,1000,OU_params_hat)

    # st.line_chart(OU_est)

    fig9 = fig_plot('OU - Simulation with estimated parameters','Year','Prices (EUR/mWh)',np.transpose(OU_est),'lime')

    st.pyplot(fig9)

    st.header('Option pricing')
    st.markdown('So... now what if we compare the option prices of varios methodologies?\
            We can compare: Black&Scholes, MonteCarlo Simulation and \
            OU-Simulation')

    st.markdown('Lets define first the initial parameters of the Options')

    x_factor = 1 #0.84 was factor for underlying 
    S = x_factor*np.array([123.4, 132.5, 121.67, 104.5, 102.59, 103.81, 100.81, 101.27, 101.4, 101.55, 105.37, 103.44, 85.34, 95.11]) ## Taken from client

    ### Time Frame:
    valuation_date = "31-03-2023"
    expiry_dates = np.array(['31-12-2023', '31-12-2024','31-12-2025','31-12-2026','31-12-2027','31-12-2028','31-12-2029','31-12-2030','31-12-2031',
    '31-12-2032','31-12-2033','31-12-2034','31-12-2035','31-12-2036'])
    T = np.array((pd.to_datetime(expiry_dates,format="%d-%m-%Y")-pd.to_datetime(valuation_date,format="%d-%m-%Y")).days/365) # Years to expiration

    length_contact = int(np.ceil((pd.to_datetime(expiry_dates[-1],format="%d-%m-%Y")-pd.to_datetime(valuation_date,format="%d-%m-%Y")).days/365)) # Length in years of the contract
    Kc = np.array([55]*length_contact)  # option call strike prices
    Kp = np.array([30]*length_contact)  # option put strike prices

    r = 0.035  # risk-free interest rate ECB
    sigma = np.std(returns.dropna())*np.sqrt(252)# Historical Volatility

    #st.write(sigma)

    options = pd.DataFrame({'Expiry Date':expiry_dates,'Time to Maturity':T,'K calls':Kc,'K puts': Kp, 'Futures': S})

    st.write(options)

    st.markdown('Now lets compare the different option pricing methodologies')

    # Define the Black-Scholes formula for option valuation

    from BSoption import black_scholes

    # Calculate the option prices using the Black-Scholes formula
    call_prices = []
    put_prices = []
    for i in enumerate(Kc):
        call, putx = black_scholes(S[i[0]], Kc[i[0]], r, T[i[0]], sigma)
        callx, put = black_scholes(S[i[0]], Kp[i[0]], r, T[i[0]], sigma)
        call_prices.append(call)
        put_prices.append(put)

    def OU_simulation(S, m, t, OU_params,r,K,T):
    ### Creating an array with simulated prices

        OU_est = np.transpose(get_OU_process(m,t,OU_params, X_0=S))

        final_prices = OU_est[-1,:]
        call_value = np.array(final_prices - K)

        put_value = np.array(K- final_prices)

        call_value[call_value < 0] = 0
        put_value[put_value < 0] = 0

        pv_call = call_value * np.exp(-r * T)
        pv_put = put_value * np.exp(-r * T)

        call = np.mean(pv_call)
        put = np.mean(pv_put)

        return call,put

    # Calculate the option prices using the Black-Scholes formula
    call_OU = []
    put_OU = []

    for i in enumerate(Kc):
        call, putx = OU_simulation(S[i[0]],10000,100,OU_params_hat,r,Kc[i[0]],T[i[0]])
        callx, put = OU_simulation(S[i[0]],10000,100,OU_params_hat,r,Kp[i[0]],T[i[0]])
        call_OU.append(call)
        put_OU.append(put)

    # plot the prices to visualize the distribution

    # Calculate the option prices using the Monte Carlo
    call_monte = []
    put_monte = []
    for i in enumerate(Kc):
        call, putx = Monte_Carlo(S[i[0]],r,sigma,T[i[0]],100,1000,Kc[i[0]])
        callx, put = Monte_Carlo(S[i[0]],r,sigma,T[i[0]],100,1000,Kp[i[0]])
        call_monte.append(call)
        put_monte.append(put)


    fig10, axs = plt.subplots(2,1, figsize=(7, 9))
    fig10.suptitle('Option Pricing comparison',fontsize=12, fontweight='bold',fontproperties=font, color = 'white')
    axs[0].plot(T,call_OU, color = 'lightblue')
    axs[0].plot(T,call_monte,color = 'lightgreen')
    axs[0].plot(T,call_prices, color = 'violet')
    axs[0].set_xlabel('Years to maturity',font = font, color = 'white')
    axs[0].set_ylabel('Call prices (EUR)' ,font = font, color = 'white')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['left'].set_color('white')
    axs[0].spines['bottom'].set_color('white')
    axs[0].tick_params(axis='x', colors='white' )   
    axs[0].tick_params(axis='y', colors='white')
    for tick in axs[0].get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in axs[0].get_yticklabels():
        tick.set_fontname("Times New Roman")

    fig10.set_facecolor('#0E1117')
    axs[0].set_facecolor("#0E1117")
    axs[1].set_facecolor("#0E1117")

    axs[1].plot(T,put_OU, color = 'lightblue', label = 'OU-process valuation')
    axs[1].plot(T,put_monte,color = 'lightgreen',label = 'Monte Carlo valuation')
    axs[1].plot(T,put_prices, color = 'violet', label = 'Black & Scholes valuation')
    axs[1].set_xlabel('Years to maturity' ,font = font, color = 'white')
    axs[1].set_ylabel('Put prices (EUR)' ,font = font, color = 'white')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    plt.legend(loc='best', prop=font)
    axs[1].spines['left'].set_color('white')
    axs[1].spines['bottom'].set_color('white')
    axs[1].tick_params(axis='x', colors='white' )   
    axs[1].tick_params(axis='y', colors='white')
    for tick in axs[1].get_xticklabels():
        tick.set_fontname("Times New Roman")
    for tick in axs[1].get_yticklabels():
        tick.set_fontname("Times New Roman");

    st.pyplot(fig10)

    st.markdown(f'Comparing the valuation methods, its clear that the Monte Carlo and Black&Scholes converge both to almost the same prices.\
            On the other hand, the OU-Simulation has a different result, specially for the calls. This is given that as the OU process, reverts to the\
            long-term mean, so for the longer periods of time, the value of the call will converge to the present value of:  ')
    st.latex('\Omega - K')

    st.markdown('where Omega is the long-term mean. This could make more sense for markets who are mean-reverting behaving, and perhaps the adoption of this modelling,\
            can help pricing options. However, what is also quite challenging is whether the assumption of this mean-reverting process is sustainable in the future...\
            As we have seen, it seems that the Energy market is not behaving as it used to!')
    
if __name__ == "__main__":
    main()
