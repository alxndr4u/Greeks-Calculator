import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime,timedelta
import pandas as pd
import pandas_datareader.data as web
import math
import yfinance as yf
from scipy.stats import norm

# It is recommended to install the required libraries using:
# pip install numpy pandas pandas-datareader yfinance scipy

yf.pdr_override()
N = norm.cdf

days_in_year=365

def get_user_input():
    """Gets user input for option codes and quantities."""
    code_list=[]
    quantity_list=[]
    while True:
      code=str(input("Code: "))
      if code=="-":
        break
      quantity=float(input("Quantity: "))
      print("-="*20)
      code_list.append(code)
      quantity_list.append(quantity)
    return code_list, quantity_list

def get_option_info(code, quantity):
  """Parses the option code and fetches market data."""
  counter=0
  alphabet=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q",'R',"S","T","U","V","W","X","Y","Z"]
  while True:
    if code[counter] not in alphabet:
      ticker_symbol=code[:counter]
      break
    counter+=1
  expiration_year=int("20"+code[counter:counter+2])
  expiration_month=int(code[counter+2:counter+4])
  expiration_day=int(code[counter+4:counter+6])
  today = date.today()
  expiration=date(expiration_year,expiration_month,expiration_day)
  DTE=expiration-today
  T=(DTE.days)/days_in_year
  option_type=code[counter+6]
  strike_price_str=code[counter+7:counter+14]
  # get market data
  ticker = yf.Ticker(ticker_symbol)
  todays_data = ticker.history(period='1d')
  option = yf.Ticker(code)
  option_price=(option.history(period="2d"))["Close"][-1]
  cost = option_price*-quantity
  return ticker,option_type,strike_price_str,T,todays_data["Close"][-1],option_price, cost

def d1_func(S,K,r,q,sigma,T):
  """Calculates d1 for the Black-Scholes model."""
  return (np.log(S/K)+(r-q+sigma**2/2)*T)/(sigma*math.sqrt(T))

def d2_func(S,K,r,q,sigma,T):
  """Calculates d2 for the Black-Scholes model."""
  d1=d1_func(S,K,r,q,sigma,T)
  return d1-sigma*math.sqrt(T)

def black_scholes_call(S,K,r,q,sigma,T):
  """Calculates the price of a call option using the Black-Scholes model."""
  d1=d1_func(S,K,r,q,sigma,T)
  d2=d2_func(S,K,r,q,sigma,T)
  return S*math.exp(-q*T)*N(d1)-math.exp(-r*T)*K*N(d2)

def black_scholes_put(S,K,r,q,sigma,T):
  """Calculates the price of a put option using the Black-Scholes model."""
  d1=d1_func(S,K,r,q,sigma,T)
  d2=d2_func(S,K,r,q,sigma,T)
  return math.exp(-r*T)*K*N(-d2)-S*math.exp(-q*T)*N(-d1)

def phi_d1_func(S,K,r,q,sigma,T):
    d1 = d1_func(S, K, r, q, sigma, T)
    return math.exp(-d1 ** 2 / 2) / math.sqrt(2 * math.pi)

def phi_d2_func(S,K,r,q,sigma,T):
    d2 = d2_func(S, K, r, q, sigma, T)
    return math.exp(-d2 ** 2 / 2) / math.sqrt(2 * math.pi)


def IV(option_price,S,K,T,r, option_type):
  """Calculates the implied volatility."""
  precision=0.00001
  upper_vol=50.0
  max_vol=50.0
  min_vol=0.0001
  lower_vol=0.0001
  iteration=0
  q = 0
  while 1:
    iteration+=1
    mid_vol=(upper_vol+lower_vol)/2.0
    if option_type=="C":
      price=black_scholes_call(S,K,r,q,mid_vol,T)
      lower_price=black_scholes_call(S,K,r,q,lower_vol,T)
      if (lower_price-option_price)*(price-option_price)>0:
        lower_vol=mid_vol
      else:
        upper_vol=mid_vol
      if abs(price-option_price)<precision:break
      if mid_vol>max_vol-5:
        mid_vol-0.0001
        break
    elif option_type=="P":
      price=black_scholes_put(S,K,r,q,mid_vol,T)
      upper_price=black_scholes_put(S,K,r,q,upper_vol,T)
      if (upper_price-option_price)*(price-option_price)>0:
        upper_vol=mid_vol
      else:
        lower_vol=mid_vol
      if abs(price-option_price)<precision:break
      if iteration>100:break
  return mid_vol

def delta_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the delta of an option."""
  d1=d1_func(S,K,r,q,sigma,T)
  if option_type=="C":
    delta=math.exp(-q*T)*N(d1)*quantity
  if option_type=="P":
    delta=-math.exp(-q*T)*N(-d1)*quantity
  return delta

def vega_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the vega of an option."""
  phid1=phi_d1_func(S,K,r,q,sigma,T)
  vega=S*math.exp(-q*T)*phid1*math.sqrt(T)/100*quantity
  return vega

def theta_func(S,K,r,q,sigma,T,option_type,days_in_year,quantity):
  """Calculates the theta of an option."""
  d1=d1_func(S,K,r,q,sigma,T)
  d2=d2_func(S,K,r,q,sigma,T)
  phid1=phi_d1_func(S,K,r,q,sigma,T)
  if option_type=="C":
    theta=((-math.exp(-q*T)*((S*phid1*sigma)/(2*math.sqrt(T)))-r*K*math.exp(-r*T)*N(d2)+q*S*math.exp(-q*T)*N(d1))/days_in_year)*quantity
  if option_type =="P":
    theta=((-math.exp(-q*T)*((S*phid1*sigma)/(2*math.sqrt(T)))+r*K*math.exp(-r*T)*N(-d2)-q*S*math.exp(-q*T)*N(-d1))/days_in_year)*quantity
  return theta

def rho_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the rho of an option."""
  d2=d2_func(S,K,r,q,sigma,T)
  if option_type=="C":
    rho=(K*T*math.exp(-r*T)*N(d2)/100)*quantity
  if option_type=="P":
    rho=(-K*T*math.exp(-r*T)*N(-d2)/100)*quantity
  return rho

def episilon_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the epsilon of an option."""
  d1=d1_func(S,K,r,q,sigma,T)
  if option_type == "C":
      epsilon = -S * T * math.exp(-q * T) * N(d1) / 100 * quantity
  if option_type == "P":
      epsilon = S * T * math.exp(-q * T) * N(-d1) / 100 * quantity
  return epsilon

def gamma_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the gamma of an option."""
  phid1=phi_d1_func(S,K,r,q,sigma,T)
  gamma=(math.exp(-q*T)*phid1/(S*sigma*math.sqrt(T)))*quantity
  return gamma

def vanna_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the vanna of an option."""
  d1=d1_func(S,K,r,q,sigma,T)
  d2=d2_func(S,K,r,q,sigma,T)
  phid1=phi_d1_func(S,K,r,q,sigma,T)
  vanna=(-math.exp(-q*T)*phid1*(d2/sigma))*quantity
  return vanna

def charm_func(S,K,r,q,sigma,T,option_type,days_in_year,quantity):
  """Calculates the charm of an option."""
  d1=d1_func(S,K,r,q,sigma,T)
  d2=d2_func(S,K,r,q,sigma,T)
  phid1=phi_d1_func(S,K,r,q,sigma,T)
  if option_type=="C":
    charm=((q*math.exp(-q*T)*N(d1)-math.exp(-q*T)*phid1*((2*(r-q)*T-d2*sigma*math.sqrt(T))/(2*T*sigma*math.sqrt(T))))/days_in_year)*quantity
  if option_type=="P":
    charm=((-q*math.exp(-q*T)*N(-d1)-math.exp(-q*T)*phid1*((2*(r-q)*T-d2*sigma*math.sqrt(T))/(2*T*sigma*math.sqrt(T))))/days_in_year)*quantity
  return charm

def vomma_func(S,K,r,q,sigma,T,option_type,quantity):
    """Calculates the vomma of an option."""
    d1 = d1_func(S, K, r, q, sigma, T)
    d2 = d2_func(S, K, r, q, sigma, T)
    vega = vega_func(S, K, r, q, sigma, T, option_type, quantity)
    return vega * d1 * d2 / sigma * quantity

def veta_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the veta of an option."""
  d1=d1_func(S,K,r,q,sigma,T)
  d2=d2_func(S,K,r,q,sigma,T)
  phid1=phi_d1_func(S,K,r,q,sigma,T)
  veta=(-S*math.exp(-q*T)*phid1*math.sqrt(T)*(q+((r-q)*d1)/(sigma*math.sqrt(T))-((1+d1*d2)/(2*T)))/((days_in_year)*100))*quantity
  return veta

def speed_func(S,K,r,q,sigma,T,option_type,quantity):
    """Calculates the speed of an option."""
    d1 = d1_func(S, K, r, q, sigma, T)
    gamma = gamma_func(S, K, r, q, sigma, T, option_type, quantity)
    return -gamma / S * (d1 / (sigma * math.sqrt(T)) + 1) * quantity

def zomma_func(S,K,r,q,sigma,T,option_type,quantity):
    """Calculates the zomma of an option."""
    d1 = d1_func(S, K, r, q, sigma, T)
    d2 = d2_func(S, K, r, q, sigma, T)
    gamma = gamma_func(S, K, r, q, sigma, T, option_type, quantity)
    return gamma * ((d1 * d2 - 1) / sigma) * quantity

def color_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the color of an option."""
  d1=d1_func(S,K,r,q,sigma,T)
  d2=d2_func(S,K,r,q,sigma,T)
  phid1=phi_d1_func(S,K,r,q,sigma,T)
  color=((-math.exp(-q*T)*((phid1)/(2*S*T*sigma*math.sqrt(T)))*(2*q*T+1+((2*(r-q)*T-d2*sigma*math.sqrt(T)))/sigma*math.sqrt(T)*d1))/days_in_year)*quantity
  return color

def ultima_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the ultima of an option."""
  d1=d1_func(S,K,r,q,sigma,T)
  d2=d2_func(S,K,r,q,sigma,T)
  vega=vega_func(S,K,r,q,sigma,T,option_type,quantity)
  ultima=(-vega/sigma**2)*(d1*d2*(1-d1*d2)+d1**2+d2**2)*quantity
  return ultima

def dual_delta_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the dual delta of an option."""
  d2=d2_func(S,K,r,q,sigma,T)
  if option_type=="C":
    dual_delta=(-math.exp(-r*T)*N(d2))*quantity
  if option_type=="P":
    dual_delta=(math.exp(-r*T)*N(-d2))*quantity
  return dual_delta

def dual_gamma_func(S,K,r,q,sigma,T,option_type,quantity):
  """Calculates the dual gamma of an option."""
  d2=d2_func(S,K,r,q,sigma,T)
  phid2=phi_d2_func(S,K,r,q,sigma,T)
  dual_gamma=math.exp(-r*T)*(phid2/(K*sigma*math.sqrt(T)))*quantity
  return dual_gamma

if __name__ == '__main__':
    code_list, quantity_list = get_user_input()

    total_cost=0
    delta, vega,theta,gamma,rho,episilon,zomma,charm,veta,vanna,vomma=0,0,0,0,0,0,0,0,0,0,0
    speed, ultima, dual_delta, dual_gamma, color = 0,0,0,0,0

    r=0.02
    q=0.0

    for i in range(len(code_list)):
        code=code_list[i]
        quantity=quantity_list[i]

        ticker,option_type,strike_price_str,T,S,option_price, cost = get_option_info(code, quantity)
        K=float(strike_price_str)/100
        total_cost += cost

        sigma=IV(option_price,S,K,T,r,option_type)

        delta+=delta_func(S,K,r,q,sigma,T,option_type,quantity)
        vega+=vega_func(S,K,r,q,sigma,T,option_type,quantity)
        theta+=theta_func(S,K,r,q,sigma,T,option_type,days_in_year,quantity)
        rho+=rho_func(S,K,r,q,sigma,T,option_type,quantity)
        episilon+=episilon_func(S,K,r,q,sigma,T,option_type,quantity)

        gamma+=gamma_func(S,K,r,q,sigma,T,option_type,quantity)
        vanna+=vanna_func(S,K,r,q,sigma,T,option_type,quantity)
        charm+=charm_func(S,K,r,q,sigma,T,option_type,days_in_year,quantity)# percentage
        vomma+=vomma_func(S,K,r,q,sigma,T,option_type,quantity)# percentage
        veta+= veta_func(S,K,r,q,sigma,T,option_type,quantity)# Percentage

        speed+=speed_func(S,K,r,q,sigma,T,option_type,quantity)
        zomma+=zomma_func(S,K,r,q,sigma,T,option_type,quantity)
        color+=color_func(S,K,r,q,sigma,T,option_type,quantity)
        ultima+=ultima_func(S,K,r,q,sigma,T,option_type,quantity)
        dual_delta+=dual_delta_func(S,K,r,q,sigma,T,option_type,quantity)
        dual_gamma+=dual_gamma_func(S,K,r,q,sigma,T,option_type,quantity)

    print(f"Total Cost {total_cost.round(2)}$")
    print("-="*20)
    print("First Order Greeks")
    print(f"Delta: {delta.round(3)} $/ΔSpot")
    print(f"Theta: {theta.round(3)} $/ΔTime")
    print(f"Vega: {vega.round(3)} $/ΔVolatility")
    print(f"Rho: {rho.round(3)} $/ΔRisk Free rate")
    print(f"Episilon: {(episilon).round(3)} $/ΔRisk Free rate")
    print("-="*20)
    print("Second Order Greeks")
    print(f"Gamma: {(gamma).round(3)} ΔDelta/ΔSpot")
    print(f"Charm: {(charm).round(3)} ΔDelta/ΔTime")
    print(f"Vomma: {(vomma).round(3)} ΔVega/ΔVolatility")
    print(f"Vanna: {vanna.round(3)} ΔVega/ΔSpot")
    print(f"Veta: {(veta).round(3)} ΔVega/ΔTime")
    print("-="*20)
    print("Third Order Greeks")
    print(f"Speed: {(speed).round(4)} ΔGamma/ΔSpot")
    print(f"Zomma: {(zomma).round(4)} ΔGamma/ΔVolatility")
    print(f"Color: {(color).round(4)} ΔGamma/ΔTime")
    print(f"Ultima: {ultima.round(2)} ΔVomma/ΔVolatility")
