import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred

# --- PAGE CONFIG ---
st.set_page_config(page_title="Core PCE Decomposition", page_icon="📊", layout="wide")

# --- CONFIG ---
FRED_API_KEY = "7e1665bada40c2a05bb6ea9d07e62920"
WEIGHTS = {'housing': 0.175, 'non_housing_services': 0.575, 'core_goods': 0.250}

# --- DATA ---
@st.cache_data(ttl=3600)
def fetch_pce_data():
    fred = Fred(api_key=FRED_API_KEY)
    core_pce = fred.get_series('PCEPILFE', observation_start='2019-01-01')
    core_services_ex_housing = fred.get_series('IA001260M', observation_start='2019-01-01')
    core_pce_ex_housing = fred.get_series('IA001176M', observation_start='2019-01-01')
    
    df = pd.DataFrame({
        'core_pce': core_pce,
        'core_services_ex_housing': core_services_ex_housing,
        'core_pce_ex_housing': core_pce_ex_housing
    }).dropna()
    
    df_mom = df.pct_change() * 100
    df_yoy = df.pct_change(periods=12) * 100
    
    df_mom['non_housing_services'] = df_mom['core_services_ex_housing']
    df_yoy['non_housing_services'] = df_yoy['core_services_ex_housing']
    df_mom['housing'] = (df_mom['core_pce'] - df_mom['core_pce_ex_housing']) / WEIGHTS['housing'] * (1 - WEIGHTS['housing'])
    df_yoy['housing'] = (df_yoy['core_pce'] - df_yoy['core_pce_ex_housing']) / WEIGHTS['housing'] * (1 - WEIGHTS['housing'])
    df_mom['core_goods'] = (df_mom['core_pce_ex_housing'] - df_mom['core_services_ex_housing'] * WEIGHTS['non_housing_services']) / WEIGHTS['core_goods']
    df_yoy['core_goods'] = (df_yoy['core_pce_ex_housing'] - df_yoy['core_services_ex_housing'] * WEIGHTS['non_housing_services']) / WEIGHTS['core_goods']
    
    return df, df_mom, df_yoy

def generate_forecast(df_mom, housing_pace, non_housing_pace, goods_pace, months=12):
    last_date = df_mom.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS')
    forecast = pd.DataFrame(index=forecast_dates)
    forecast['housing_mom'] = housing_pace
    forecast['non_housing_services_mom'] = non_housing_pace
    forecast['core_goods_mom'] = goods_pace
    forecast['housing_contrib'] = housing_pace * WEIGHTS['housing']
    forecast['non_housing_contrib'] = non_housing_pace * WEIGHTS['non_housing_services']
    forecast['goods_contrib'] = goods_pace * WEIGHTS['core_goods']
    forecast['total_mom'] = forecast['housing_contrib'] + forecast['non_housing_contrib'] + forecast['goods_contrib']
    forecast['annualized'] = ((1 + forecast['total_mom']/100)**12 - 1) * 100
    return forecast

def calculate_yoy_path(df_mom, forecast):
    recent_actual = df_mom.tail(12).copy()
    hist_housing = recent_actual['housing'].fillna(0)
    hist_non_housing = recent_actual['non_housing_services'].fillna(0)
    hist_goods = recent_actual['core_goods'].fillna(0)
    combined_housing = pd.concat([hist_housing, forecast['housing_mom']])
    combined_non_housing = pd.concat([hist_non_housing, forecast['non_housing_services_mom']])
    combined_goods = pd.concat([hist_goods, forecast['core_goods_mom']])
    
    yoy_path = []
    for i in range(len(forecast)):
        h_12m = combined_housing.iloc[i:i+12].sum()
        nh_12m = combined_non_housing.iloc[i:i+12].sum()
        g_12m = combined_goods.iloc[i:i+12].sum()
        total_yoy = h_12m * WEIGHTS['housing'] + nh_12m * WEIGHTS['non_housing_services'] + g_12m * WEIGHTS['core_goods']
        yoy_path.append({
            'date': forecast.index[i], 'total_yoy': total_yoy,
            'housing_contrib_yoy': h_12m * WEIGHTS['housing'],
            'non_housing_contrib_yoy': nh_12m * WEIGHTS['non_housing_services'],
            'goods_contrib_yoy': g_12m * WEIGHTS['core_goods']
        })
    return pd.DataFrame(yoy_path).set_index('date')

def create_tufte_chart(df_yoy, yoy_path):
    colors = {'housing': '#2E86AB', 'non_housing': '#A23B72', 'goods': '#F18F01', 'total': '#1B1B1E', 'forecast_bg': 'rgba(200,200,200,0.15)', 'grid': 'rgba(0,0,0,0.06)'}
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.55, 0.45], vertical_spacing=0.12,
                        subplot_titles=('<b>Core PCE Inflation: Year-over-Year Path</b>', '<b>Component Contributions to YoY Change</b>'))
    
    hist_yoy = df_yoy['core_pce'].dropna()
    fig.add_trace(go.Scatter(x=hist_yoy.index, y=hist_yoy.values, mode='lines', name='Core PCE YoY (Actual)',
                             line=dict(color=colors['total'], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=yoy_path.index, y=yoy_path['total_yoy'], mode='lines+markers', name='Forecast Path',
                             line=dict(color=colors['total'], width=2.5, dash='dot'), marker=dict(size=6)), row=1, col=1)
    fig.add_hline(y=2.0, line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'), annotation_text='2% Target', row=1, col=1)
    fig.add_vrect(x0=yoy_path.index[0], x1=yoy_path.index[-1], fillcolor=colors['forecast_bg'], layer='below', line_width=0, row=1, col=1)
    
    hist_housing_contrib = df_yoy['housing'].dropna() * WEIGHTS['housing']
    hist_goods_contrib = df_yoy['core_goods'].dropna() * WEIGHTS['core_goods']
    hist_nh_contrib = df_yoy['non_housing_services'].dropna() * WEIGHTS['non_housing_services']
    common_idx = hist_housing_contrib.index.intersection(hist_goods_contrib.index).intersection(hist_nh_contrib.index)
    
    fig.add_trace(go.Scatter(x=common_idx, y=hist_goods_contrib.loc[common_idx], mode='lines', name='Core Goods',
                             fill='tozeroy', fillcolor='rgba(241, 143, 1, 0.6)', line=dict(color=colors['goods'], width=0.5), stackgroup='hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=common_idx, y=hist_nh_contrib.loc[common_idx], mode='lines', name='Non-Housing Services',
                             fill='tonexty', fillcolor='rgba(162, 59, 114, 0.6)', line=dict(color=colors['non_housing'], width=0.5), stackgroup='hist'), row=2, col=1)
    fig.add_trace(go.Scatter(x=common_idx, y=hist_housing_contrib.loc[common_idx], mode='lines', name='Housing Services',
                             fill='tonexty', fillcolor='rgba(46, 134, 171, 0.6)', line=dict(color=colors['housing'], width=0.5), stackgroup='hist'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=yoy_path.index, y=yoy_path['goods_contrib_yoy'], mode='lines', fill='tozeroy', fillcolor='rgba(241, 143, 1, 0.4)',
                             line=dict(color=colors['goods'], width=1, dash='dot'), stackgroup='fcst', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=yoy_path.index, y=yoy_path['non_housing_contrib_yoy'], mode='lines', fill='tonexty', fillcolor='rgba(162, 59, 114, 0.4)',
                             line=dict(color=colors['non_housing'], width=1, dash='dot'), stackgroup='fcst', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=yoy_path.index, y=yoy_path['housing_contrib_yoy'], mode='lines', fill='tonexty', fillcolor='rgba(46, 134, 171, 0.4)',
                             line=dict(color=colors['housing'], width=1, dash='dot'), stackgroup='fcst', showlegend=False), row=2, col=1)
    fig.add_vrect(x0=yoy_path.index[0], x1=yoy_path.index[-1], fillcolor=colors['forecast_bg'], layer='below', line_width=0, row=2, col=1)
    
    fig.update_layout(height=600, font=dict(family='Georgia, serif', size=12, color='#1a1a1a'),
                      paper_bgcolor='#fafafa', plot_bgcolor='#fafafa', margin=dict(l=60, r=40, t=80, b=60),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5), hovermode='x unified')
    fig.update_xaxes(showgrid=False, showline=True, linecolor='rgba(0,0,0,0.2)', dtick='M6', tickformat='%b\n%Y')
    fig.update_yaxes(showgrid=True, gridcolor=colors['grid'], ticksuffix='%', zeroline=True, zerolinecolor='rgba(0,0,0,0.15)')
    return fig

def create_component_chart(df_yoy, yoy_path):
    colors = {'housing': '#2E86AB', 'non_housing': '#A23B72', 'goods': '#F18F01', 'total': '#1B1B1E', 'forecast_bg': 'rgba(200,200,200,0.15)', 'grid': 'rgba(0,0,0,0.06)'}
    fig = go.Figure()
    
    cutoff_date = df_yoy.index[-1] - pd.DateOffset(years=2)
    hist_housing = df_yoy['housing'].loc[df_yoy.index >= cutoff_date].dropna()
    hist_non_housing = df_yoy['non_housing_services'].loc[df_yoy.index >= cutoff_date].dropna()
    hist_goods = df_yoy['core_goods'].loc[df_yoy.index >= cutoff_date].dropna()
    hist_total = df_yoy['core_pce'].loc[df_yoy.index >= cutoff_date].dropna()
    common_idx = hist_housing.index.intersection(hist_non_housing.index).intersection(hist_goods.index).intersection(hist_total.index)
    
    hist_goods_contrib = hist_goods.loc[common_idx] * WEIGHTS['core_goods']
    hist_nh_contrib = hist_non_housing.loc[common_idx] * WEIGHTS['non_housing_services']
    hist_housing_contrib = hist_housing.loc[common_idx] * WEIGHTS['housing']
    
    fig.add_trace(go.Scatter(x=common_idx, y=hist_goods_contrib, mode='lines', name='Core Goods', fill='tozeroy', fillcolor='rgba(241, 143, 1, 0.6)', line=dict(color=colors['goods'], width=0.5), stackgroup='hist'))
    fig.add_trace(go.Scatter(x=common_idx, y=hist_nh_contrib, mode='lines', name='Non-Housing Services', fill='tonexty', fillcolor='rgba(162, 59, 114, 0.6)', line=dict(color=colors['non_housing'], width=0.5), stackgroup='hist'))
    fig.add_trace(go.Scatter(x=common_idx, y=hist_housing_contrib, mode='lines', name='Housing Services', fill='tonexty', fillcolor='rgba(46, 134, 171, 0.6)', line=dict(color=colors['housing'], width=0.5), stackgroup='hist'))
    
    fig.add_trace(go.Scatter(x=yoy_path.index, y=yoy_path['goods_contrib_yoy'], mode='lines', fill='tozeroy', fillcolor='rgba(241, 143, 1, 0.4)', line=dict(color=colors['goods'], width=1, dash='dot'), stackgroup='fcst', showlegend=False))
    fig.add_trace(go.Scatter(x=yoy_path.index, y=yoy_path['non_housing_contrib_yoy'], mode='lines', fill='tonexty', fillcolor='rgba(162, 59, 114, 0.4)', line=dict(color=colors['non_housing'], width=1, dash='dot'), stackgroup='fcst', showlegend=False))
    fig.add_trace(go.Scatter(x=yoy_path.index, y=yoy_path['housing_contrib_yoy'], mode='lines', fill='tonexty', fillcolor='rgba(46, 134, 171, 0.4)', line=dict(color=colors['housing'], width=1, dash='dot'), stackgroup='fcst', showlegend=False))
    
    fig.add_trace(go.Scatter(x=hist_total.index, y=hist_total.values, mode='lines', name='Core PCE YoY', line=dict(color=colors['total'], width=2.5), fill=None))
    fig.add_trace(go.Scatter(x=yoy_path.index, y=yoy_path['total_yoy'], mode='lines+markers', line=dict(color=colors['total'], width=2.5, dash='dot'), marker=dict(size=5), showlegend=False, fill=None))
    
    fig.add_hline(y=2.0, line=dict(color='rgba(0,0,0,0.3)', width=1, dash='dash'), annotation_text='2% Target')
    fig.add_vrect(x0=yoy_path.index[0], x1=yoy_path.index[-1], fillcolor=colors['forecast_bg'], layer='below', line_width=0)
    
    fig.update_layout(
        title=dict(text='<b>Component Contributions: Last 2 Years + Forecast</b>', y=0.95),
        height=450,
        paper_bgcolor='#fafafa', 
        plot_bgcolor='#fafafa',
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        hovermode='x unified',
        yaxis_title='YoY % / Contribution (pp)'
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor='rgba(0,0,0,0.2)', dtick='M3', tickformat='%b\n%Y')
    fig.update_yaxes(showgrid=True, gridcolor=colors['grid'], ticksuffix='%', zeroline=True, zerolinecolor='rgba(0,0,0,0.15)')
    return fig

# --- MAIN APP ---
st.title("Core PCE Inflation Decomposition")
st.markdown("Adjust the monthly pace assumptions for each pillar to generate a 12-month forecast.")
st.markdown("---")

df, df_mom, df_yoy = fetch_pce_data()

# Sidebar
st.sidebar.header("Monthly Pace Assumptions (% MoM)")
st.sidebar.caption("Pre-pandemic average ~0.15% MoM")
housing_pace = st.sidebar.slider("🏠 Housing", -0.2, 0.8, 0.30, 0.02, format="%.2f")
non_housing_pace = st.sidebar.slider("💼 Non-Housing Services", -0.2, 0.6, 0.25, 0.02, format="%.2f")
goods_pace = st.sidebar.slider("📦 Core Goods", -0.4, 0.4, 0.00, 0.02, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Weights:** Housing {WEIGHTS['housing']*100:.1f}% · Non-Housing {WEIGHTS['non_housing_services']*100:.1f}% · Goods {WEIGHTS['core_goods']*100:.1f}%")
st.sidebar.markdown(f"**Data through:** {df.index[-1].strftime('%B %Y')}")

# Generate forecast
forecast = generate_forecast(df_mom, housing_pace, non_housing_pace, goods_pace)
yoy_path = calculate_yoy_path(df_mom, forecast)

# Metrics
current_yoy = df_yoy['core_pce'].dropna().iloc[-1]
final_yoy = yoy_path['total_yoy'].iloc[-1]
implied_monthly = forecast['total_mom'].iloc[0]
implied_annual = forecast['annualized'].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Core PCE YoY", f"{current_yoy:.2f}%")
col2.metric("12-Month Forecast YoY", f"{final_yoy:.2f}%", f"{final_yoy - current_yoy:.2f}pp", delta_color="inverse")
col3.metric("Implied Monthly Pace", f"{implied_monthly:.3f}%")
col4.metric("Annualized Rate", f"{implied_annual:.2f}%")

st.markdown("---")

# Charts
st.plotly_chart(create_component_chart(df_yoy, yoy_path), use_container_width=True)
