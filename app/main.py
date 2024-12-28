import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.offline as py

from datetime import datetime, timezone, timedelta

import random

import streamlit as st

st.set_page_config(
    page_title=" Backtesting Dynamische Strompreise",
    page_icon=":bar_chart:",
    layout="wide"
)

hide_st_style = '''
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
//header {visibility: hidden;}
</style>
'''

st.markdown(hide_st_style, unsafe_allow_html=True)

# load base dataframes

@st.cache_data
def get_data():
    _prices_df = pd.read_pickle('prices.pkl')
    _solar_production_df = pd.read_pickle('solar_production.pkl')
    _solar_production_df.loc[_solar_production_df.index == datetime(2024, 2, 28, 16, 0), 'solar_production'] = 0.0

    return _prices_df, _solar_production_df


_prices_df, _solar_production_df = get_data()

###
# define inputs
###

st.html('''
    <div style="text-align: center; margin-top: -30px;"> 
    <h2>Backtest Dynamische Strompreise vs. Festtarif </h2>
    </div>''')

left_column, middle_column, right_column = st.columns([2, 4, 2])
with left_column:
    ll_column, lr_column = st.columns(2)
    with ll_column:
        st.markdown('*Anlage*')
        target_kwp = st.number_input("Anlagen-Größe in kWp", value=24)
        battery_capacity = st.number_input("Speicher-Kapazität in kWh", value=22)
        st.markdown('*Verbrauch*')
        yearly_consumption = st.number_input("Jahres-Verbrauch in kWh", value=24000)
        consumption_model = st.selectbox(
            "Verbrauchsprofil",
            ("Morgens-und-Abends",
             "Tagsüber",
             "Konstant"),
        )
    with lr_column:
        st.markdown('*Markt*')
        selling_price = st.number_input("Verkaufspreis in €", value=0.25)
        arbeits_price = st.number_input("Arbeitspreis in €", value=0.34)
        einspeise_price = st.number_input("Einspeisepreis in €", value=0.07)


_max_battery_load = min(battery_capacity/2, 9.0)

_kWp = 13
_kWp_factor = target_kwp/_kWp

_orig_df = _solar_production_df.copy()
_orig_df.solar_production = _orig_df.solar_production * _kWp_factor
_daily_consumption = yearly_consumption/365

consumption_mask_working_family = pd.DataFrame([
    dict(start='01:00', consumption=0.01 * _daily_consumption),
    dict(start='02:00', consumption=0.01 * _daily_consumption),
    dict(start='03:00', consumption=0.01 * _daily_consumption),
    dict(start='04:00', consumption=0.01 * _daily_consumption),
    dict(start='05:00', consumption=0.01 * _daily_consumption),
    dict(start='06:00', consumption=0.085* _daily_consumption),
    dict(start='07:00', consumption=0.09* _daily_consumption),
    dict(start='08:00', consumption=0.072* _daily_consumption),
    dict(start='09:00', consumption=0.03* _daily_consumption),
    dict(start='10:00', consumption=0.01* _daily_consumption),
    dict(start='11:00', consumption=0.01* _daily_consumption),
    dict(start='12:00', consumption=0.02* _daily_consumption),
    dict(start='13:00', consumption=0.04* _daily_consumption),
    dict(start='14:00', consumption=0.02* _daily_consumption),
    dict(start='15:00', consumption=0.01* _daily_consumption),
    dict(start='16:00', consumption=0.01* _daily_consumption),
    dict(start='17:00', consumption=0.01* _daily_consumption),
    dict(start='18:00', consumption=0.046*_daily_consumption),
    dict(start='19:00', consumption=0.076* _daily_consumption),
    dict(start='20:00', consumption=0.121* _daily_consumption),
    dict(start='21:00', consumption=0.115* _daily_consumption),
    dict(start='22:00', consumption=0.110* _daily_consumption),
    dict(start='23:00', consumption=0.05* _daily_consumption),
    dict(start='00:00', consumption=0.025* _daily_consumption)
])

consumption_mask_company = pd.DataFrame([
    dict(start='01:00', consumption=0.04 * _daily_consumption),
    dict(start='02:00', consumption=0.04 * _daily_consumption),
    dict(start='03:00', consumption=0.04 * _daily_consumption),
    dict(start='04:00', consumption=0.04 * _daily_consumption),
    dict(start='05:00', consumption=0.04 * _daily_consumption),
    dict(start='06:00', consumption=0.041* _daily_consumption),
    dict(start='07:00', consumption=0.042* _daily_consumption),
    dict(start='08:00', consumption=0.044* _daily_consumption),
    dict(start='09:00', consumption=0.045* _daily_consumption),
    dict(start='10:00', consumption=0.045* _daily_consumption),
    dict(start='11:00', consumption=0.045* _daily_consumption),
    dict(start='12:00', consumption=0.045* _daily_consumption),
    dict(start='13:00', consumption=0.045* _daily_consumption),
    dict(start='14:00', consumption=0.045* _daily_consumption),
    dict(start='15:00', consumption=0.045* _daily_consumption),
    dict(start='16:00', consumption=0.046* _daily_consumption),
    dict(start='17:00', consumption=0.046* _daily_consumption),
    dict(start='18:00', consumption=0.045*_daily_consumption),
    dict(start='19:00', consumption=0.04* _daily_consumption),
    dict(start='20:00', consumption=0.04* _daily_consumption),
    dict(start='21:00', consumption=0.04* _daily_consumption),
    dict(start='22:00', consumption=0.04* _daily_consumption),
    dict(start='23:00', consumption=0.04* _daily_consumption),
    dict(start='00:00', consumption=0.04* _daily_consumption)
])

consumption_mask_constant = pd.DataFrame([
    dict(start='01:00', consumption=0.042 * _daily_consumption),
    dict(start='02:00', consumption=0.042 * _daily_consumption),
    dict(start='03:00', consumption=0.042 * _daily_consumption),
    dict(start='04:00', consumption=0.042 * _daily_consumption),
    dict(start='05:00', consumption=0.042 * _daily_consumption),
    dict(start='06:00', consumption=0.042* _daily_consumption),
    dict(start='07:00', consumption=0.042* _daily_consumption),
    dict(start='08:00', consumption=0.042* _daily_consumption),
    dict(start='09:00', consumption=0.042* _daily_consumption),
    dict(start='10:00', consumption=0.042* _daily_consumption),
    dict(start='11:00', consumption=0.042* _daily_consumption),
    dict(start='12:00', consumption=0.042* _daily_consumption),
    dict(start='13:00', consumption=0.042* _daily_consumption),
    dict(start='14:00', consumption=0.042* _daily_consumption),
    dict(start='15:00', consumption=0.042* _daily_consumption),
    dict(start='16:00', consumption=0.042* _daily_consumption),
    dict(start='17:00', consumption=0.042* _daily_consumption),
    dict(start='18:00', consumption=0.042*_daily_consumption),
    dict(start='19:00', consumption=0.042* _daily_consumption),
    dict(start='20:00', consumption=0.042* _daily_consumption),
    dict(start='21:00', consumption=0.042* _daily_consumption),
    dict(start='22:00', consumption=0.042* _daily_consumption),
    dict(start='23:00', consumption=0.042* _daily_consumption),
    dict(start='00:00', consumption=0.042* _daily_consumption)
])

match consumption_model:
    case 'Morgens-und-Abends':
        _consumption_mask = consumption_mask_working_family
    case 'Tagsüber':
        _consumption_mask = consumption_mask_company
    case 'Konstant':
        _consumption_mask = consumption_mask_constant

with middle_column:
    fig = px.line(_consumption_mask, x='start', y=['consumption'])
    fig.update_layout(showlegend=False, xaxis_title='Zeit', yaxis_title='Verbrauch', title='Verbrauchsprofil')
    st.plotly_chart(fig)


pd.options.mode.chained_assignment = None


@st.cache_data
def get_solar_forecast(start, end):
    _q = f'start >= "{start}" & start < "{end}"'
    return _solar_production_df.query(_q).solar_production.sum()

@st.cache_data
def get_known_prices(start, end):
    _q = f'start >= "{start}" & start < "{end}"'
    return _prices_df.query(_q)


@st.cache_data
def run_simulation(battery_capacity,
                    yearly_consumption,
                    target_kwp,
                    consumption_model,
                    selling_price,
                    arbeits_price,
                    einspeise_price):
    _df = _orig_df.copy()

    _df['consumption'] = 0.0
    _df['energy_deficit_dyn_prices'] = 0.0
    _df['energy_deficit_norm_prices'] = 0.0

    _df['battery_load_dyn_prices'] = 0.0
    _df['battery_load_norm_prices'] = 0.0

    _df['cur_battery_capacity_dyn_prices'] = 0.0
    _df['cur_battery_capacity_norm_prices'] = 0.0

    _df['grid_export_dyn_prices'] = 0.0
    _df['grid_export_norm_prices'] = 0.0

    _df['grid_import_dyn_prices'] = 0.0
    _df['grid_import_norm_prices'] = 0.0

    _df['norm_costs'] = 0.0
    _df['dyn_price'] = 0.0
    _df['dyn_costs'] = 0.0

    __cur_battery_capacity_dyn_prices = 0.0
    __cur_battery_capacity_norm_prices = 0.0

    for index, row in _df.iterrows():
        _time = index.strftime('%H:%M')
        _cons_val = _consumption_mask.loc[_consumption_mask.start == _time].iloc[0].consumption
        _consumption = random.randint(int(_cons_val * 100 * 0.7), int(_cons_val * 100 * 1.5)) / 100
        _df.loc[index, 'consumption'] = _consumption
        solar_production = row['solar_production']

        # load battery
        _bat_load_dyn_prices = 0.0
        _bat_load_norm_prices = 0.0
        if solar_production > _consumption:
            # dyn price case
            _bat_load_1 = min((solar_production - _consumption), _max_battery_load)
            _bat_load_dyn_prices = min(battery_capacity - __cur_battery_capacity_dyn_prices, _bat_load_1)
            __cur_battery_capacity_dyn_prices = __cur_battery_capacity_dyn_prices + _bat_load_dyn_prices
            _df.loc[index, 'cur_battery_capacity_dyn_prices'] = __cur_battery_capacity_dyn_prices + _bat_load_dyn_prices
            _df.loc[index, 'battery_load_dyn_prices'] = -1 * _bat_load_dyn_prices
            _df.loc[index, 'grid_export_dyn_prices'] = solar_production - _consumption - _bat_load_dyn_prices

            # norm price case
            _bat_load_1 = min((solar_production - _consumption), _max_battery_load)
            _bat_load_norm_prices = min(battery_capacity - __cur_battery_capacity_norm_prices, _bat_load_1)
            __cur_battery_capacity_norm_prices = __cur_battery_capacity_norm_prices + _bat_load_norm_prices
            _df.loc[index, 'cur_battery_capacity_dyn_prices'] = __cur_battery_capacity_norm_prices + _bat_load_norm_prices
            _df.loc[index, 'battery_load_norm_prices'] = -1 * _bat_load_norm_prices
            _df.loc[index, 'grid_export_norm_prices'] = solar_production - _consumption - _bat_load_norm_prices

        # load battery in dynamic prices case
        start = datetime(index.year, index.month, index.day, 0, 0)
        end = start + timedelta(days=1)

        _known_prices = get_known_prices(start, end)
        _current_price = _prices_df.loc[
            _prices_df['start'] == datetime(index.year, index.month, index.day, index.hour, index.minute)]

        if len(_current_price) > 0:
            _current_price = _current_price.iloc[0]['min']
        else:
            _current_price = 0.33

        if len(_known_prices) > 0:
            _q2 = _known_prices['min'].quantile(0.2)
            _q25 = _known_prices['min'].quantile(0.25)
            _q65 = _known_prices['min'].quantile(0.65)

            _df.loc[index, 'dyn_price'] = _current_price

            _solar_forecast = get_solar_forecast(start, end)

            if _solar_forecast < (_daily_consumption * 1.1) and _current_price < _q25:
                needed_capacity = (_daily_consumption * 0.7) - _solar_forecast
                _bat_load_dyn_prices = min((battery_capacity - __cur_battery_capacity_dyn_prices), needed_capacity,
                                           _max_battery_load)
                if _bat_load_dyn_prices > 0:
                    __cur_battery_capacity_dyn_prices = __cur_battery_capacity_dyn_prices + _bat_load_dyn_prices
                    _df.loc[index, 'battery_load_dyn_prices'] = -1 * _bat_load_dyn_prices
                    _df.loc[index, 'grid_import_dyn_prices'] = -1 * _bat_load_dyn_prices
                    _df.loc[index, 'dyn_costs'] = _bat_load_dyn_prices * _current_price
                else:
                    _bat_load_dyn_prices = 0

        # remaining dyn case
        _battery_unload = 0.0
        if solar_production < _consumption:
            _battery_unload = _consumption if __cur_battery_capacity_dyn_prices - _consumption >= 0 else max(
                __cur_battery_capacity_dyn_prices, 0.0)
            _df.loc[index, 'battery_load_dyn_prices'] = _battery_unload - _bat_load_dyn_prices
            __cur_battery_capacity_dyn_prices = __cur_battery_capacity_dyn_prices - _battery_unload

        if (solar_production + __cur_battery_capacity_dyn_prices) < _consumption:
            _deficit = (_consumption - solar_production - _battery_unload)
            _df.loc[index, 'energy_deficit_dyn_prices'] = _deficit
            _df.loc[index, 'grid_import_dyn_prices'] = -1 * _deficit
            _df.loc[index, 'dyn_costs'] = _deficit * _current_price

            # remaining norm case
        _battery_unload = 0.0
        if solar_production < _consumption:
            _battery_unload = _consumption if __cur_battery_capacity_norm_prices - _consumption >= 0 else max(
                __cur_battery_capacity_norm_prices, 0.0)
            _df.loc[index, 'battery_load_norm_prices'] = _battery_unload - _bat_load_norm_prices
            __cur_battery_capacity_norm_prices = __cur_battery_capacity_norm_prices - _battery_unload

        if (solar_production + __cur_battery_capacity_norm_prices) < _consumption:
            _deficit = (_consumption - solar_production - _battery_unload)
            _df.loc[index, 'energy_deficit_norm_prices'] = _deficit
            _df.loc[index, 'grid_import_norm_prices'] = -1 * _deficit
            _df.loc[index, 'norm_costs'] = _deficit * arbeits_price

    return _df


_df = run_simulation(battery_capacity,
                    yearly_consumption,
                    target_kwp,
                    consumption_model,
                    selling_price,
                    arbeits_price,
                    einspeise_price)

fig = px.line(_df, x=_df.index, y=['solar_production', 'consumption', 'energy_deficit_dyn_prices',
                                   'battery_load_dyn_prices', 'grid_import_dyn_prices', 'grid_export_dyn_prices',
                                   'energy_deficit_norm_prices',
                                   'battery_load_norm_prices', 'grid_import_norm_prices', 'grid_export_norm_prices'])
fig.update_layout(xaxis_title='Datum', yaxis_title='kWh', title='Energie-Verteilung (Simulation, Solaretrag skaliert aus Echt-Daten 2024)')
st.plotly_chart(fig)

fig = px.line(_prices_df, x=_prices_df.start, y=['min'])
fig.update_layout(showlegend=False, xaxis_title='Datum', yaxis_title='Euro', title='Dynamischer Endkunden-Strompreis, inklusive aller Steuern und Abgaben, Echtdaten aus 2024')
st.plotly_chart(fig)

#st.dataframe(_prices_df)
#st.dataframe(_solar_production_df)



_daily = _df.groupby(pd.Grouper(freq='D'))[['solar_production', 'consumption', 'energy_deficit_dyn_prices',
                                   'battery_load_dyn_prices', 'grid_import_dyn_prices', 'grid_export_dyn_prices',
                                   'energy_deficit_norm_prices',
                                   'battery_load_norm_prices', 'grid_import_norm_prices', 'grid_export_norm_prices',
                                    'norm_costs', 'dyn_costs']].sum()

result_df = pd.DataFrame([dict(
    norm_costs=int(_daily.norm_costs.sum()),
    dyn_costs=int(_daily.dyn_costs.sum()),
    energy_deficit_norm_prices=int(_daily.energy_deficit_norm_prices.sum()),
    energy_deficit_dyn_prices=int(_daily.energy_deficit_dyn_prices.sum()),
    grid_import_norm_prices=int(_daily.grid_import_norm_prices.sum()),
    grid_import_dyn_prices=int(_daily.grid_import_dyn_prices.sum()),
    consumption=int(_daily.consumption.sum()),
    consumption_costs=int(_daily.consumption.sum()*arbeits_price),
    solar_production_pure_profit=int(_daily.solar_production.sum() * einspeise_price),
    sale_tenant_dyn_profit= int( (_daily.consumption.sum() * selling_price) - (_daily.dyn_costs.sum()) + (_daily.grid_export_dyn_prices.sum() * einspeise_price)),
    sale_tenant_norm_profit=int( (_daily.consumption.sum() * selling_price) - (_daily.norm_costs.sum()) + (_daily.grid_export_norm_prices.sum() * einspeise_price)),
    tenant_costs_dyn_prices=int( (_daily.consumption.sum() * selling_price)),
    tenant_costs_savings=int( (_daily.consumption.sum() * arbeits_price) - (_daily.consumption.sum() * selling_price)),
)])

with right_column:
    rl_column, rr_column = st.columns(2)
    with rl_column:
        st.metric('Profit Dynamische Preise', str(result_df.iloc[0].sale_tenant_dyn_profit) + ' €',
                  delta=str(result_df.iloc[0].sale_tenant_dyn_profit - result_df.iloc[0].solar_production_pure_profit) + ' € (zu Volleinspeisung)', delta_color="normal", help=None, label_visibility="visible", border=False)
        st.metric('Profit Festpreis', str(result_df.iloc[0].sale_tenant_norm_profit),
                  delta=str(result_df.iloc[0].sale_tenant_norm_profit - result_df.iloc[0].solar_production_pure_profit) + ' € (zu Volleinspeisung)', delta_color="normal", help=None, label_visibility="visible",
                  border=False)
        st.metric('Profit Volleinspeisung', str(result_df.iloc[0].solar_production_pure_profit) + ' €',
                  delta=str(0), delta_color="normal", help=None, label_visibility="visible",
                  border=False)
        st.markdown('---')
        st.metric('Eigenverbrauch Einsparung Dynamisch', str(int(
            _daily.consumption.sum() * arbeits_price - _daily.dyn_costs.sum() + _daily.grid_export_dyn_prices.sum() * einspeise_price)) + ' €',
                  delta=str(
                      int(_daily.grid_import_dyn_prices.sum())) + ' kWh Bezug',
                  delta_color="off", help=None,
                  label_visibility="visible",
                  border=False)

    with rr_column:
        st.metric('Bezug Dynamische Preise', str(-1 * result_df.iloc[0].grid_import_dyn_prices) + ' kWh',
                                            delta=str(-1 * result_df.iloc[0].dyn_costs) + ' €', delta_color="normal", help=None, label_visibility="visible", border=False)
        st.metric('Bezug Festpreis', str(-1 * result_df.iloc[0].grid_import_norm_prices) + ' kWh',
                  delta=str(-1 * result_df.iloc[0].norm_costs)+ ' €', delta_color="normal", help=None, label_visibility="visible",
                  border=False)
        st.metric('Mieter Einsparung', str(result_df.iloc[0].tenant_costs_savings) + ' €',
                  delta=str(int(_daily.consumption.sum() * arbeits_price)) + ' € - ' + str(int(_daily.consumption.sum() * selling_price)) + '€', delta_color="off", help=None, label_visibility="visible",
                  border=False)

        st.markdown('---')
        st.metric('Eigenverbrauch Einsparung', str(int(
            _daily.consumption.sum() * arbeits_price - _daily.norm_costs.sum() + _daily.grid_export_norm_prices.sum() * einspeise_price)) + ' €',
                  delta=str(
                      int(_daily.grid_import_norm_prices.sum())) + ' kWh Bezug',
                  delta_color="off", help=None,
                  label_visibility="visible",
                  border=False)