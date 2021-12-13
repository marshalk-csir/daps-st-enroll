import streamlit as st
from datetime import date
import pandas as pd
import numpy as np

#Date
from datetime import datetime
from datetime import date

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.plot import add_changepoints_to_plot
import holidays
from streamlit import caching
from plotly import graph_objs as go
import plotly.express as px

# START = "2010-01-01"
# TODAY = date.today().strftime("%Y-%m-%d")

st.title('Data Analytics for Post-school')
# st.write('Exploring DHET Data')

def load_data():
    unisa_df=pd.read_csv('./All_Inst_Status.csv')
    #del unisa_df
    unisa_df['ENROL_STATUS_DATE'] = pd.to_datetime(unisa_df['ENROL_STATUS_DATE']) # CHANGE TYPE
    #unisa_df['REPORT_YEAR'] = pd.to_datetime(unisa_df.REPORT_YEAR, format='%Y-%M-%D')
    #unisa_df["REPORT_YEAR"].replace(unisa_df['REPORT_YEAR'].dt.to_period('Y'), inplace=True)
    #unisa_df['REPORT_YEAR'] = pd.DatetimeIndex(unisa_df['REPORT_YEAR']).year
    #unisa_df.reset_index(inplace=True)
    return unisa_df

# def load_data_unisa():
#     unisa_df=pd.read_csv('./Unisa_df.csv')
#     #del unisa_df
#     unisa_df['REPORT_YEAR'] = pd.to_datetime(unisa_df['REPORT_YEAR']) # CHANGE TYPE
#     #unisa_df['REPORT_YEAR'] = pd.to_datetime(unisa_df.REPORT_YEAR, format='%Y-%M-%D')
#     #unisa_df["REPORT_YEAR"].replace(unisa_df['REPORT_YEAR'].dt.to_period('Y'), inplace=True)
#     #unisa_df['REPORT_YEAR'] = pd.DatetimeIndex(unisa_df['REPORT_YEAR']).year
#     #unisa_df.reset_index(inplace=True)
#     return unisa_df

# def load_data_ukzn():
#     ukzn_df=pd.read_csv('./Ukzn_df.csv')
#     #del unisa_df
#     #ukzn_df['REPORT_YEAR'] = pd.to_datetime(ukzn_df['REPORT_YEAR']) # CHANGE TYPE
#     ukzn_df['REPORT_YEAR'] = pd.to_datetime(ukzn_df.REPORT_YEAR, format='%Y')
#     #ukzn_df["REPORT_YEAR"].replace(ukzn_df['REPORT_YEAR'].dt.to_period('Y'), inplace=True)
#     #unisa_df['REPORT_YEAR'] = pd.to_datetime(unisa_df.REPORT_YEAR, format='%Y-%M-%D')
#     #unisa_df["REPORT_YEAR"].replace(unisa_df['REPORT_YEAR'].dt.to_period('Y'), inplace=True)
#     #unisa_df['REPORT_YEAR'] = pd.DatetimeIndex(unisa_df['REPORT_YEAR']).year
#     #unisa_df.reset_index(inplace=True)
#     return ukzn_df
 

#data_load_state = st.text('Loading data...')
with st.spinner('Loading data...'):
    all_data = load_data()
    st.subheader('All Institutons - Raw data')
    st.write(all_data.tail())
st.success('Loading data... done!')
# unisa_data = load_data_unisa()
# st.subheader('University of South Africa - Raw data')
# st.write(unisa_data.tail())

# ukzn_data = load_data_ukzn()
# st.subheader('University of KwaZulu-Natal - Raw data')
# st.write(ukzn_data.tail())

#data_load_state.text('Loading data... done!')



options = all_data['PROVIDER_NAME'].unique()
selected_option = st.selectbox('Select Academic Institution for prediction', options)

#st.write('You selected:', selected_option)

st.subheader(selected_option + '- Raw Data')
selected_inst_data = all_data.loc[all_data['PROVIDER_NAME'] == selected_option]
st.write(selected_inst_data)

df_types = pd.DataFrame(selected_inst_data.dtypes, columns=['Data Type'])
numerical_cols = df_types[~df_types['Data Type'].isin(['object', 'bool'])].index.values

df_types['Count'] = selected_inst_data.count()
df_types['Unique Values'] = selected_inst_data.nunique()
df_types['Min'] = selected_inst_data[numerical_cols].min()
df_types['Max'] = selected_inst_data[numerical_cols].max()
df_types['Average'] = selected_inst_data[numerical_cols].mean()
df_types['Median'] = selected_inst_data[numerical_cols].median()
df_types['St. Dev.'] = selected_inst_data[numerical_cols].std()

st.subheader('Summary of ' + selected_option + '\'s Data')
st.write(df_types.astype(str))

forc_inst_data = selected_inst_data.drop(['PROVIDER_NAME'], axis = 1)
#st.write(forc_inst_data)

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Line(x=forc_inst_data['ENROL_STATUS_DATE'], y=forc_inst_data['Enrol_No'], name="Enrolment Per Year"))
	#fig.add_trace(go.Scatter(x=selected_inst_data['REPORT_YEAR'], y=selected_inst_data['PROVIDER_NAME'], name="stock_close"))
	fig.layout.update(title_text='Time Series data for ' + selected_option, xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()



# def plot_all_inst():
#     fig2 = px.line(all_data, x=all_data['REPORT_YEAR'], y=all_data['Enrol_No'], color=all_data['PROVIDER_NAME'], symbol=all_data['PROVIDER_NAME'])
#     fig2.layout.update(title_text='Time Series data for all Institutions', xaxis_rangeslider_visible=True)
#     #fig2.show()
#     st.plotly_chart(fig2)
# plot_all_inst()

# Predict forecast with Prophet.
df_train = forc_inst_data[['ENROL_STATUS_DATE','Enrol_No']]
df_train = df_train.rename(columns={"ENROL_STATUS_DATE": "ds", "Enrol_No": "y"})


st.subheader('Predict ' + selected_option + '\'s enrollment')

n_years = st.slider('Choose Years to Forecast For:', 1, 10)
period = n_years * 365

st.markdown(f'##### {selected_option}\'s enrollment for the next **{n_years}** years')

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period, freq='D')
forecast = m.predict(future)


# Show and plot forecast
#st.subheader('Forecast data for ' + selected_option)
st.write(forecast.tail())

st.markdown("Visualization of Forcast Rests for " + selected_option)
fig2 = m.plot_components(forecast)
st.write(fig2)

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# def plot_raw_pred():
#     fig1 = go.Figure()
#     fig1.add_trace(go.Bar(x=df_train['ds'], y=df_train['yhat'], name="Enrolment Per Year"))
# 	#fig.add_trace(go.Scatter(x=selected_inst_data['REPORT_YEAR'], y=selected_inst_data['PROVIDER_NAME'], name="stock_close"))
#     fig1.layout.update(title_text='Time Series data for ' + selected_option, xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig1)
    
# plot_raw_pred()

# st.markdown("Cross Validation for " + selected_option)
# df_cv = cross_validation(m, initial='730 days', period=period, horizon = '365 days')
# st.write(df_cv.tail())

# st.markdown("Performance matrics for " + selected_option)
# df_p = performance_metrics(df_cv)
# st.write(df_p.head())

# fig = plot_cross_validation_metric(df_cv, metric='rmse')

# file = st.file_uploader("Upload file", type=['csv'])

# st.write(file)



# def load_data(tinker):
#     unisa_df = yf.download(ticker, START, TODAY)
#     unisa_df=pd.read_csv('./UNISA_enrol_Ready.csv')
#     unisa_df.reset_index(inplace=True)
#     return unisa_df
#@st.cache


# data_load_state = st.text('Loading data...')
# unisa_df = load_data(selected_option)
# data_load_state.text('Loading data... done!')

# #unisa_df['REPORT_YEAR'] = pd.to_datetime(unisa_df.REPORT_YEAR, format='%Y')
# #unisa_df["REPORT_YEAR"].replace(unisa_df['REPORT_YEAR'].dt.to_period('Y'), inplace=True)

# st.subheader('Raw data')
# st.write(unisa_df.tail())

# # SUMMARY
# # def explore(unisa_df):
# df_types = pd.DataFrame(unisa_df.dtypes, columns=['Data Type'])
# numerical_cols = df_types[~df_types['Data Type'].isin(['object', 'bool'])].index.values

# df_types['Count'] = unisa_df.count()
# df_types['Unique Values'] = unisa_df.nunique()
# df_types['Min'] = unisa_df[numerical_cols].min()
# df_types['Max'] = unisa_df[numerical_cols].max()
# df_types['Average'] = unisa_df[numerical_cols].mean()
# df_types['Median'] = unisa_df[numerical_cols].median()
# df_types['St. Dev.'] = unisa_df[numerical_cols].std()
    
# st.subheader('Summary')
# st.write(df_types.astype(str))


#     # return unisa_df

# # Plot raw data
# df_plot = px.unisa_df
# fig = px.line(df_plot, x='REPORT_YEAR', y="Enrol_No")
# fig.show()