import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import streamlit as st
import statsmodels.api as sm


#Load Data
def load_data():
    df = pd.read_csv("dashboard/PRSA_Data_Aotizhongxin_20130301-20170228.csv", delimiter=',')
    return df

#helper function

#missing Value handling
def missing_values_handling(data):
    for i in data.columns:
        if data[i].dtype in ['float64', 'int64']:
            data[i] = data[i].fillna(data[i].mean())
        else:
            data[i] = data[i].fillna(data[i].mode()[0])
    return data

#outlier handling
def outlier_handling(data):
    
    data = pd.to_numeric(data, errors='coerce')
    
    while True:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        MIN, MAX = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        
        outlier = (data < MIN) | (data > MAX)
        data = data.mask(outlier, data.median())
        
        outliers_count = outlier.sum()
        
        if outliers_count == 0:
            break
    
    return data


#Temp data
def temp_data(data):
    temp = data.groupby(['year', 'month'])['TEMP'].mean().reset_index()
    temp_data = pd.DataFrame(temp)
    return temp_data

#plotting average temperature
def avg_temp_dif(data):
    avg_temp_of_month = data.groupby(['year', 'month'])['TEMP'].mean().reset_index()
    avg_temp_of_month['month'] = avg_temp_of_month['month'].apply(lambda x: calendar. month_name[x])
    avg_temp_of_month.rename(columns={
    "TEMP": "Average Temperature"
    }, inplace=True)
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=avg_temp_of_month, x='month', y='Average Temperature', hue='year')
    plt.title('Average temperature difference by year')
    st.pyplot(plt)


#avarage temperature per month
def avg_temp_month(data):
    plt.figure(figsize=(14, 7))
    avg_temp_of_month = data.groupby(['year', 'month'])['TEMP'].mean().reset_index()
    avg_temp_of_month['month'] = avg_temp_of_month['month'].apply(lambda x: calendar. month_name[x])
    avg_temp_of_month.rename(columns={
        "TEMP": "Average Temperature"
    }, inplace=True)
    for year in avg_temp_of_month['year'].unique():
        avg_temp_of_month = avg_temp_of_month[avg_temp_of_month['year']==year]
        plt.plot(
            avg_temp_of_month["month"],
            avg_temp_of_month["Average Temperature"],
            marker='o', 
            linewidth=2,
            label=year
        )
    plt.title("Average Temperature per Month", loc="center", fontsize=20)
    plt.xlabel("Month", fontsize=15)
    plt.ylabel("Temperature", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(plt)

#CO avarage 
def co_avarage(data):
    plt.figure(figsize=(14, 7))
    average_co = data.groupby('hour')['CO'].mean().reset_index()
    for year in average_co['hour'].unique():
        plt.plot(
            average_co["hour"],
            average_co["CO"],
            marker='o', 
            linewidth=2,
            label=year
        )
    plt.title("High average CO emission rate", loc="center", fontsize=20)
    plt.xlabel("Hour", fontsize=15)
    plt.ylabel("CO average", fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(plt)

#PM2.5 comparison
def pollutan_comp_25(data):
    x25 = data['CO']
    y25 = data['PM2.5']
    x25 = sm.add_constant(x25)
    pm25 = sm.OLS(y25, x25).fit()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data['CO'], data['PM2.5'], alpha=0.5)
    plt.plot(data['CO'], pm25.predict(x25), color='red', label='Regresi Linier')
    plt.title('CO and PM2.5 relationship')
    plt.xlabel('CO concentration (μg/m³)')
    plt.ylabel('PM2.5 concentration (μg/m³)')
    plt.legend()
    st.pyplot(plt)

#PM10 comparison
def pollutan_comp_10(data):
    x10 = data['CO']
    y10 = data['PM10']
    x10 = sm.add_constant(x10)
    pm10 = sm.OLS(y10, x10).fit()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data['CO'], data['PM10'], alpha=0.5)
    plt.plot(data['CO'], pm10.predict(x10), color='red', label='Regresi Linier')
    plt.title('CO and PM10 relationship')
    plt.xlabel('CO concentration (μg/m³)')
    plt.ylabel('PM10 concentration (μg/m³)')
    plt.legend()
    st.pyplot(plt)



def main():

    st.header("Aotizhongxin Air Quality Data Visualization:sparkles:")

    # Load data
    df = load_data()
    df = missing_values_handling(df)

    variabels = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
    for i in variabels:
        df[i] = outlier_handling(df[i])

    # Sidebar for navigation and options
    with st.sidebar:
        st.image("https://img.icons8.com/pulsar-gradient/48/climate-care.png", use_column_width=True)
        st.title("Aotizhongxin Air Quality")
        options = st.selectbox(" ", ("Overview", "Temperature Analysis", "CO Pollutant"))

    

    # Overview
    if options == "Overview":
        st.subheader("Dataset Overview:")
        st.write("This dataset contains air quality data measured at Aotizhongxin from 2013 to 2017.")

        total, collection = st.columns(2)
        with total:
            total_data = df.shape[0]
            st.metric("Total Data", value=total_data)
        with collection:
            st.metric("year of data collection", value=('2013 - 2017'))
        
        # sample data
        st.subheader("Data Sample:")
        st.dataframe(df.head(10))


    # Temperature Analysis
    elif options == "Temperature Analysis":
        st.subheader("Average Temperature Visualization")

        value, mean, year = st.columns(3)
        with value:
            temp_total = df['TEMP'].shape[0]
            st.metric("Total Temperature Data", value=temp_total)
        with mean:
            temp_mean = df['TEMP'].mean()
            st.metric("Average Temperature Data", value=temp_mean)
        with year:

            st.metric("year of data collection", value=('2013 - 2017'))



        st.subheader("Choose a visualization type:")
        
        
        # select box 
        temp_visualization = st.selectbox("Select visualization type:", ["Average Temperature by Month", "Temperature Trends"])
        
        if temp_visualization == "Average Temperature by Month":

            st.subheader("Average Temperature by Month")
            max, min = st.columns(2)
            with min:
                st.metric("min temperature", value=df['TEMP'].min())
            with max:
                st.metric("max temperature", value=df['TEMP'].max())
            
            avg_temp_month(df)

        elif temp_visualization == "Temperature Trends":
            st.subheader("Temperature Trends")
            max, min = st.columns(2)
            with min:
                st.metric("min temperature", value=df['TEMP'].min())
            with max:
                st.metric("max temperature", value=df['TEMP'].max())
            avg_temp_dif(df)

    # CO Pollutant 
    elif options == "CO Pollutant":
        st.subheader("Correlation of PM2.5 and PM10 with CO")
        #pollutan selecter
        polutant_selection = st.selectbox("Select pollutant to visualize:", ['CO Pollutan', 'CO & PM2.5', 'CO & PM10'])
        
        if polutant_selection == 'CO Pollutan':
            st.subheader("CO Pollutan")
            total, mean, max, min = st.columns(4)
            with total:
                co_total = df['CO'].shape[0]
                st.metric("Total CO Data", value=co_total)
            with mean:
                co_mean = df['CO'].mean()
                st.metric("Average CO Data", value=co_mean)
            with max:
                st.metric("Max CO (μg/m³)", value=df['CO'].max())
            with min:
                st.metric("Min CO (μg/m³)", value=df['CO'].min())
            co_avarage(df)
        elif polutant_selection == 'CO & PM2.5':
            st.subheader("Correlation of CO and PM2.5")
            total, mean, corr = st.columns(3)
            with total:
                pm25_total = df['PM2.5'].shape[0]
                st.metric("Total PM2.5 Data", value=pm25_total)
            with mean:
                pm25_mean = df['PM2.5'].mean()
                st.metric("Average PM2.5 Data", value=pm25_mean)
            with corr:
                st.metric("Correlation", value=df['PM2.5'].corr(df['CO']))

            pollutan_comp_25(df[df['PM2.5'].notnull()])
        elif polutant_selection == 'CO & PM10':
            st.subheader("Correlation of CO and PM10")
            total, mean, corr = st.columns(3)
            with total:
                pm25_total = df['PM10'].shape[0]
                st.metric("Total PM10 Data", value=pm25_total)
            with mean:
                pm25_mean = df['PM10'].mean()
                st.metric("Average PM10 Data", value=pm25_mean)
            with corr:
                st.metric("Correlation", value=df['PM10'].corr(df['CO']))
            pollutan_comp_10(df[df['PM10'].notnull()])

    # Footer 
    st.markdown("""
        <style>
        .footer {
            font-size: 12px;
            color: #7F8C8D;
            text-align: center;
            margin-top: 50px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="footer">©Air Quality Data Visualization</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
