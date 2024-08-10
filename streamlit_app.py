import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt


st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Dashboard `version 1`')



# Row A
st.markdown('### Total')
col1, col2, col3= st.columns(3)
col1.metric("Number of foreign workers (2023)", "2,048,675", "225,950 ")
col2.metric("Number of Establishments (2023)", "318,775")
col3.metric("Number of foreign workers (2023)", "1,822,725")


def page_one():
    st.title("Number of foreign workers by nationality/residency status (2023)")
    #st.write("This is the home page of our app. Use the sidebar to navigate to other pages.")

    dataNt = pd.read_csv('https://raw.githubusercontent.com/Deymomanka/labor-employment/main/LaborByNationality.csv')
    dfNt = pd.DataFrame(dataNt)
    #dfNt['Total'] = dfNt['Total'].str.replace(',', '').astype(int)

    def remove_commas_and_convert_to_numeric(value):
        if isinstance(value, str):
            value = value.replace(',', '')
        return pd.to_numeric(value)


# Apply the function to the desired columns in the DataFrame
    dfNt.iloc[:, 1:] = dfNt.iloc[:, 1:].applymap(remove_commas_and_convert_to_numeric)


#--------------Number of Foreign Workers by Nationality
    st.markdown('### Number of foreign workers by residency status:')
    fig11 = px.bar(dfNt, x='Country', y='Total')
    fig11.update_xaxes(title='Country')
    fig11.update_yaxes(title='Number Of People')
    st.plotly_chart(fig11)


 #--------------Number of Foreign Workers by shikaku
    st.markdown('### Number of foreign workers by nationality:')

    categories = dfNt.columns[2:]

# Create the grouped bar chart using Plotly
    fig12 = go.Figure()

    for category in categories:
        fig12.add_trace(go.Bar(
            x=dfNt['Country'],
            y=dfNt[category],
            name=category
        ))

    # Set the title and axis labels
    fig12.update_layout(
        title='Grouped Bar Chart',
        xaxis_title='Country',
        yaxis_title='Number Of People'
    )

    # Show the chart
    st.plotly_chart(fig12, config={'displayModeBar': False, 'responsive': True}, use_container_width=True)


    
def page_two():
    st.title("Number of Business Establishments and Foreign Workers by Industry")
    #st.write("This is the about page. Here, you can learn more about our app and the team behind it.")
    # Row B
    # ~~~~~~~~~~~~Load the data for DonatChart
    dataDC = pd.read_csv('https://raw.githubusercontent.com/Deymomanka/labor-employment/main/IndustryENG.csv')
    dfDC = pd.DataFrame(dataDC)
    # df['2021'] = df['2021'].str.replace(',', '')
    #dfDC['Proportion'] = dfDC['Proportion'].astype(int)
    #dfDC.iloc[:, 1:] = dfDC.iloc[:, 1:].apply(lambda x: x.str.replace(',', '').astype(int))
    #dfDC['Proportion'] = dfDC['Proportion'].apply(lambda x: x.str.replace('%', '').astype(int))
    dfDC['Proportion'] = dfDC['Proportion'].str.replace(',', '').str.replace('%', '').astype(int) / 10
    dfDC['People'] = dfDC['People'].apply(lambda x: x.replace('\xa0', '')).astype(int)

    #~~~~~~~~~~~~~~~~~~~~~~Load the Data for Company
    dataDC2 = pd.read_csv("https://raw.githubusercontent.com/Deymomanka/labor-employment/main/byCompany.csv")
    dfDC2 = pd.DataFrame(dataDC2)
    dfDC2['Proportion'] = dfDC2['Proportion'].str.replace(',', '').str.replace('%', '').astype(int) / 10
    dfDC2['People'] = dfDC2['People'].apply(lambda x: x.replace('\xa0', '')).astype(int)

    #----------------Percentage of foreign workers in each industry
    st.markdown('### Percentage of foreign workers in each industry:')
    dfDC.loc[dfDC['Proportion'] < 2, 'Industry ENG'] = 'Other'
    dfDC = dfDC.groupby('Industry ENG').sum().reset_index()
    fig = go.Figure(data=[go.Pie(labels=dfDC['Industry ENG'], values=dfDC['Proportion'], hole=0.4)])
    #fig.update_layout(title='Industry Proportions')
    st.plotly_chart(fig)

    #--------------Number of people
    st.markdown('### Number of Foreign Workers Employed:')
    fig2 = px.bar(dfDC, x='Industry ENG', y='People')
    fig2.update_xaxes(title='Industry')
    fig2.update_yaxes(title='Number of People')
    st.plotly_chart(fig2)


    #----------------Percentage of Establishments Employing Foreign Workers
    st.markdown('### Percentage of Establishments Employing Foreign Workers:')
    dfDC2.loc[dfDC2['Proportion'] < 2, 'Industry ENG'] = 'Other'
    dfDC2 = dfDC2.groupby('Industry ENG').sum().reset_index()
    fig3 = go.Figure(data=[go.Pie(labels=dfDC2['Industry ENG'], values=dfDC2['Proportion'], hole=0.4)])
    #fig.update_layout(title='Industry Proportions')
    st.plotly_chart(fig3)

    #--------------Number of Establishments
    st.markdown('### Number of Establishments Employing Foreign Workers:')
    fig4 = px.bar(dfDC2, x='Industry ENG', y='People')
    fig4.update_xaxes(title='Industry')
    fig4.update_yaxes(title='Number of Establishments')
    st.plotly_chart(fig4)
        

    
def page_three():
    st.title("Number of foreign workers/Establishments Employing Foreign Workers by prefecture")
    #st.write("")
    # fig = plot_map()
    # st.pyplot(fig)
    #==========================
    data = "https://raw.githubusercontent.com/Deymomanka/data_by_continent/main/byPref.csv"
    prf_df = pd.read_csv(data)

    prf_df['Ratio'] = prf_df['Ratio'].str.rstrip('%')
    prf_df['Ratio'] = prf_df['Ratio'].astype(float)

    # Setting the path to the shapefile
    SHAPEFILE = './map/jpn_admbnda_adm1_2019.shp'
    # Read shapefile using Geopandas
    df = gpd.read_file(SHAPEFILE)
    columns_to_drop = ['ADM0_EN', 'ADM0_JA', 'ADM0_PCODE', 'ADM1_JA']
    df = df.drop(columns_to_drop, axis=1)

    merged_df = pd.merge(left=df, right=prf_df, how='left', left_on='ADM1_PCODE', right_on='ADM1_PCODE')

    df1 = merged_df.copy()

    title = 'Number of foreign workers by prefecture'
    col = 'Ratio'
    source = 'Source: Ministry of Health, Labour and Welfare of Japan \nFormat: % '
    vmin = df1[col].min()
    vmax = df1[col].max()
    cmap = 'cool'
    # Create figure and axes for Matplotlib
    fig, ax = plt.subplots(1, figsize=(25, 10))
    # Remove the axis
    ax.axis('off')
    df1.plot(column=col, ax=ax, edgecolor='0.8', linewidth=1, cmap=cmap)
    # Add a title
    ax.set_title(title, fontdict={'fontsize': '25', 'fontweight': '3'})
    # Create an annotation for the data source
    ax.annotate(source, xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left',
                verticalalignment='bottom', fontsize=10)

    # Create colorbar as a legend
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    # Empty array for the data range
    sm._A = []
    # Add the colorbar to the figure
    cbaxes = fig.add_axes([0.15, 0.25, 0.01, 0.4])
    cbar = fig.colorbar(sm, cax=cbaxes)

    st.pyplot(fig)

#=================================================================
    #st.title("Establishments Employing Foreign Workers by prefecture")

    data2 = "https://raw.githubusercontent.com/Deymomanka/labor-employment/main/byPrefEstab.csv"
    prf_df2 = pd.read_csv(data2)
    prf_df2.head()

    prf_df2['Ratio'] = prf_df2['Ratio'].str.rstrip('%')
    prf_df2['Ratio'] = prf_df2['Ratio'] .astype(float)
    prf_df2.head()

    merged_df2 = pd.merge(left=df, right=prf_df2, how='left', left_on='ADM1_PCODE', right_on='ADM1_PCODE')
    df3 = merged_df2.copy()
    df3.head()


    title1 = 'Establishments Employing Foreign Workers'
    col1 = 'Ratio'
    source1 = 'Source: Ministry of Health, Labour and Welfare of Japan \nScale: % '
    vmin1 = df3[col].min()
    vmax1 = df3[col].max()
    cmap1 = 'BuGn'
    # Create figure and axes for Matplotlib
    fig1, ax1 = plt.subplots(1, figsize=(25, 10))
    # Remove the axis
    ax1.axis('off')
    df3.plot(column=col1, ax=ax1, edgecolor='0.8', linewidth=1, cmap=cmap1)
    # Add a title
    ax1.set_title(title1, fontdict={'fontsize': '25', 'fontweight': '3'})
    # Create an annotation for the data source
    ax1.annotate(source1, xy=(0.1, .08), xycoords='figure fraction', horizontalalignment='left', 
                verticalalignment='bottom', fontsize=10)
                
    # Create colorbar as a legend
    sm1 = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin1, vmax=vmax1), cmap=cmap1)
    # Empty array for the data range
    sm1._A = []
    # Add the colorbar to the figure
    cbaxes1 = fig1.add_axes([0.15, 0.25, 0.01, 0.4])
    cbar1 = fig1.colorbar(sm1, cax=cbaxes1)

    st.pyplot(fig1)

def page_four():
    st.title("Cooming soon")
    #st.write("")
    
# Define sidebar options
pages = {
    "Number of foreign workers by nationality (2023):": page_one,
    "Number of Business Establishments and Foreign Workers by Industry (2023)": page_two,
    "Number of foreign workers/Establishments Employing Foreign Workers by prefecture (2023)": page_three,
    "2024 (Cooming soon)": page_four
}

# Define the sidebar
st.sidebar.title("Navigation")
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Display the selected page with the corresponding function
pages[selection]()

st.sidebar.markdown('''
---
Created by [Yuria](https://github.com/Deymomanka/).
''')
