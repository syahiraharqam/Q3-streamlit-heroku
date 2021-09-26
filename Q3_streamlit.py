import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt 
from streamlit_folium import folium_static
import folium
import geopandas as gp
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_regression

#Read CSV files
cases_state = pd.read_csv('cases_state_cleaned.csv')
deaths_state = pd.read_csv('deaths_state_cleaned.csv')
hospital = pd.read_csv('hospital_cleaned.csv')
icu = pd.read_csv('icu_cleaned.csv')
pkrc = pd.read_csv('pkrc_cleaned.csv')
population = pd.read_csv('population_cleaned.csv')

cases = cases_state[(cases_state['date']>='2020-08-01') & (cases_state['date']<'2021-08-01')]
deaths = deaths_state[(deaths_state['date']>='2020-08-01') & (deaths_state['date']<'2021-08-01')]
hosp = hospital[(hospital['date']>='2020-08-01') & (hospital['date']<'2021-08-01')]

data = [deaths, hosp]
count = 0
for i in data:
    if(count==0):
        merged_df = cases.merge(i, on=['state','date'])
    else:
        merged_df = merged_df.merge(i, on=['state','date'])
    count += 1

st.title('Assignment 1 - Question 3')

st.markdown('Prepared by:')
st.markdown('1. Nurul Syahirah Binti Mohd Arqam (1191302465)')
st.markdown('2. Koogan Letchumanan (1181102004)')
st.markdown('3. Kan Chern Hann (1181101441)')


#PART I
st.header('Part (i) - Exploratory Data Analysis')

st.markdown('For Question 3, the timeframe of the data that is used is from 01-08-2020 untik 31-07-2021.')
st.markdown('The missing values were identified during the data cleaning proccess. Due to the abundance of null data values, the rows with NA values are removed. When dealing with sensitive data like COVID-19, filling in the null values with the mean may not be appropriate.')
st.markdown('The outliers can be identified using a boxplot. The comparative boxplot figures below are on the attribute: *cases_new*. The data set used is from the fourth quarter part of the year. We decided to limit the data size as we found that by using a year worth of data leads to an insurmountable amount of outliers.')

from PIL import Image 
boxplot = Image.open('boxplot_casesQ4.jpeg')
st.image(boxplot, width=1000)

st.success("From the comparative boxplot above, we noticed the boxes of Perlis and W.P. Putrajaya are too narrow to be clearly analyzed. Selangor has the widest variation of new cases values as its box is the widest among the boxes of other states. Outliers can be found in all states except for Sarawak and W.P. Labuan.")

#PART II
st.header('Part (ii) - States exhibiting Strong Correlation with:')

cases_new_df = cases[['date', 'state', 'cases_new']]

#Dataframe used to find the correlation between the states is dependent on the cases_new values of each day for the selected period of 1-year.
corr_df = pd.DataFrame(cases_new_df.date.unique(), columns=['date'])
states = cases_new_df.state.unique()
for i in states:
    col = cases_new_df[cases_new_df['state']==i]
    col = col.drop(columns=['state'])
    col = col.rename(columns={'cases_new': i})
    corr_df = corr_df.merge(col)

corr = corr_df.corr()

st.markdown('The data attribute used to find the correlation is: *cases_new*.')

heatmap_ii = Image.open('heatmap_ii.jpeg')
st.image(heatmap_ii, width=1000, caption='Correlation Heatmap of New Cases Between States')

state_ii = st.selectbox('Choose a state', ['Pahang', 'Johor'])

if(state_ii=='Pahang'):
    #Filter states with correlation value greater than 0.7
    pahang_corr = corr[['Pahang']]>0.7
    pahang_corr = pahang_corr[(pahang_corr['Pahang'] == True)]
    pahang_corr = pahang_corr.reset_index().rename(columns={'index': 'States'})
    pahang_corr = pahang_corr.drop(columns='Pahang')
    pahang_corr_val = corr[['Pahang']]
    pahang_corr_val = pahang_corr_val.reset_index().rename(columns={'index': 'States', 'Pahang': 'corr_val'})
    pahang_corr = pahang_corr.merge(pahang_corr_val)
    pahang_corr = pahang_corr.set_index('States')
    pahang_corr = pahang_corr.drop('Pahang', axis=0)
    pahang_corr
    st.success('The states that are highly correlated with Pahang have a correlation coefficient value above 0.7. The correlated states are: Kedah, Melaka, Negeri Sembilan, Pulau Pinang, Selangor, Terengganu, W.P. Kuala Lumpur, and W.P. Putrajaya. All the mentioned states have a strong positive correlation with Pahang. From the correlation heatmap, we can tell that there are no strong negative correlation between the states. Kedah has the highest correlation coefficient value with 0.899. A strong positive correlation indicates that as the number of daily new cases increase in Pahang, the number of daily new cases in the correlated states also increase.')

elif(state_ii=='Johor'):
    #Filter states with correlation value greater than 0.7
    johor_corr = corr[['Johor']]>0.7
    johor_corr = johor_corr[(johor_corr['Johor'] == True)]
    johor_corr = johor_corr.reset_index().rename(columns={'index': 'States'})
    johor_corr = johor_corr.drop(columns='Johor')
    johor_corr_val = corr[['Johor']]
    johor_corr_val = johor_corr_val.reset_index().rename(columns={'index': 'States', 'Johor': 'corr_val'})
    johor_corr = johor_corr.merge(johor_corr_val)
    johor_corr = johor_corr.set_index('States')
    johor_corr = johor_corr.drop('Johor', axis=0)
    johor_corr
    st.success('The states that are highly correlated with Johor have a correlation coefficient value above 0.7. The correlated states are: Pulau Pinang, Selangor, and W.P. Kuala Lumpur. All the mentioned states have a strong positive correlation with Johor. From the correlation heatmap, we can tell that there are no strong negative correlation between the states. Selangor has the highest correlation coefficient value with 0.740. A strong positive correlation indicates that as the number of daily new cases increase in Johor, the number of daily new cases in the correlated states also increase.')


#PART III
st.header('Part (iii) - Strong Features to Daily Cases for:')

st.markdown("Two feature selection methods are used: Pearson's correlation coefficient, and mutual information (or information gain).")
st.markdown('Mutual information or information gain have the same meaning. In the context of decision tree, the term information gain is used while in feature selection, mutual information is used. The greater the mutual information/information gain, the greater the relationship between the two variables.')

state_iii = st.selectbox('Choose a state', ['Pahang', 'Kedah', 'Johor', 'Selangor'], key='partiii')

if(state_iii=='Pahang'):
    pahang_df = merged_df[merged_df['state']=='Pahang']
    pahang_df = pahang_df.drop(columns=['date', 'state'])
    p_corr = pahang_df.corr(method ='pearson')

    st.markdown("**Pearson's Correlation Coefficient**")
    p_iii = Image.open('p_iii.jpeg')
    st.image(p_iii, width=1000, caption='Correlation Heatmap Between Features')

    #Filter states with correlation value greater than 0.7
    pahang_cases = p_corr[['cases_new']]>0.7
    pahang_cases = pahang_cases[(pahang_cases['cases_new'] == True)]
    pahang_cases = pahang_cases.reset_index().rename(columns={'index': 'Features'})
    pahang_cases = pahang_cases.drop(columns='cases_new')
    pahang_cases_val = p_corr[['cases_new']]
    pahang_cases_val = pahang_cases_val.reset_index().rename(columns={'index': 'Features', 'cases_new': 'corr_val_pearson'})
    pahang_cases = pahang_cases.merge(pahang_cases_val)
    pahang_cases = pahang_cases.set_index('Features')
    pahang_cases = pahang_cases.drop('cases_new', axis=0)
    pahang_cases.reset_index(inplace=True)
    pahang_cases
    st.markdown('The above dataframe includes the features that exhibit a strong correlation with the _cases_new_ attribute.')


    st.markdown('**Mutual Information**')
    #Display barchart of the mutual information (or information gain) between cases_new and other features
    pahang_X = pahang_df.drop(columns='cases_new')
    pahang_y = pahang_df['cases_new']
    p_imp = mutual_info_regression(pahang_X, pahang_y, random_state=0)
    p_feat_imp = pd.Series(p_imp, pahang_X.columns)
    p_feat_imp = p_feat_imp.sort_values()

    p_f = Image.open('p_feat_imp.jpeg')
    st.image(p_f, width=700, caption='Mutual Information Between New Cases and Features')

    #Display the top 5 highest features
    p_imp_df = pd.DataFrame(p_feat_imp.tail(5))
    p_imp_df = p_imp_df.rename(columns={0: 'corr_val_gain'})
    p_imp_df.index.names = ['Features']
    p_imp_df.reset_index(inplace=True)
    p_imp_df

    st.markdown('The above dataframe includes the top 5 strongest features to the _cases_new_ attribute.')
    
    pahang_corr_feat = pahang_cases.merge(p_imp_df, how='inner', on='Features')
    pahang_corr_feat = pahang_corr_feat.drop(columns=['corr_val_pearson', 'corr_val_gain'])
    pahang_corr_feat

    st.success('Based on the results of the two methods, we extract the common strong features. So, the daily new cases in Pahang are highly dependent on the features: _admitted_covid_, _discharged_covid_, and _hosp_covid_.')

elif(state_iii=='Kedah'):
    kedah_df = merged_df[merged_df['state']=='Kedah']
    kedah_df = kedah_df.drop(columns=['date', 'state'])
    k_corr = kedah_df.corr(method ='pearson')

    st.markdown("**Pearson's Correlation Coefficient**")
    k_iii = Image.open('k_iii.jpeg')
    st.image(k_iii, width=1000, caption='Correlation Heatmap Between Features')
    
    #Filter states with correlation value greater than 0.7
    kedah_cases = k_corr[['cases_new']]>0.7
    kedah_cases = kedah_cases[(kedah_cases['cases_new'] == True)]
    kedah_cases = kedah_cases.reset_index().rename(columns={'index': 'Features'})
    kedah_cases = kedah_cases.drop(columns='cases_new')
    kedah_cases_val = k_corr[['cases_new']]
    kedah_cases_val = kedah_cases_val.reset_index().rename(columns={'index': 'Features', 'cases_new': 'corr_val_pearson'})
    kedah_cases = kedah_cases.merge(kedah_cases_val)
    kedah_cases = kedah_cases.set_index('Features')
    kedah_cases = kedah_cases.drop('cases_new', axis=0)
    kedah_cases.reset_index(inplace=True)
    kedah_cases
    st.markdown('The above dataframe includes the features that exhibit a strong correlation with the _cases_new_ attribute.')


    st.markdown('**Mutual Information**')
    #Display barchart of the mutual information (or information gain) between cases_new and other features
    kedah_X = kedah_df.drop(columns='cases_new')
    kedah_y = kedah_df['cases_new']
    k_imp = mutual_info_regression(kedah_X, kedah_y, random_state=0)
    k_feat_imp = pd.Series(k_imp, kedah_X.columns)
    k_feat_imp = k_feat_imp.sort_values()

    k_f = Image.open('k_feat_imp.jpeg')
    st.image(k_f, width=700, caption='Mutual Information Between New Cases and Features')

    #Display the top 5 highest features
    k_imp_df = pd.DataFrame(k_feat_imp.tail(5))
    k_imp_df = k_imp_df.rename(columns={0: 'corr_val_gain'})
    k_imp_df.index.names = ['Features']
    k_imp_df.reset_index(inplace=True)
    k_imp_df

    st.markdown('The above dataframe includes the top 5 strongest features to the _cases_new_ attribute.')
    
    kedah_corr_feat = kedah_cases.merge(k_imp_df, how='inner', on='Features')
    kedah_corr_feat = kedah_corr_feat.drop(columns=['corr_val_pearson', 'corr_val_gain'])
    kedah_corr_feat

    st.success('Based on the results of the two methods, we extract the common strong features. So, the daily new cases in Kedah are highly dependent on the features: _cases_recovered_, _beds_covid_, _admitted_covid_ and _hosp_covid_.')

elif(state_iii=='Johor'):
    johor_df = merged_df[merged_df['state']=='Johor']
    johor_df = johor_df.drop(columns=['date', 'state'])
    j_corr = johor_df.corr(method ='pearson')

    st.markdown("**Pearson's Correlation Coefficient**")

    j_iii = Image.open('j_iii.jpeg')
    st.image(j_iii, width=1000, caption='Correlation Heatmap Between Features')

    #Filter states with correlation value greater than 0.7
    johor_cases = j_corr[['cases_new']]>0.7
    johor_cases = johor_cases[(johor_cases['cases_new'] == True)]
    johor_cases = johor_cases.reset_index().rename(columns={'index': 'Features'})
    johor_cases = johor_cases.drop(columns='cases_new')
    johor_cases_val = j_corr[['cases_new']]
    johor_cases_val = johor_cases_val.reset_index().rename(columns={'index': 'Features', 'cases_new': 'corr_val_pearson'})
    johor_cases = johor_cases.merge(johor_cases_val)
    johor_cases = johor_cases.set_index('Features')
    johor_cases = johor_cases.drop('cases_new', axis=0)
    johor_cases.reset_index(inplace=True)
    johor_cases
    st.markdown('The above dataframe includes the features that exhibit a strong correlation with the _cases_new_ attribute.')


    st.markdown('**Mutual Information**')
    #Display barchart of the mutual information (or information gain) between cases_new and other features
    johor_X = johor_df.drop(columns='cases_new')
    johor_y = johor_df['cases_new']
    j_imp = mutual_info_regression(johor_X, johor_y, random_state=0)
    j_feat_imp = pd.Series(j_imp, johor_X.columns)
    j_feat_imp = j_feat_imp.sort_values()

    j_f = Image.open('j_feat_imp.jpeg')
    st.image(j_f, width=700, caption='Mutual Information Between New Cases and Features')

    #Display the top 5 highest features
    j_imp_df = pd.DataFrame(j_feat_imp.tail(5))
    j_imp_df = j_imp_df.rename(columns={0: 'corr_val_gain'})
    j_imp_df.index.names = ['Features']
    j_imp_df.reset_index(inplace=True)
    j_imp_df

    st.markdown('The above dataframe includes the top 5 strongest features to the _cases_new_ attribute.')
    
    johor_corr_feat = johor_cases.merge(j_imp_df, how='inner', on='Features')
    johor_corr_feat = johor_corr_feat.drop(columns=['corr_val_pearson', 'corr_val_gain'])
    johor_corr_feat

    st.success('Based on the results of the two methods, we extract the common strong features. So, the daily new cases in Johor are highly dependent on the features: _admitted_covid_, _discharged_covid_, and _hosp_covid_.')


elif(state_iii=='Selangor'):
    selangor_df = merged_df[merged_df['state']=='Selangor']
    selangor_df = selangor_df.drop(columns=['date', 'state'])
    s_corr = selangor_df.corr(method ='pearson')

    st.markdown("**Pearson's Correlation Coefficient**")
    s_iii = Image.open('s_iii.jpeg')
    st.image(s_iii, width=1000, caption='Correlation Heatmap Between Features')

    #Filter states with correlation value greater than 0.7
    selangor_cases = s_corr[['cases_new']]>0.7
    selangor_cases = selangor_cases[(selangor_cases['cases_new'] == True)]
    selangor_cases = selangor_cases.reset_index().rename(columns={'index': 'Features'})
    selangor_cases = selangor_cases.drop(columns='cases_new')
    selangor_cases_val = s_corr[['cases_new']]
    selangor_cases_val = selangor_cases_val.reset_index().rename(columns={'index': 'Features', 'cases_new': 'corr_val_pearson'})
    selangor_cases = selangor_cases.merge(selangor_cases_val)
    selangor_cases = selangor_cases.set_index('Features')
    selangor_cases = selangor_cases.drop('cases_new', axis=0)
    selangor_cases.reset_index(inplace=True)
    selangor_cases
    st.markdown('The above dataframe includes the features that exhibit a strong correlation with the _cases_new_ attribute.')


    st.markdown('**Mutual Information**')
    #Display barchart of the mutual information (or information gain) between cases_new and other features
    selangor_X = selangor_df.drop(columns='cases_new')
    selangor_y = selangor_df['cases_new']
    s_imp = mutual_info_regression(selangor_X, selangor_y)
    s_feat_imp = pd.Series(s_imp, selangor_X.columns)
    s_feat_imp = s_feat_imp.sort_values()

    s_f = Image.open('s_feat_imp.jpeg')
    st.image(s_f, width=700, caption='Mutual Information Between New Cases and Features')

    #Display the top 5 highest features
    s_imp_df = pd.DataFrame(s_feat_imp.tail(5))
    s_imp_df = s_imp_df.rename(columns={0: 'corr_val_gain'})
    s_imp_df.index.names = ['Features']
    s_imp_df.reset_index(inplace=True)
    s_imp_df

    st.markdown('The above dataframe includes the top 5 strongest features to the _cases_new_ attribute.')
    
    selangor_corr_feat = selangor_cases.merge(s_imp_df, how='inner', on='Features')
    selangor_corr_feat = selangor_corr_feat.drop(columns=['corr_val_pearson', 'corr_val_gain'])
    selangor_corr_feat

    st.success('Based on the results of the two methods, we extract the common strong features. So, the daily new cases in Selangor are highly dependent on the features: _cases_recovered_, _beds_covid_, _admitted_covid_, _discharged_covid_, and _hosp_covid_.')


st.header('Part (iv) - Comparison of Regression and Classification Models to Predict Daily Cases')

state_iv = st.selectbox('Choose a state', ['Pahang', 'Kedah', 'Johor', 'Selangor'], key='partiv')

eva_r_index = ['MAE', 'MSE', 'RMSE']

if(state_iv=='Pahang'):
    p_df = merged_df[merged_df['state']=='Pahang']

    st.markdown('**Regression Models**')
    p_r = {'Linear Regression':[18.72, 663.01, 25.75], 'Decision Tree Regressor':[18.71, 1230.09, 35.07]}
    p_r_df = pd.DataFrame(p_r, index=['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'])
    p_r_df

    p_eva_r = Image.open('p_eva_r.jpeg')
    st.image(p_eva_r, width=700)

    st.success("Based on the grouped barchart above, the mean absolute error (MAE) for both linear regression and decision tree regressor models are approximately the same. Though, the mean squared error (MSE) and root mean squared error (RMSE) for decision tree regressor model is higher than the linear regression model. As the lower the MAE, MSE, and RMSE values are, the better the model is. The prediction of the number of daily new cases in Pahang using **linear regression** is better in comparison to decision tree regressor. This is due to the low values of MSE and RMSE for the linear regression model.")

    st.markdown('**Classification Models**')

    p_c = {'Naive Bayes':[94.52, 85.96, 85.96], 'K-Nearest Neighbors':[89.04, 70.07, 55.48]}
    p_c_df = pd.DataFrame(p_c, index=['Accuracy', 'Precision', 'Recall'])
    p_c_df

    p_eva_c = Image.open('p_eva_c.jpeg')
    st.image(p_eva_c, width=700)

    st.success("Based on the grouped barchart above, the accuracy, precision, and recall values for Naive Bayes is higher than the K-Nearest Neighbors (KNN). For classification models, we aim for high accuracy, precision, and recall values. Naive Bayes is better at identifying true positives as the precision value is greater than KNN. Due to these reasons, the prediction of daily new binned cases in Pahang using **Naive Bayes** is better in comparison to KNN.")
    
elif(state_iv=='Kedah'):
    st.markdown('**Regression Models**')
    k_r = {'Linear Regression':[54.72, 5184.10, 72.00], 'Decision Tree Regressor':[37.58, 4433.58, 66.59]}
    k_r_df = pd.DataFrame(k_r, index=['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'])
    k_r_df

    k_eva_r = Image.open('k_eva_r.jpeg')
    st.image(k_eva_r, width=700)

    st.success("Based on the grouped barchart above, the mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE) for linear regression model is higher than the decision tree regressor model. As the lower the MAE, MSE, and RMSE values are, the better the model is. The prediction of the number of daily new cases in Kedah using **decision tree regressor** is better in comparison to linear regression. This is due to the low values of MAE, MSE, and RMSE for the decision tree regressor model.")

    st.markdown('**Classification Models**')

    k_c = {'Naive Bayes':[94.52, 66.67, 97.18], 'K-Nearest Neighbors':[95.89, 70.00, 97.89]}
    k_c_df = pd.DataFrame(k_c, index=['Accuracy', 'Precision', 'Recall'])
    k_c_df

    k_eva_c = Image.open('k_eva_c.jpeg')
    st.image(k_eva_c, width=700)

    st.success("Based on the grouped barchart above, the accuracy, precision, and recall values for K-Nearest Neighbors (KNN) is slightly higher than Naive Bayes. For classification models, we aim for high accuracy, precision, and recall values. KNN is better at identifying true positives as the precision value is greater than Naive Bayes. Based on the accuracy, the number of correct predictions made by KNN is more compared to Naive Bayes. Due to these reasons, the prediction of daily new binned cases in Kedah using **KNN** is better in comparison to Naive Bayes. Though, both models can be further optimized for better results as these models have approximately similar results.")

elif(state_iv=='Johor'):
    st.markdown('**Regression Models**')
    j_r = {'Linear Regression':[90.40, 20739.32, 144.01], 'Decision Tree Regressor':[91.74, 30401.99, 174.36]}
    j_r_df = pd.DataFrame(j_r, index=['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'])
    j_r_df

    j_eva_r = Image.open('j_eva_r.jpeg')
    st.image(j_eva_r, width=700)

    st.success("Based on the grouped barchart above, the mean absolute error (MAE) for decision tree regressor is slighly higher as compared to the linear regression model. Though, the mean squared error (MSE) and root mean squared error (RMSE) for decision tree regressor model is higher than the linear regression model. As the lower the MAE, MSE, and RMSE values are, the better the model is. The prediction of the number of daily new cases in Johor using **linear regression** is better in comparison to decision tree regressor. This is due to the low values of MSE and RMSE for the linear regression model.")

    st.markdown('**Classification Models**')

    j_c = {'Naive Bayes':[82.19, 41.07, 54.53], 'K-Nearest Neighbors':[91.78, 45.20, 45.67]}
    j_c_df = pd.DataFrame(j_c, index=['Accuracy', 'Precision', 'Recall'])
    j_c_df

    j_eva_c = Image.open('j_eva_c.jpeg')
    st.image(j_eva_c, width=700)

    st.success("Based on the grouped barchart above, the accuracy, and precision for K-Nearest Neighbors (KNN) is higher than Naive Bayes. Though, the recall value for Naive Bayes is greater than KNN. For classification models, we aim for high accuracy, precision, and recall values. Naive Bayes is better at identifying true positives as the precision value is greater than KNN. Despite the recall value of Naive Bayes being higher, KNN has greater accuracy and precision. Based on the accuracy, the number of correct predictions made by KNN is more compared to Naive Bayes. Due to these reasons, the prediction of daily new binned cases in Johor using **KNN** is better in comparison to Naive Bayes.")

elif(state_iv=='Selangor'):
    st.markdown('**Regression Models**')
    s_r = {'Linear Regression':[224.80, 85272.03, 292.01], 'Decision Tree Regressor':[224.59, 131661.66, 362.85]}
    s_r_df = pd.DataFrame(s_r, index=['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'])
    s_r_df

    s_eva_r = Image.open('s_eva_r.jpeg')
    st.image(s_eva_r, width=700)

    st.success("Based on the grouped barchart above, the mean absolute error (MAE) for both linear regression and decision tree regressor models are approximately the same. Though, the mean squared error (MSE) and root mean squared error (RMSE) for decision tree regressor model is higher than the linear regression model. As the lower the MAE, MSE, and RMSE values are, the better the model is. The prediction of the number of daily new cases in Selangor using **linear regression** is better in comparison to decision tree regressor. This is due to the low values of MSE and RMSE for the linear regression model.")

    st.markdown('**Classification Models**')

    s_c = {'Naive Bayes':[91.78, 62.5, 95.77], 'K-Nearest Neighbors':[100.00, 100.00, 100.00]}
    s_c_df = pd.DataFrame(s_c, index=['Accuracy', 'Precision', 'Recall'])
    s_c_df

    s_eva_c = Image.open('s_eva_c.jpeg')
    st.image(s_eva_c, width=700)

    st.success("Based on the grouped barchart above, the accuracy, precision, and recall values for K-Nearest Neighbors (KNN) is higher than Naive Bayes. For classification models, we aim for high accuracy, precision, and recall values. KNN is better at identifying true positives as the precision value is greater than Naive Bayes. Based on the accuracy, the number of correct predictions made by KNN is more compared to Naive Bayes. KNN is also able to achieve 100 percent in accuracy, precision, and recall. Due to these reasons, the prediction of daily new binned cases in Selangor using **KNN** is better in comparison to Naive Bayes.")

st.markdown("**Conclusion**")

st.markdown("We were not able to compare the classification and regression models as these models use different evaluation metrics. Classification and regression models are used in different situations. In order to predict the number of daily new cases in a state, the regression model is used. While classification models are used for the prediction of the daily new cases in categories (low, medium, high). Thus, the comparison between classification and regression models is unavailable.")

st.success("Based on the comparison of the two regression models, the majority of the states have better prediction of the number of daily new cases using linear regression model. Kedah is the only state that prefers to use decision tree regressor model for prediction.")

st.success("Based on the comparison of the two classification models, the majority of the states have better prediction of the daily new cases in categories (low, medium, high) using K-Nearest Neighbors model. Pahang is the only state that prefers to use Naive Bayes model for prediction.")