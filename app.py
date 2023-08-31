import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image

@st.cache_data
def load_data():
    df = pd.read_csv("Feature_Engineered_with_Outliers.csv")
    return df


def scatter_plot(df, ax):
    st.subheader("Duration vs Campaign")
    dur_cam = sns.scatterplot(data=df,x='duration', y='campaign', 
                     hue='deposit',
                     alpha=0.6, ax=ax)

    ax.set_xlim([0,42])
    ax.set_ylim([0,42])
    ax.set_ylabel('Number of Calls')
    ax.set_xlabel('Duration of Calls (Minutes)')
    ax.set_title('The Relationship between the Number and Duration of Calls (with Response Result)')

    # Annotation
    ax.axhline(y=5, linewidth=2, color="k", linestyle='--')
    ax.annotate('Higher subscription rate when calls <5',xytext=(35,13),
                 arrowprops=dict(color='k', width=1), xy=(30,6))
    return dur_cam


def kmeans_clustering(df, ax=None):
    st.subheader("Duration vs Campaign")
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(df[['duration','campaign']])
    df['cluster'] = kmeans.labels_

    if ax:
        ax = sns.scatterplot(data=df, x='duration', y='campaign', hue='cluster', ax=ax)
        ax.set_xlim([0, 65])
        ax.set_ylim([0, 65])
        ax.set_title('KMeans Clustering')
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = sns.scatterplot(data=df, x='duration', y='campaign', hue='cluster')
        ax.set_xlim([0, 65])
        ax.set_ylim([0, 65])
        ax.set_title('KMeans Clustering')

    return ax

def agebalance(df,ax):
    #ax = plt.subplots(figsize=(10, 8))
    st.subheader("Age vs Balance")
    dur_cam = sns.scatterplot(x='age', y='balance', data=df,
                              hue='deposit',
                              alpha=0.7, ax=ax)

    ax.set_ylabel('Balance')
    ax.set_xlabel('Age')
    ax.set_title('The Relationship between Age and Balance (with Response Result)')

    # Calculate the threshold value
    threshold = np.percentile(df[df['deposit'] == 'yes']['balance'], 95)

    # Add vertical line to plot at the threshold value
    ax.axhline(y=threshold, color='red', linestyle='--')

    # Annotation
    ax.annotate('Higher subscription rate when balance < 6838', xytext=(50, threshold * 4),
                arrowprops=dict(color='k', width=1), xy=(35, threshold))

    # Display the figure in Streamlit
    return dur_cam


def cluster_analysis(df,ax):
    st.subheader("Age vs Balance")
    # select the relevant features for clustering
    X = df[['age', 'balance']]

    # normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # set the number of clusters
    k = 5

    # train the model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    # predict the cluster labels
    labels = kmeans.predict(X_scaled)

    # add the cluster labels to the original dataset
    df['cluster'] = labels

    # visualize the clusters
    #fig, ax = plt.subplots(figsize=(10, 8))
    fig_sc = sns.scatterplot(data=df, x='age', y='balance', hue='cluster', palette=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628'], ax=ax)
    # Plot the centroids with darker and bigger markers
    centroids = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=50, linewidths=3, color='black', zorder=4)
    plt.title("Clusters with centroids")

    # Display the figure in Streamlit
    return fig_sc

def durcamp(df,ax):
    st.subheader("Duration vs Campaign")
    # select the relevant features for clustering
    X = df[['duration', 'campaign']]

    # normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # set the number of clusters
    k = 5

    # train the model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    # predict the cluster labels
    labels = kmeans.predict(X_scaled)

    # add the cluster labels to the original dataset
    df['cluster'] = labels

    # visualize the clusters
    #fig, ax = plt.subplots(figsize=(10, 8))
    fig_sc = sns.scatterplot(data=df, x='duration', y='campaign', hue='cluster', palette=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628'])
    # Plot the centroids with darker and bigger markers
    centroids = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=50, linewidths=3, color='black', zorder=4)
    plt.title("Clusters with centroids")

    # Display the figure in Streamlit
    return fig_sc

def previouscamp(df,ax):
    st.subheader("Previous vs Campaign")
    # select the relevant features for clustering
    X = df[['previous', 'campaign']]

    # normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # set the number of clusters
    k = 6

    # train the model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    # predict the cluster labels
    labels = kmeans.predict(X_scaled)

    # add the cluster labels to the original dataset
    df['cluster'] = labels
    df=df.drop(df[df['previous']==275].index)

    # visualize the clusters
    #fig, ax = plt.subplots(figsize=(10, 8))
    fig_sc = sns.scatterplot(data=df, x='previous', y='campaign', hue='cluster', palette=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628'])
    # Plot the centroids with darker and bigger markers
    centroids = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids)
    #plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=50, linewidths=3, color='black', zorder=4)
    plt.title("Clusters with centroids")

    # Display the figure in Streamlit
    return fig_sc

def monthcontact(df):
    st.subheader("Month vs Subscription Rate")
    pivot_table = pd.pivot_table(df, index='month', columns='deposit', aggfunc='size')
    percentages = pivot_table.apply(lambda x: x/x.sum()*100, axis=1)
    
    month = pd.DataFrame(df['month'].value_counts())
    month['% Contacted'] = month['month']*100/month['month'].sum()
    month['% Subscription'] = percentages['yes']
    month.drop('month',axis = 1,inplace = True)

    month['Month'] = [5,7,8,6,11,4,2,1,10,9,3,12]
    month = month.sort_values('Month',ascending = True)
    #fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(month.index, month['% Subscription'], label='% Subscription', marker='o')
    ax.plot(month.index, month['% Contacted'], label='% Contacted', marker='o')

    ax.set_title('Subscription vs. Contact Rate by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Subscription and Contact Rate')

    ax.set_xticks(np.arange(1, 13, 1))

    y = month['% Contacted'].max()
    x = month['% Contacted'].idxmax()
    ax.annotate('May: Peak of contact', xy=(x+0.1, y+0.1), xytext=(x+1,y+4), arrowprops=dict(facecolor='black', headwidth=6, width=1, headlength=4), horizontalalignment='left', verticalalignment='top')

    y = month['% Subscription'].max()
    x = month['% Subscription'].idxmax()
    ax.annotate('March: Peak Subscription rate', xy=(x+0.1, y+0.1), xytext=(x+1,y+1), arrowprops=dict(facecolor='black', headwidth=6, width=1, headlength=4), horizontalalignment='left', verticalalignment='top')

    st.pyplot(fig)
    
def ageclassvssub(df,ax):
    st.subheader("Age Class vs Subscription Rate")
    df["age_group"]=None
    df["age_group"].loc[df['age']<30]=20
    df["age_group"].loc[(df['age']>=30) & (df['age']<=39)]=30
    df["age_group"].loc[(df['age']>=40) & (df['age']<=49)]=40
    df["age_group"].loc[(df['age']>=50) & (df['age']<=59)]=50
    df["age_group"].loc[(df['age']>=60) & (df['age']<=69)]=60
    df["age_group"].loc[df['age']>=70]=70
    df['age_group']=df['age_group'].astype(int)
    # Create pivot table
    pivot_table = pd.pivot_table(df, index='age_group', columns='deposit', aggfunc='size')

    # Calculate percentages
    percentages = pivot_table.apply(lambda x: x/x.sum()*100, axis=1)
    #fig, ax = plt.subplots(figsize=(8, 6))
    fig_age = sns.heatmap(percentages, annot=True, fmt='g')
    plt.xlabel('Age Class')
    plt.ylabel('Subscription Rate')
    return fig_age

def ageclassvssubbar(df,ax):
    st.subheader("Age Group vs Subscription Rate")
    # Create pivot table
    pivot_table = pd.pivot_table(df, index='age_group', columns='deposit', aggfunc='size')

    # Calculate percentages
    percentages = pivot_table.apply(lambda x: x/x.sum()*100, axis=1)
    age_balance1 = pd.DataFrame(df.groupby(['age_group','balance_status'])['binary_deposit'].sum())
    age_balance2 = pd.DataFrame(df.groupby(['age_group','balance_status'])['deposit'].count())
    age_balance1['deposit'] = age_balance2['deposit']
    age_balance1['deposit_rate'] = age_balance1['binary_deposit']/ (age_balance1['deposit'])
    age_balance1 = age_balance1.drop(['binary_deposit','deposit'],axis =1)
    age_bal = age_balance1.plot(kind='bar',figsize = (10,6))

    # Set x ticks
    plt.xticks(np.arange(6),('<30', '30-39', '40-49', '50-59', '60-69', '70+'),rotation = 'horizontal')

    # Set legend
    #plt.legend(['High Balance', 'Low Balance', 'Medium Balance', 'Negative Balance', 'Very High Balance'],loc = 'best',ncol = 1)

    plt.ylabel('Subscription Rate')
    plt.xlabel('Age Group')
    plt.title('The Subscription Rate of Different Balance Levels in Each Age Group')
    plt.show()
    return age_bal

def jobbydeposit(df):
    st.subheader("Job vs Deposit")
    #fig, ax = plt.subplots(figsize=[14,5])
    job_count = df.groupby(['job', 'deposit']).size().reset_index(name='count')
    job_count['percent'] = job_count.groupby(['job'])['count'].apply(lambda x: 100*x/np.sum(x))
    fig = sns.barplot(x='job', y='percent', hue='deposit', data=job_count, edgecolor="black")
    plt.title("Percentage plot of job by deposit")
    fig.set_ylabel('Percentage')
    for p in fig.patches:
        fig.annotate('{:.1f}%'.format(p.get_height()), ((p.get_x()+p.get_width()/2.)+0.04, p.get_height()+0.5), ha='center')
    return fig

def maritalbydeposit(df):
    st.subheader("Marital vs Deposit")
    #fig, ax = plt.subplots(figsize=[14,5])
    job_count = df.groupby(['marital', 'deposit']).size().reset_index(name='count')
    job_count['percent'] = job_count.groupby(['marital'])['count'].apply(lambda x: 100*x/np.sum(x))
    fig = sns.barplot(x='marital', y='percent', hue='deposit', data=job_count, edgecolor="black")
    plt.title("Percentage plot of Marital by deposit")
    fig.set_ylabel('Percentage')
    for p in fig.patches:
        fig.annotate('{:.1f}%'.format(p.get_height()), ((p.get_x()+p.get_width()/2.)+0.04, p.get_height()+0.5), ha='center')
    return fig

def educationbydeposit(df):
    st.subheader("Education vs Deposit")
    #fig, ax = plt.subplots(figsize=[14,5])
    job_count = df.groupby(['education', 'deposit']).size().reset_index(name='count')
    job_count['percent'] = job_count.groupby(['education'])['count'].apply(lambda x: 100*x/np.sum(x))
    fig = sns.barplot(x='education', y='percent', hue='deposit', data=job_count, edgecolor="black")
    plt.title("Percentage plot of Education by deposit")
    fig.set_ylabel('Percentage')
    for p in fig.patches:
        fig.annotate('{:.1f}%'.format(p.get_height()), ((p.get_x()+p.get_width()/2.)+0.04, p.get_height()+0.5), ha='center')
    return fig

def defaultersbydeposit(df):
    st.subheader("Loan Defaulters vs Deposit")
    #fig, ax = plt.subplots(figsize=[14,5])
    job_count = df.groupby(['default', 'deposit']).size().reset_index(name='count')
    job_count['percent'] = job_count.groupby(['default'])['count'].apply(lambda x: 100*x/np.sum(x))
    fig = sns.barplot(x='default', y='percent', hue='deposit', data=job_count, edgecolor="black")
    plt.title("Percentage plot of Loan Defaulters by deposit")
    fig.set_xlabel('Loan Defaulters')
    fig.set_ylabel('Percentage')
    for p in fig.patches:
        fig.annotate('{:.1f}%'.format(p.get_height()), ((p.get_x()+p.get_width()/2.)+0.04, p.get_height()+0.5), ha='center')
    return fig

def housingbydeposit(df,ax):
    st.subheader("Housing vs Deposit")
    #fig, ax = plt.subplots(figsize=[14,5])
    job_count = df.groupby(['housing', 'deposit']).size().reset_index(name='count')
    job_count['percent'] = job_count.groupby(['housing'])['count'].apply(lambda x: 100*x/np.sum(x))
    fig = sns.barplot(x='housing', y='percent', hue='deposit', data=job_count, edgecolor="black")
    plt.title("Percentage plot of Loan Defaulters by deposit")
    fig.set_xlabel('Housing')
    fig.set_ylabel('Percentage')
    for p in fig.patches:
        fig.annotate('{:.1f}%'.format(p.get_height()), ((p.get_x()+p.get_width()/2.)+0.04, p.get_height()+0.5), ha='center')
    return fig

def loanbydeposit(df,ax):
    st.subheader("Loan vs Deposit")
    #fig, ax = plt.subplots(figsize=[14,5])
    job_count = df.groupby(['loan', 'deposit']).size().reset_index(name='count')
    job_count['percent'] = job_count.groupby(['loan'])['count'].apply(lambda x: 100*x/np.sum(x))
    fig = sns.barplot(x='loan', y='percent', hue='deposit', data=job_count, edgecolor="black")
    plt.title("Percentage plot of Loan Defaulters by deposit")
    fig.set_ylabel('Percentage')
    for p in fig.patches:
        fig.annotate('{:.1f}%'.format(p.get_height()), ((p.get_x()+p.get_width()/2.)+0.04, p.get_height()+0.5), ha='center')
    return fig

def balance(df,ax):
    st.subheader("Balance vs Deposit")
    df["balance_status"].loc[(df['balance']>=df.balance.quantile(0.0)) & 
                         (df['balance']<=df.balance.quantile(0.15))] = 'negative'

    df["balance_status"].loc[(df['balance']>df.balance.quantile(0.15)) & 
                             (df['balance']<=df.balance.quantile(0.4))] = 'low'

    df["balance_status"].loc[(df['balance']>df.balance.quantile(0.4)) & 
                             (df['balance']<=df.balance.quantile(0.6))] = 'medium'

    df["balance_status"].loc[(df['balance']>df.balance.quantile(0.6)) & 
                             (df['balance']<=df.balance.quantile(0.8))] = 'high'

    df["balance_status"].loc[(df['balance']>df.balance.quantile(0.8)) & 
                             (df['balance']<=df.balance.quantile(1.0))] = 'very high'
    # Create pivot table of counts
    pivot_table = pd.pivot_table(df, index='balance_status', columns='deposit', aggfunc='size')

    # Calculate percentages
    pivot_table = pivot_table.apply(lambda x: x / x.sum(), axis=1)

    # Create heatmap
    fig_balance = sns.heatmap(pivot_table, annot=True, fmt='g')
    plt.xlabel('Deposit')
    plt.ylabel('Balance Status')
    return fig_balance


def generate_inferences(plot_1, plot_2, df):
    inferences = ""

    if plot_1 == "Age Class by Subscription":
        if plot_2 == "Age by Balance":
            inferences += "1. Majority of customers who subscribed to the term deposit plan have a balance less than 6840, indicating that balance is an important factor in predicting subscription.\n"
            inferences += "2. Only a small percentage (5%) of customers with a balance equal to or more than 6840 subscribed to the term deposit plan, suggesting that the bank should focus less on promoting the plan to customers with higher balances.\n"
            inferences += "3. The clusters 3rd(orange) and 4th(brown) have a higher subscription rate of almost 15% each as compared to the other clusters. Hence, the bank should focus on promoting the term deposit plan to clients belonging to these clusters to maximize their profits.\n"            
        elif plot_2 == "Duration by Campaign":
            inferences += "1. Customers belonging to the clusters 2nd and 5th have a higher percentage of subscription, 25% and 42%, respectively indicating that the bank should focus on targeting these groups in their marketing campaigns.\n"
            inferences += "2. It is important for the bank to limit the number of campaigns to 5 or less, as the percentage of customers subscribing to the plan decreases significantly beyond this point.\n"
            inferences += "3. Another important factor to consider is the duration of the calls. Customers who were engaged for a duration of less than 7 minutes had a lower subscription rate (7%) compared to those who were engaged for longer durations.\n"          
        elif plot_2 == "Month by Subscription Rate":
            inferences += "1. The month of March has the highest subscription rate of over 50%. The bank contacted the most clients in May and the highest contact rate was around 30%. However, the subscription rate in May was much lower.\n"
            inferences += "2. The bank's marketing campaign has the highest contact rate between May and August but the highest subscription rate occurs in March, September, October, and December. \n"
            inferences += "3. Therefore, the bank should consider initiating the telemarketing campaign in fall and spring when the subscription rate tends to be higher.\n"
        else:
            inferences += "1. The subscription rate for term deposits is highest among the oldest and youngest age groups, with 30% and 42% of subscriptions coming from clients aged 60 and 70+, and 18% coming from clients aged 18 to 29.\n"
            inferences += "2. This trend can be explained by the investment objectives of these age groups, with older clients focused on retirement savings and younger clients preferring lower-risk investments with higher returns than traditional savings accounts.\n"
            inferences += "3. The bank's marketing efforts have focused on middle-aged clients, who have returned lower subscription rates compared to younger and older clients. To increase the effectiveness of future marketing campaigns, the bank should target younger and older clients.\n"

    elif plot_1 == "Age Group by Subscription":
        inferences += "1. For all age groups, there is a positive correlation between balance levels and subscription rates.\n"
        inferences += "2. The bank should prioritize its telemarketing efforts on clients who are above 60 years old and have positive balances and young clients with positive balances to maximize their subscription rates.\n"
        inferences += "3. The marketing campaign should focus less on middle-aged clients, as they have a lower subscription rate compared to younger and older age groups.\n"
    elif plot_1 == "Job by Deposit":
        inferences += "1. Students and retired clients have the highest subscription rates among all job types.\n"
        inferences += "2. More than 50% of the subscriptions came from students and retired clients.\n"
        inferences += "3. Bank should focus its marketing efforts on students and retired clients to increase the subscription rate in the future.\n"
    elif plot_1 == "Marital By Deposit":
        inferences += "1. Clients who are single have a higher subscription rate of approximately 15%, while the other marital status categories have a subscription rate of approximately 10%.\n"
        inferences += "2. Therefore, the bank should focus its telemarketing efforts on single clients in order to increase the number of term deposits.\n"  
    elif plot_1 == "Education By Deposit":
        inferences += "1. Clients who have a professional course or university degree, have a higher subscription rate than those with primary or secondary education.\n"
        inferences += "2. Therefore, the bank should target clients with higher education levels, such as those with a professional course or university degree, in their marketing campaigns to increase their subscription rates.\n"
    elif plot_1 == "Loan Defaulters By Deposit":
        inferences += "1. Clients who have not defaulted on any loan are twice as likely to subscribe to the term deposit compared to those who have. \n"
        inferences += "2. The subscription rate for clients with no defaults is approximately 12%, while the subscription rate for those with defaults is only around 6%. \n"
        inferences += "3. Therefore, the bank should target clients with no loan defaults in their future telemarketing campaigns.\n"       
    elif plot_1 == "Housing & Loan By Deposit":
        inferences += "1. Clients who have not taken any housing loan are more likely to subscribe to the term deposit plan.\n"
        inferences += "2. The subscription rate of clients who have not taken any other loans (apart from housing loan) is slightly higher than those who have taken loans.\n"
        inferences += "3. The bank should prioritize contacting clients who have not taken any housing or other loans as they have a higher chance of subscribing to the term deposit plan.\n"        
    elif plot_1 == "Previous by Campaign":
        inferences += "1. The clusters 4th(pink) and 5th(brown) have a higher percentage of subscription, 28% and 21%, respectively indicating that the bank should focus on these groups while promoting their term deposit plan.\n"
        inferences += "2. It is important for the bank to perform more number of contacts i.e, more than 3 times to customers before the recent campaign, as the percentage of customers subscribing to the plan decreases significantly below this point.\n" 
        inferences += "3. Another important factor to consider is bank need to ensure to not make more than 5 calls in current campaign as the percentage of customers subscribing beyond this point reduces to almost 5%.\n"
    elif plot_1 == "Balance":  
        inferences += "1. Clients with high or very high balances have significantly higher subscription rates, nearly 14 and 17% respectively.\n"
        inferences += "2. Clients with negative balances, low and medium balances only returned a subscription rate of 7%, 10% and 12%.\n" 
        inferences += "3. Targeting clients with higher balances particularly when they have a balance of 1875 or more can increase the subscription rate.\n" 
    return inferences


def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; margin-top: 0px;'>Bank Marketing Campaign Analysis</h1>", unsafe_allow_html=True)


    # Load data
    df = load_data()

    # Create a dropdown menu to select the first visualization
    plot_1 = st.sidebar.selectbox("Univariate Visualisations", ["Age Class by Subscription","Age Group by Subscription","Job by Deposit","Marital By Deposit","Education By Deposit","Loan Defaulters By Deposit","Housing & Loan By Deposit","Previous by Campaign","Balance"])
    
    # Create a dropdown menu to select the second visualization
    plot_2 = st.sidebar.selectbox("Bivariate Visualisations", ["","Age by Balance", "Duration by Campaign","Month by Subscription Rate"])

    # Create a figure with subplots using gridspec
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(nrows=1, ncols=2, figure=fig, width_ratios=[6, 6], wspace=0.4)
    
    if plot_1 == "Age Class by Subscription":
        if plot_2 == "Age by Balance":
            # Add the scatter plot to the grid
            agebalance(df, fig.add_subplot(gs[0]))
            cluster_analysis(df, fig.add_subplot(gs[1]))
            inferences = generate_inferences(plot_1, plot_2, df) 
        elif plot_2 == "Duration by Campaign":
            # Add the KMeans clustering plot to the grid
            scatter_plot(df, fig.add_subplot(gs[0]))
            durcamp(df, fig.add_subplot(gs[1]))
            inferences = generate_inferences(plot_1, plot_2, df)  
        elif plot_2 == "Month by Subscription Rate":
            # Add the KMeans clustering plot to the grid
            monthcontact(df)
            inferences = generate_inferences(plot_1, plot_2, df)    
        else:
            # Add the KMeans clustering plot to the grid
            ageclassvssub(df, fig.add_subplot())
            inferences = generate_inferences(plot_1, plot_2, df)
    elif plot_1 == "Age Group by Subscription":
        # display image
        image = Image.open('download.png')
        st.image(image, caption='Age Group by Subscription')
        inferences = generate_inferences(plot_1, plot_2, df)
    elif plot_1 == "Job by Deposit":
        jobbydeposit(df)
        inferences = generate_inferences(plot_1, plot_2, df)
    elif plot_1 == "Marital By Deposit":
        maritalbydeposit(df)
        inferences = generate_inferences(plot_1, plot_2, df)   
    elif plot_1 == "Education By Deposit":
        educationbydeposit(df)
        inferences = generate_inferences(plot_1, plot_2, df)
    elif plot_1 == "Loan Defaulters By Deposit":
        defaultersbydeposit(df)
        inferences = generate_inferences(plot_1, plot_2, df)
    elif plot_1 == "Housing & Loan By Deposit":
        housingbydeposit(df, fig.add_subplot(gs[0]))
        loanbydeposit(df, fig.add_subplot(gs[1]))
        inferences = generate_inferences(plot_1, plot_2, df)
    elif plot_1 == "Previous by Campaign":
        previouscamp(df, fig.add_subplot())
        inferences = generate_inferences(plot_1, plot_2, df)
    elif plot_1 == "Balance":
        balance(df, fig.add_subplot())
        inferences = generate_inferences(plot_1, plot_2, df)
    
    # Set the boundaries of the grids
    gs.tight_layout(fig, rect=[0.05, 0.05, 0.95, 0.95])

    # Set the background color of the figure
    fig.patch.set_facecolor('#F5F5F5')

    # Add a dark outline around the grids
    for ax in fig.axes:
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')

    # Display the figure
    fig_container = st.container()
    with fig_container:
        st.pyplot(fig)
        
    # Display inferences for the selected plot
    st.header('Inferences')
    st.write(inferences)


    # Add CSS for the inferences container
    st.markdown("""
        <style>
            .inferences-container {
                background-color: #FFFFFF;
                border-radius: 5px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
                padding: 20px;
                margin-top: 40px;
                margin-bottom: 40px;
            }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()