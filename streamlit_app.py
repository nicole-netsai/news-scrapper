import streamlit as st
import pandas as pd
!pip install scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
from datetime import datetime
import scrapy
from scrapy.crawler import CrawlerProcess
import csv

# NewsSpider class from the provided code (modified to return data)
class NewsSpider(scrapy.Spider):
    name = 'news_spider'
    
    custom_settings = {
        'CONCURRENT_REQUESTS': 1,
        'DOWNLOAD_DELAY': 2,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'news_articles.csv'
    }

    def start_requests(self):
        newspapers = {
            'BBC': {
                'Business': 'https://www.bbc.com/news/business',
                'Politics': 'https://www.bbc.com/news/politics',
                'Arts': 'https://www.bbc.com/news/entertainment_and_arts',
                'Sports': 'https://www.bbc.com/sport'
            },
            'CNN': {
                'Business': 'https://www.cnn.com/business',
                'Politics': 'https://www.cnn.com/politics',
                'Arts': 'https://www.cnn.com/entertainment',
                'Sports': 'https://www.cnn.com/sport'
            },
            'Reuters': {
                'Business': 'https://www.reuters.com/business/',
                'Politics': 'https://www.reuters.com/politics/',
                'Arts': 'https://www.reuters.com/lifestyle/',
                'Sports': 'https://www.reuters.com/sports/'
            },
            'Guardian': {
                'Business': 'https://www.theguardian.com/uk/business',
                'Politics': 'https://www.theguardian.com/politics',
                'Arts': 'https://www.theguardian.com/uk/culture',
                'Sports': 'https://www.theguardian.com/uk/sport'
            }
        }

        for paper, categories in newspapers.items():
            for category, url in categories.items():
                yield scrapy.Request(url=url, callback=self.parse,
                                   meta={'paper': paper, 'category': category})

    def parse(self, response):
        paper = response.meta['paper']
        category = response.meta['category']

        if paper == 'BBC':
            articles = response.css('div.gs-c-promo')
            for article in articles:
                yield {
                    'title': article.css('h3.gs-c-promo-heading__title::text').get(),
                    'url': response.urljoin(article.css('a::attr(href)').get()),
                    'summary': article.css('p.gs-c-promo-summary::text').get(),
                    'paper': paper,
                    'category': category,
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

        elif paper == 'CNN':
            articles = response.css('div.container__item')
            for article in articles:
                yield {
                    'title': article.css('span.container__headline-text::text').get(),
                    'url': response.urljoin(article.css('a::attr(href)').get()),
                    'summary': article.css('div.container__description::text').get(),
                    'paper': paper,
                    'category': category,
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

        elif paper == 'Reuters':
            articles = response.css('article.story')
            for article in articles:
                yield {
                    'title': article.css('h3.story-title a::text').get(),
                    'url': response.urljoin(article.css('h3.story-title a::attr(href)').get()),
                    'summary': article.css('p.story-excerpt::text').get(),
                    'paper': paper,
                    'category': category,
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

        elif paper == 'Guardian':
            articles = response.css('div.fc-item')
            for article in articles:
                yield {
                    'title': article.css('a.js-headline-text::text').get(),
                    'url': response.urljoin(article.css('a::attr(href)').get()),
                    'summary': article.css('div.fc-item__standfirst::text').get(),
                    'paper': paper,
                    'category': category,
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

def run_spider():
    process = CrawlerProcess()
    process.crawl(NewsSpider)
    process.start()

def load_data():
    try:
        df = pd.read_csv('news_articles.csv')
        return df
    except FileNotFoundError:
        st.warning("No data found. Scraping news articles...")
        run_spider()
        return load_data()

def preprocess_data(df):
    # Clean data
    df = df.dropna(subset=['title'])
    df = df.drop_duplicates(subset=['title'])
    
    # Fill missing summaries with empty string
    df['summary'] = df['summary'].fillna('')
    
    # Combine title and summary for clustering
    df['text'] = df['title'] + ' ' + df['summary']
    
    return df

def cluster_articles(df, n_clusters=5):
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['text'])
    
    # Cluster articles
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X.toarray())
    
    df['x'] = reduced_features[:, 0]
    df['y'] = reduced_features[:, 1]
    
    return df, vectorizer, kmeans

def plot_clusters(df):
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        color='cluster',
        hover_data=['title', 'paper', 'category'],
        title='News Article Clusters'
    )
    st.plotly_chart(fig)

def display_cluster_details(df, selected_cluster):
    cluster_df = df[df['cluster'] == selected_cluster]
    
    st.subheader(f"Cluster {selected_cluster} - {len(cluster_df)} Articles")
    
    # Display cluster summary
    st.write("**Most common words:**")
    
    # Show articles in the cluster
    st.write("**Articles in this cluster:**")
    for idx, row in cluster_df.iterrows():
        st.write(f"- [{row['title']}]({row['url']}) ({row['paper']}, {row['category']})")

def main():
    st.title("News Article Clustering Dashboard")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Sidebar controls
    st.sidebar.header("Settings")
    n_clusters = st.sidebar.slider("Number of clusters", 3, 10, 5)
    
    # Cluster articles
    df, vectorizer, kmeans = cluster_articles(df, n_clusters)
    
    # Display clusters
    st.header("Article Clusters Visualization")
    plot_clusters(df)
    
    # Cluster selection
    st.header("Explore Clusters")
    selected_cluster = st.selectbox("Select a cluster to explore", sorted(df['cluster'].unique()))
    display_cluster_details(df, selected_cluster)
    
    # Show raw data
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.write(df)

if __name__ == "__main__":
    main()
