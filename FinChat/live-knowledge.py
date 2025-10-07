from sentence_transformers import SentenceTransformer
import finnhub
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pinecone
from pinecone import ServerlessSpec

load_dotenv()
finnhub_api=os.getenv('FINHUB_API')

finnhub_client=finnhub.Client(api_key=finnhub_api)

today=datetime.utcnow().date()
three_days_ago=today-timedelta(days=3)

PINECONE_API=os.getenv("PINECONE_API")
pc=pinecone.Pinecone(api_key=PINECONE_API)
index_name="news-embeddings"
if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
index=pc.Index(index_name)
model=model=SentenceTransformer("all-MiniLM-L6-v2")


data_nvda=finnhub_client.company_news('NVDA', _from=three_days_ago, to=today)

tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "JPM", "XOM", "KO"]

for ticker in tickers:
    print(f"Fetching news for {ticker}...")


    data = finnhub_client.company_news(ticker, _from=three_days_ago, to=today)

    for i, item in enumerate(data):
        if "headline" in item and "summary" in item and "related" in item:
            text = f"Headline: {item['headline']}\nSummary: {item['summary']}\nRelated: {item['related']}"
            embedding = model.encode(text)

            # Metadata
            metadata = {
                "ticker": item["related"],
                "headline": item["headline"],
                "summary": item["summary"],
                "source": item.get("source", ""),
                "datetime": item.get("datetime", "")
            }

            vector_id = f"{ticker}_{i}_{int(item.get('datetime', 0))}"

            index.upsert([(vector_id, embedding.tolist(), metadata)])

    print(f"Stored {len(data)} articles for {ticker}")

print("✅ Done storing news embeddings for all tickers!")

