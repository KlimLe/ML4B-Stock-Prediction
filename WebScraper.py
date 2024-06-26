import requests
import json
from bs4 import BeautifulSoup
import gdelt
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures

# Initialize GDELT version 2
gd2 = gdelt.gdelt(version=2)

# Expanded list of American financial publishers
financial_publishers = [
    'reuters.com', 'wsj.com', 'ft.com', 'cnbc.com',
    'businessinsider.com', 'marketwatch.com', 'economist.com', 'forbes.com',
    'investopedia.com', 'thestreet.com', 'fortune.com',
    'barrons.com', 'money.cnn.com', 'finance.yahoo.com', 'fool.com', 'nasdaq.com'
]

# Finance-related keywords
finance_keywords = [
    'finance', 'stock', 'market', 'economy', 'investment', 'bank', 'trading',
    'shares', 'bonds', 'forex', 'cryptocurrency', 'money', 'fiscal', 'revenue',
    'profits', 'earnings', 'fund'
]

# Function to download and scrape article headline
def scrape_headline(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            headline = soup.find('h1')
            headline_text = headline.get_text() if headline else "No headline found"
            return headline_text
        else:
            return "Failed to retrieve the page"
    except Exception as e:
        return f"An error occurred: {e}"

# Function to fetch metadata and scrape headlines
def fetch_and_scrape(date):
    try:
        results = gd2.Search(date, table='mentions', output='json')
        results_json = json.loads(results)
        data = []
        for entry in results_json:
            source = entry.get("MentionSourceName")
            if any(publisher in source for publisher in financial_publishers) or \
               any(keyword in source.lower() for keyword in finance_keywords):
                url = entry.get("MentionIdentifier")
                if url:
                    headline = scrape_headline(url)
                    date = entry.get("MentionTimeDate")
                    data.append((datetime.strptime(str(date), '%Y%m%d%H%M%S'), headline, source))
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Define the date range for 2024
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 5, 30)

# List to hold all the data
all_data = []

# Use ThreadPoolExecutor to fetch and scrape data concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = []
    current_date = start_date
    while current_date <= end_date:
        futures.append(executor.submit(fetch_and_scrape, current_date.strftime('%Y %b %d')))
        current_date += timedelta(days=1)

    for future in concurrent.futures.as_completed(futures):
        all_data.extend(future.result())

# Convert the data to a DataFrame
df = pd.DataFrame(all_data, columns=['Date', 'Headline', 'Source'])

# Save the DataFrame to a CSV file
output_filename = 'gdelt_financial_headlines_2024.csv'
df.to_csv(output_filename, index=False)

print(f"Headlines saved to {output_filename}")
