import finnhub
import os
from dotenv import load_dotenv
import json

load_dotenv()
finnhub_api=os.getenv('FINHUB_API')

finnhub_client=finnhub.Client(api_key=finnhub_api)

data=finnhub_client.company_news('NVDA', _from="2025-09-03", to="2025-09-05")
pretty_json = json.dumps(data, indent=4, sort_keys=True)
print(pretty_json)