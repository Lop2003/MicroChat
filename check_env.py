import os
required = ['GOOGLE_API_KEY','COHERE_API_KEY','LINE_CHANNEL_ACCESS_TOKEN','LINE_CHANNEL_SECRET','NGROK_AUTHTOKEN']
print('Checking environment variables:')
for k in required:
    print(f"{k}: {'SET' if os.getenv(k) else 'MISSING'}")
