# 📰 NewsAPI Integration Guide

## How to Enable Live News API

The Fake News Detection system now supports fetching real-time news from **NewsAPI.org**, which provides headlines from hundreds of news sources worldwide.

### Step 1: Get a Free API Key

1. Visit [NewsAPI.org](https://newsapi.org/)
2. Click **"Get API Key"** (top-right)
3. Sign up with your email address
4. Verify your email
5. You'll receive a **FREE API key** with 100 requests per day (perfect for development)

### Step 2: Set Up Environment Variable

#### Option A: Using .env File (Recommended)

1. Create a `.env` file in the project root:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and add your API key:
   ```
   NEWS_API_KEY=your_actual_api_key_here
   ```

3. Save and restart the Flask app

#### Option B: Using System Environment Variable

On Windows (PowerShell):
```powershell
$env:NEWS_API_KEY="your_api_key_here"
```

On Windows (Command Prompt):
```cmd
set NEWS_API_KEY=your_api_key_here
```

On Mac/Linux:
```bash
export NEWS_API_KEY="your_api_key_here"
```

### Step 3: Install python-dotenv

```bash
pip install python-dotenv
```

Or update all packages:
```bash
pip install -r requirements.txt
```

### Step 4: Restart Flask App

```bash
python app/app.py
```

The app will now:
- ✅ Fetch real headlines from 100+ news sources
- ✅ Cache results for 1 hour to avoid rate limits
- ✅ Fall back to sample articles if API fails
- ✅ Display region and category information

## Features

- **Real-time headlines** from major news outlets
- **Global coverage** with region identification
- **Category tagging** (Technology, Health, Business, etc.)
- **Automatic caching** for performance
- **Graceful fallback** to sample articles if API is unavailable
- **Refresh button** to shuffle between fetched and sample articles

## API Limits

- **Free Plan**: 100 requests/day (enough for testing)
- **Paid Plans**: Higher limits for production

## Troubleshooting

### "No news appearing?"
- Check your API key is correctly set in `.env`
- Verify the `.env` file is in the project root directory
- Ensure python-dotenv is installed: `pip install python-dotenv`
- Check Flask debug output for error messages

### "Only sample articles showing?"
- Your API key may not be set correctly
- Check that `.env` file exists and has `NEWS_API_KEY=...`
- Restart the Flask app after setting the API key
- Check network connectivity

### "API quota exceeded?"
- Free plan has 100 requests/day limit
- The app caches results for 1 hour to minimize requests
- Upgrade to a paid plan on NewsAPI.org for higher limits

## Alternative News APIs

If you prefer other APIs:

- **Guardian API** - https://open-platform.theguardian.com/
- **New York Times API** - https://developer.nytimes.com/
- **BBC News API** - Various third-party integrations
- **RSS Feeds** - Can be parsed with `feedparser` library

To switch APIs, modify the `fetch_latest_news()` function in `app/app.py`.

## Security Note

⚠️ **Never commit your `.env` file to Git!**

The `.gitignore` already excludes `.env`, so it won't be pushed to GitHub.
Only `.env.example` (without the key) is committed as a template.
