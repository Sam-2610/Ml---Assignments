# Web Scraping with Python

A beginner's guide to web scraping in Python using the `requests` library and `BeautifulSoup`. This program covers the core concepts of fetching web pages, saving data, sending headers, consuming APIs, and parsing HTML.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `requests` | Send HTTP requests and fetch web page content |
| `beautifulsoup4` | Parse and extract data from HTML |
| `pandas` | Normalize and work with JSON/API data |

### Install Dependencies

```bash
pip install requests beautifulsoup4 pandas
```

---

## Concepts Covered

### 1. Basic GET Request

Sends an HTTP GET request to a URL and prints the response.

```python
import requests
url = "https://example.com"
res = requests.get(url)

print(res.status_code)   # 200 means success
print(res.text)          # HTML content as a string
print(res.content)       # raw bytes content
print(res.headers)       # response headers (content-type, server, etc.)
```

| Property | Description |
|----------|-------------|
| `status_code` | HTTP status — `200` OK, `404` Not Found, `403` Forbidden |
| `text` | Page content as a readable string |
| `content` | Raw bytes — useful for images or binary files |
| `headers` | Metadata about the response |

---

### 2. Downloading and Saving to a File

Fetches a web page and saves its HTML content to a local file for later use.

```python
response = requests.get(url)

with open("scraped_data", "w") as f:
    f.write(response.text)
```

- `open("scraped_data", "w")` — opens a file in write mode (creates it if it doesn't exist)
- `f.write(response.text)` — writes the HTML content into the file
- This saved file is later read by BeautifulSoup for parsing

---

### 3. Sending Headers

Some websites block requests that don't look like they're coming from a real browser. Sending a `User-Agent` header mimics a browser request.

```python
header = {
    "user-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=header)
print(response.text)
```

- `User-Agent` tells the server what browser/client is making the request
- `Mozilla/5.0` is a common browser identifier that most websites accept
- Without this, some sites return a `403 Forbidden` or block the request

---

### 4. Consuming APIs

Fetches data from a JSON API and loads it into a pandas DataFrame for structured analysis.

```python
res = requests.get(url)
json_data = res.json()           # parse response as JSON

import pandas as pd
df = pd.json_normalize(json_data["data"])
df = df[['id', 'Title', 'Year', 'Json']]   # select specific columns
```

- `res.json()` — converts the JSON response into a Python dictionary
- `pd.json_normalize()` — flattens nested JSON into a flat table (DataFrame)
- Selecting specific columns keeps only the data you need

> ⚠️ **Note:** The URL used in this section points to a GitHub README page, not a real JSON API. For this code to work correctly, replace the URL with an actual API endpoint that returns JSON (e.g. a public REST API).

---

### 5. Parsing HTML with BeautifulSoup

Reads the saved HTML file and parses it using BeautifulSoup.

```python
from bs4 import BeautifulSoup

with open("scraped_data", "r") as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, "html.parser")
print(soup)
```

- `html.parser` — Python's built-in HTML parser (no extra install needed)
- `soup` is now a navigable tree of the entire HTML page
- You can search, filter, and extract any element from it

---

### 6. BeautifulSoup Methods

Demonstrates key methods for navigating and extracting data from parsed HTML.

```python
# Find all <h3> tags
all_h3 = soup.find_all("h3")

for h3 in all_h3:
    name = h3.get_text(strip=True)           # extract clean text
    print(h3.find_parent("div"))             # get the parent <div>
    population = h3.find_next("div").select("span.country-population")[0].get_text(strip=True)

# Using select_one instead of select + index
print(h3.find_next("div").select_one("span.country-population").get_text(strip=True))
```

| Method | Description |
|--------|-------------|
| `find_all("tag")` | Returns a list of all matching tags |
| `get_text(strip=True)` | Extracts inner text, removes extra whitespace |
| `find_parent("tag")` | Finds the nearest parent with that tag |
| `find_next("tag")` | Finds the next sibling/element with that tag |
| `select("css.selector")` | CSS selector — returns a list of matches |
| `select_one("css.selector")` | CSS selector — returns only the first match |

---

## Program Flow

```
1. Send GET request to URL
        ↓
2. Print status, text, content, headers
        ↓
3. Save HTML to local file ("scraped_data")
        ↓
4. Re-fetch with User-Agent header
        ↓
5. Fetch JSON from API → load into DataFrame
        ↓
6. Read saved file → parse with BeautifulSoup
        ↓
7. Extract data using BS4 methods
```

---

## How to Run

```bash
python Web_Scraping.py
```

> Make sure all dependencies are installed first:
> ```bash
> pip install requests beautifulsoup4 pandas
> ```

---

## Things to Keep in Mind

- Always check a website's **robots.txt** (e.g. `https://example.com/robots.txt`) before scraping — it tells you which pages are allowed to be scraped.
- Add **delays** between requests (`time.sleep()`) to avoid overwhelming a server.
- The **API section** requires a real JSON API URL to work correctly.
- Some websites require **authentication** or **cookies** — the `requests.Session()` object handles those cases.

---

## Author

**Satyam Sagar**
📧 satyamsagar827@gmail.com
