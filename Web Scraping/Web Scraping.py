# Basic GET request

import requests
url = "https://github.com/practical-tutorials/project-based-learning?tab=readme-ov-file#html-and-css"
res = requests.get(url)

print(res.status_code)
print(res.text)
print(res.content)
print(res.headers)

# Downloading and Storing to Files

url = "https://github.com/practical-tutorials/project-based-learning?tab=readme-ov-file#html-and-css"
response = requests.get(url)

with open("scraped_data","w") as f:
    f.write(response.text)

# Headers

header = {
    "user-Agent":"Mozilla/5.0"
}

response = requests.get("https://github.com/practical-tutorials/project-based-learning?tab=readme-ov-file#html-and-css",headers=header)
print(response.text)

# APIs

url = "https://github.com/practical-tutorials/project-based-learning?tab=readme-ov-file#html-and-css"
res = requests.get(url)
json_data = res.json()

import pandas as pd
df = pd.json_normalize(json_data["data"])
df = df[['id', 'Title', 'Year', 'Json']]

# Code for Beautiful Soup

from bs4 import BeautifulSoup
with open("scraped_data","r") as f:
    html_content = f.read()

soup = BeautifulSoup(html_content,"html.parser")
print(soup)

# Methods

all_h3 = soup.find_all("h3")
for h3 in all_h3:
    name = h3.get_text(strip=True)
    print(h3.find_parent("div"))
    population = h3.find_next("div").select("span.country-population")[0].get_text(strip=True)

print(h3.find_next("div").select_one("span.country-population").get_text(strip=True))