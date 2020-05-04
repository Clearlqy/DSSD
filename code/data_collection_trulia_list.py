import os
import requests
from bs4 import BeautifulSoup as BS
import time
import sys
import numpy as np
import pandas as pd
import requests
import csv
import random
import json

def get_one(url, i):
    
    # use an agent to prevent getting blocked by the website
    agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    page = requests.get(url, headers=agent)
    root = BS(page.content, "html.parser")
    content = root.find_all(type = "application/ld+json")
    href = root.find_all("a")
    links = []

    #extract address information from the link
    for link in href:
        u = link.get("href")
        if (u != None) and (u.startswith("/p")) and (u != "/post-rental/"):
            links.append(u)
    res = []
    for a in content:
        c = json.loads(a.text)
        temp = {}

        #filter out th information that we did not successfully filter out before
        #extract useful information
        if "address" not in c:
            continue
        if "address" in c.keys():
            if ("streetAddress" in c["address"].keys()): 
                temp["street_address"] = c["address"]["streetAddress"]
        if "address" in c.keys():
            if ("postalCode" in c["address"].keys()): 
                temp["postal_code"]= c["address"]["postalCode"]
        
        if "address" in c.keys():
            if ("addressLocality" in c["address"].keys()): 
                temp["address_locality"] = c["address"]["addressLocality"]

        if "address" in c.keys():
            if ("addressRegion" in c["address"].keys()): 
                temp["address_region"] = c["address"]["addressRegion"]

        if "geo" in c.keys():
            if ("latitude" in c["geo"].keys()): 
                temp["latitude"] = c["geo"]["latitude"]
        if "geo" in c.keys():
            if ("longitude" in c["geo"].keys()): 
                temp["longitude"] = c["geo"]["longitude"]

        # to prevent duplicate results because the results of the website might change
        if temp not in res:
            res.append(temp)
            
    for j in range(len(res)):
        r = res[j]

        # sometimes the link starts with property and the format is different, this indicates
        # if the link includes "property" and we will manually format the information 

        if links[j].startswith("property"):
            r["property"] = 1
        else:
            r["property"] = 0
        r["link"] = "https://www.trulia.com"+ links[j]

        # to compare if the two addresses match
        ccc = " ".join(links[j].split('/')[-1].split('-')[:-1])
        r["link_generated"] = ccc.strip()
        r["page"] = i
        r["num"] = j
        st = r["street_address"] + " " + r["address_locality"] + " " + r["address_region"] + " " + r["postal_code"]
        st = st.replace("#",'')
        r["scrape_generated"] = st.lower()
        if(r["scrape_generated"] == r["link_generated"]):
            r["is_same"] = 1
        else:
            r["is_same"] = 0
            
    headers = ['street_address',
               'postal_code',
               'latitude',
               'longitude',
               'link',
               'page',
               'num',
               "address_locality",
               "address_region",
               "scrape_generated",
               "link_generated",
               "is_same",
               "property"]

    csv_file = "scrape_trulia_list.csv"
    with open(csv_file, 'a') as f:
        writer = csv.DictWriter(f, delimiter=',', lineterminator='\n', fieldnames=headers)
        writer.writeheader()
        writer.writerows(res)

def get_all():
    for i in range(1, 219):
        url = "https://www.trulia.com/sold/Pittsburgh,PA/APARTMENT,CONDO,COOP,MULTI-FAMILY,SINGLE-FAMILY_HOME,TOWNHOUSE_type/%d_p/" %i
        get_one(url, i)
        # stop scraping for random seconds to prevent getting blocked by the website
        t = random.randint(3, 7)
        time.sleep(t)

def main():
    if __name__ == '__main__':
        get_all()

main()   