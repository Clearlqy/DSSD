import io, time, json
import requests
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import random

# source code from course homework
# Args: url (string):
# Returns: status_code (integer), raw_html (string)
def retrieve_html(url):
   
    response = requests.get(url)
    status = response.status_code
    content = response.text
    
    return (status, content)

def read_api_key(filepath="api_key_clear.txt"):
    return Path(filepath).read_text().strip()


# Make an authenticated request to the Yelp API, modified from 
# Args:  query (tuple): (latitude, longitude, radius, category)
# returns: total (integer): total number of businesses on Yelp corresponding to the query 
#          businesses (list): list of dicts representing each business
def yelp_search(api_key, query):
        
    header =  {'Authorization': 'Bearer '+ api_key}
    url = "https://api.yelp.com/v3/businesses/search"
    searchKey = dict()
    
    
    lat, long, radius, cat = query
    
    searchKey["latitude"] = lat
    searchKey["longitude"] = long
    searchKey["radius"] = radius
    searchKey["categories"] = cat
    
    searchKey["sort_by"] = "distance"
    searchKey["location"] = "Pittsburgh"
    
    response = requests.get(url, params=searchKey, headers=header)
     
    
    business = []
    
    responseDict = json.loads(response.text)
    
    if "businesses" in responseDict:
        business = responseDict["businesses"]
        
    total = responseDict["total"]
    
    return total, business


# Get average rating and price level of business list
# Args: total: total (integer): total number of businesses
#       businesses (list): list of dicts representing each business
# Returns: average_rating(float), average_price(float)
def get_average(total, business):
    
    ratings = 0
    prices = 0
    rating_count = 0
    price_count = 0
    res_rating = 0
    res_price = 0
    
    for res in business:

        if "rating" in res:
            rating = res["rating"]
            ratings += rating
            rating_count += 1
        
        
        if "price" in res:
            price = res["price"]
            prices += len(price)
            price_count += 1
            
    if rating_count != 0:
        res_rating = ratings / rating_count
        
    if price_count != 0:
        res_price = prices / price_count
    
    if total == 0:
        return 0, 0
        
    return res_rating, res_price

# Fill in nearby restaurant information for each house in the database
def get_restaurant():
    for i in range(len(df)):
        lat, long = df.loc[i,["latitude", "longitude"]]
        query = (lat, long, 1000, "restaurants")
        total, business = yelp_search(api_key_clear, query)
        rating, price = getAverage(total, business)
        df.loc[i, "restaurant_count"] = total
        df.loc[i, "restaurant_rating"] = rating
        df.loc[i, "restaurant_price"] = price
        time.sleep(0.5)
        if i % 100 == 0:
            time.sleep(random.randint(1, 3))

# Fill in nearby art and entertainment count for each house in the database
def get_arts():
    for i in range(5339, len(df)):
        lat, long = df.loc[i,["latitude", "longitude"]]
        query = (lat, long, 1500, "arts")
        total, business = yelp_search(api_key_clear, query)
        if total != 0:
            count = 0
            for item in business:
                dis = item["distance"]
                if dis <= 1500:
                    count += 1
                else:
                    break

            df.loc[i, "arts_count"] = count

        # df.loc[i, "arts"] = total

        time.sleep(0.5)

        if i % 100 == 0:
            time.sleep(random.randint(1, 3))


# Fill in nearby groceries count for each house in the database
def get_grocery():
    for i in range(len(df)):
        lat, long = df.loc[i,["latitude", "longitude"]]
        query = (lat, long, 1500, "grocery")
        total, business = yelp_search(api_key_clear, query)
        if total != 0:
            count = 0
            for item in business:
                dis = item["distance"]
                if dis <= 1500:
                    count += 1
                else:
                    break

            df.loc[i, "grocery_count"] = count

        # df.loc[i, "grocery"] = total

        time.sleep(0.5)

        if i % 100 == 0:
            time.sleep(random.randint(1, 2))

# Generate link to corresponding walkscore webpage according to a house's trulia link
# Args: link(string): trulia link of the house
# Return: url(string): url of corresponding walkscore page for the house
def get_url(link):
    walkscore_path = "https://www.walkscore.com/score/"
    temp = "-".join(link.split("/")[-1].split("-")[:-2])
    url = walkscore_path + temp
    return url

# Scrape walk, bike and transit score from webpage
# Args: url(string): url of walkscore page for a house
# Returns: score(dict): dictonary stores walk, bike and transit score for the house
def get_score(url):
    agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    page = requests.get(url, headers=agent)
    root = BeautifulSoup(page.content, "html.parser")
    content = root.find_all("img")

    score = dict()
    
    for img in content:
        src = img.get("src")
        alt = img.get("alt")
        
        if src != None and src.startswith("//pp.walk.sc/badge/"):
            name = src.split("/")[-3].lower()
            s = src.split("/")[-1].split(".")[0].lower()
            score[name] = s
        if len(score) == 3:
            break
    return score


# Fill all three scores for each house
def get_allscores():
    for i in range(len(df)):
        link = df["link"][index]
        url = get_url(link)

        walk = np.nan
        transit = np.nan
        bike = np.nan

        score = get_score(url)

        if "walk" in score:
            walk = score["walk"]

        if "transit" in score:        
            transit = score["transit"]

        if "bike" in score:
            bike = score["bike"]

        # df.loc[i, "scoreLink"] = url
        df.loc[i, "walk"] = walk
        df.loc[i, "transit"] = transit
        df.loc[i, "bike"] = bike
    
        time.sleep(random.randint(1, 3))
    
        if i % 100 == 0:
            time.sleep(random.randint(1, 5))


def main():
    global df
    path = "scrape_data/scrape_trulia_list.csv"
    df = pd.read_csv(path)
    names = ["street_address", "postal_code", "latitude", "longitude", "link"]
    df = df[names]
    # df.head

    # Initialize colunms for nearby restaturants information
    radius = 1000
    df["restaurant_count"] = np.nan
    df["restaurant_rating"] = np.nan
    df["restaurant_price"] = np.nan

    get_restaurant()

    # Inititalize columns for arts and grocery
    # df["arts"] = np.nan
    df["arts_count"] = np.nan
    # df["grocery"] = np.nan
    df["grocery_count"] = np.nan

    get_arts()
    get_grocery()

    # Initialize columns for walkscore
    df["walk"] = np.nan
    df["transit"] = np.nan
    df["bike"] = np.nan
    # df["scoreLink"] = np.nan

    get_allscores()

    df.to_csv("scrape_data/yelp_ws_information.csv")


if __name__ == '__main__':
    main()
