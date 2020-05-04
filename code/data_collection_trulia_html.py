# =============================================================================
# Version 2 of cwarling one page: 
# updates:
# price(), mortage, homeDetails
# =============================================================================

from bs4 import BeautifulSoup as BS
from urllib.request import urlopen
import csv
import pandas as pd
import json, re, time
import requests
import lxml.html as lh
import pandas as pd


# 2604 addresses
def getLinks(path):
    df = pd.read_csv(path)
    df = pd.DataFrame(df, columns= ['streetAddress', 'link'])
    address_link_lst = [df.columns.values.tolist()] + df.values.tolist()  # len 2605, included column header
    return address_link_lst


def getSoup(url):
    agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    page = requests.get(url, headers=agent)
    root = BS(page.content, "html.parser")
        
    content_json = root.find_all(type="application/ld+json")
    
    content_div = root.find_all('div', class_="MediaBlock__MediaContent-ldzu2c-1 bumWFt")
    
    content_mortage = root.find_all('div', class_="Text__TextBase-sc-1i9uasc-0-div Text__TextContainerBase-sc-1i9uasc-1 dkumKO")

    content_homeDetail = root.find_all('li', class_="FeatureList__FeatureListItem-iipbki-0 dArMue")

    content_priceTrends = root.find_all('div', class_="MediaBlock__MediaContent-ldzu2c-1 hWgsUV")
    
    content_price_and_mtg = root.find_all('div', class_="Text__TextBase-sc-1i9uasc-0-div Text__TextContainerBase-sc-1i9uasc-1 gtxlcQ")
    

#    content_price_and_mtg = root.find_all('div', class_="MediaBlock__MediaContent-ldzu2c-1 bumWFt")
#    print(content_price_and_mtg)
#    print(content_json)
    
    return page, content_json, content_div, content_mortage, content_homeDetail, content_priceTrends, content_price_and_mtg

def storeSoup(url, outputF):
    agent = {"User-Agent":'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    page = requests.get(url, headers=agent)
    root = BS(page.content, "html.parser")
    root = root.prettify()
        
    with open(outputF, "w", encoding='utf-8') as file:
        file.write(str(root))

#FIXME: content_div VS content_price_and_mtg
def price(content_div):
    for div in content_div: 
        res = re.sub('[^0-9]','', div.text)
        return res
            
def bedsBathsArea(content_div):
    res = {"nBeds": 0, "nBaths": 0, "area": 0}
    for div in content_div: 
#        print(div, "\n")
        s = [i for i in div.text.split()]
        if "Beds" in s:
            res["nBeds"] = s[0]
#            nb = s[0]
#            break
        if "sqft" in s:
            res["area"] = s[0]
#            ar = s[0]
#            continue
        if "Baths" in s or "Bath" in s:
            res["nBaths"] = s[0] 
    return res

def mortage(content_mortage):
    i = 0
    res = []
    for div in content_mortage:
        res.append(div.text)
        i += 1
        if i == 2:
            return res

    
def homeDetails(content_homeDetail):
    res = dict()
#    res1 = []
    for div in content_homeDetail:
        s = div.text
#        print(s)
        if s.find(":") != -1:
            elem = s.split(":")
            res[elem[0]] = elem[1][1:] # remove white space
        else:
            res[s] = ""
    res1 = [[i, v] for i, v in res.items()]
    for i in res1:
        for j in i:
            if j == "":
                i.remove(j)
    return res1 

def cleanHomeDetails(content_homeDetail):
# commented out on date 12/5/2019 2AM
#    res = homeDetails(content_homeDetail)
#    bldgtype = ["Townhouse", "Multi Family", "Single Family Home", "Condo"]
#    withComa = ["Lot Size", "Cooling System", "Heating", "Heating Fuel", "Stories"]
#    res1 = dict()
#    for i in range(len(res)):
#        if (len(res[i]) == 2) and (res[i][0] in withComa):
#            key = res[i][0].lower()
#            res1[key] = res[i][1]
#            
#        if (len(res[i]) == 1):
#            elem = res[i][0]
#            if "Built" in elem:
#                res1["built in year"] = elem.split()[-1]
#            if "Rooms" in elem:
#                res1["number of rooms"] = elem.split()[0]
#            if "Architecture" in elem:
#                res1["architecture type"] = elem.split()[0]
#            if elem in bldgtype:
#                res1["building type"] = elem
#    if "stories" in res1.keys():
#        tmp = res1["stories"].split()
#        if (len(tmp) > 1):
#            res1["stories"] = tmp[0]
#    return res1

    res = homeDetails(content_homeDetail)
    bldgtype = ["Townhouse", "Multi Family", "Single Family Home", "Condo"]
    withComa = ["Lot Size", "Cooling System", "Heating", "Heating Fuel", "Stories", "Parking", "Exterior", "Roof", "Parking Spaces"]
    res1 = dict()
    # print(res)
    for i in ["Dishwasher", "Microwave", "Washer", "Dryer", "Refrigerator"]:
        res1[i] = 0
    for i in range(len(res)):
        if (len(res[i]) == 2) and (res[i][0] in withComa):
            key = res[i][0].lower()
            res1[key] = res[i][1]
            
        if (len(res[i]) == 1):
            elem = res[i][0]
            # print(type(elem))
            if elem == "Dishwasher":
                res1["Dishwasher"] = 1
            if elem == "Microwave":
                res1["Microwave"] = 1
            if elem == "Washer":
                res1["Washer"] = 1
            if elem == "Dryer":
                res1["Dryer"] = 1
            if elem == "Refrigerator":
                res1["Refrigerator"] = 1
            if "Built" in elem:
                res1["built in year"] = elem.split()[-1]
            if "Rooms" in elem:
                res1["number of rooms"] = elem.split()[0]
            if "Architecture" in elem:
                res1["architecture type"] = elem.split()[0]
            if elem in bldgtype:
                res1["building type"] = elem
    if "stories" in res1.keys():
        tmp = res1["stories"].split()
        if (len(tmp) > 1):
            res1["stories"] = tmp[0]
    return res1

        
def getTables(page): 
    doc = lh.fromstring(page.content)
    # parse content between <tr>..</tr> of HTML
    tr_elements = doc.xpath('//tr')
    table_price_hist = dict()
    table_property_tax = dict()
    
    # price history table
    try: 
        for t in zip(tr_elements[0], tr_elements[1]):
            name = t[0].text_content()
            table_price_hist[name] = t[1].text_content()
            
        # property tax table
        for i in range(3, 9):
            for t in range(len(tr_elements[i])):
                k = tr_elements[i][t].text_content()
                if (t%2 == 0) and (t+1 < len(tr_elements[i])):
                    table_property_tax[k] = tr_elements[i][t+1].text_content()
    except:
        return {}, {}
    return table_price_hist, table_property_tax

#def getTables1(table):
#    res = dict()
#    for table_row in table.findAll('tr'):
#        text = table_row.text
#        
#        title = re.sub(r"\d", "", text)
#        title = re.sub("[^a-zA-Z]+", "", title)
#        value = ''.join(re.findall(r'\d+', text))
#        res[title] = value
#    print(res)
#    return res
#        print(result)
        
#        columns = table_row.findAll('td')
#        for c in columns:
#            print(c.text)
#            pass
#    print(out)
    

    
def priceTrends(content_priceTrends):
    count = 0
    lst = []
    for i in content_priceTrends:
        lst.append([i.text])
        count += 1
        if count == 4:
            break
    return lst[1:]
    
## return flatten json object to the deepest level
#def flatten_json(json_str):
#    res = dict()
#    
#    def flatten(x, name=''):
##        print(name)
#        if isinstance(x, dict):
#            for a in x.keys():
#                # use -> to append sub-key
#                flatten(x[a], name + a + '->')
#        elif isinstance(x, list):
#            for i in range(len(x)):
#                index = str(i)
#                flatten(x[i], name + index + '->')
#        else:
#            # otherwise, return original
#            res[name[:-2]] = x
#
#    flatten(json_str)
#    return res

#def resultAggre(infile, outfile, tryOut):
#    addr_links = getLinks(infile)
#    addr_links = addr_links[1:]
#    prc = price(content_json)
#    addr = []
#    links = []
#    addr = [i[0] for i in addr_links]
#    links = [i[1] for i in addr_links]
##    print(len(addr), len(links))
#    
#    data = {'address': addr, 'link': links, ''} 
#    df = pd.DataFrame(data, index =[i for i in range(len(addr_links))]) 
#
#    df.to_csv(tryOut, index=False)
    
# aggregate result for one address
def resultAggre(addr, url):
    page, content_json, content_div, content_mortage, content_homeDetail, content_priceTrends, content_price_and_mtg = getSoup(url)
   
    #FIXME: USE content_div VS content_price_and_mtg
    prc = price(content_price_and_mtg)
    bba = bedsBathsArea(content_div)
    mtg = mortage(content_mortage)
    hmdetail = cleanHomeDetails(content_homeDetail)
#    ph, pt = getTables(page)
#    pT = priceTrends(content_priceTrends)
    
#    t = getTables1(tables)
    
#    res = {"address": addr, "link": url, "price": prc, "beds": bba["nBeds"], "baths": bba["nBaths"], "area": bba["area"], "mortage": mtg, "homeDetail": hmdetail, "priceHistory": ph, "propertyTax": pt, "priceTrends": pT}
    res = {"address": addr, "link": url, "beds": bba["nBeds"], "baths": bba["nBaths"], "area": bba["area"], "sold information": mtg}
    
    for k, v in hmdetail.items():
        res.update({k: v})
    res.update({"price": prc})
    return res
    
def storeData(infile, outfile):
    addr_links = getLinks(infile)
    addr_links = addr_links[1:]
    addr = [i[0] for i in addr_links]
    links = [i[1] for i in addr_links]
    numRows = len(addr_links)
    
    df = pd.DataFrame()
    
    a1 = "228 Magnolia Ave"
    l1 = "https://www.trulia.com/p/pa/pittsburgh/228-magnolia-ave-pittsburgh-pa-15229--2014225767"
    a2 = "802 Forest Ridge Dr"
    l2 = "https://www.trulia.com/p/pa/pittsburgh/802-forest-ridge-dr-pittsburgh-pa-15221--2014187526"
    res1 = resultAggre(a1, l1)
    res2 = resultAggre(a2, l2)
    
    print("RES->", res1)
    print("RES->", res2)
    # crawl 1000 for train model (on date 12/3/2019)
    for i in range(6000, 6003):
        a = addr[i]
        l = links[i]
        a1 = "228 Magnolia Ave"
        l1 = "https://www.trulia.com/p/pa/pittsburgh/228-magnolia-ave-pittsburgh-pa-15229--2014225767"
        a2 = "802 Forest Ridge Dr"
        l2 = "https://www.trulia.com/p/pa/pittsburgh/802-forest-ridge-dr-pittsburgh-pa-15221--2014187526"
        res = resultAggre(a, l)
        print("RES->", res)
        time.sleep(0.1)
        print(i)
        tmp = pd.DataFrame([res])
        
        df = pd.concat([df, tmp], sort=False)
#    sequence = ["address", "link", "area", "Stories", "Number of rooms", "Lot Size", "beds", "baths", "Built in year", "Architecture type", "Building Type", "Cooling System", "Heating", "Heating Fuel",  "sold information", "price"]
#    df = df.reindex(columns=sequence)
    df.to_csv(outfile, index=False)
    
#    print(df)

    

    
   
    
    
    
    
# debug: https://www.trulia.com/p/pa/pittsburgh/527-idlewood-rd-pittsburgh-pa-15235--1150255986
    
# 
if __name__ == '__main__':
    inFile = "final.csv"
#    getLinks(inFile)
#    storeSoup(url1, "soup2.txt")
#    json_str = {"offers":{"@type":"Offer","price":4900000,"priceCurrency":"USD","url":"https://www.trulia.com/p/pa/pittsburgh/1145-beechwood-blvd-pittsburgh-pa-15206--2014061753"}}
#    json_str = {"@context":"http://schema.org","@type":"BreadcrumbList","itemListElement":[{"@type":"ListItem","position":1,"item":{"@id":"/sitemap/Pennsylvania-real-estate/","name":"PA"}},{"@type":"ListItem","position":2,"item":{"@id":"/PA/Pittsburgh/","name":"Pittsburgh"}},{"@type":"ListItem","position":3,"item":{"@id":"/PA/Pittsburgh/15206/","name":"15206"}}]}
#    outputF1 = "crawl_res_v1.txt"
    
#    sampleList = ["5565 Northumberland St", "https://www.trulia.com/p/pa/pittsburgh/5565-northumberland-st-pittsburgh-pa-15217--1119318112"]
#    addr1, url1 = sampleList
    addr1 = "435 Forest Highlands Dr"
    url1 = "https://www.trulia.com/p/pa/pittsburgh/435-forest-highlands-dr-pittsburgh-pa-15238--1012564856"
    page, content_json, content_div, content_mortage, content_homeDetail, content_priceTrends, content_price_and_mtg = getSoup(url1)
    
#    # result from crawling
#    prc = price(content_price_and_mtg)
    
#    print(prc)
#    
#    bba = bedsBathsArea(content_div)
#    mtg = mortage(content_mortage)
#    print(mtg)
#    cleanhmdetail = cleanHomeDetails(content_homeDetail)
#    print(cleanhmdetail)
#    ph, pt = getTables(page)
#    pT = priceTrends(content_priceTrends)
    

#
#    r = resultAggre(addr1, url1)
    storeData(inFile, "2_miss.csv")
#    print(r)
    
#    print(prc, bba, mtg, hmdetail, ph, pt, pT)
#    print(flatten_json(json_str))
    
#tree = ET.parse('https://www.trulia.com/p/pa/pittsburgh/808-parkway-ave-pittsburgh-pa-15235--2014253517.xml')
#root = tree.getroot()
#print(root)
    
        