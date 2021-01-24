import requests as r
import pandas
from urllib.parse import urlencode

from tqdm.notebook import tqdm
import multiprocessing as mp
from functools import partial

import numpy as np

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from bs4.element import Comment

import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import spacy
nlp = spacy.load('en_core_web_lg')
from collections import Counter

stop_words = stopwords.words('english')
stop_words += ['register', 'checkout', 'login', 'cart', 'shopping', 'subtotal', 'view', 
               'order', 'history', 'search', 'wishlist', 'buy', 'credit', 'product',
               'policy', 'newsletter', 'sign', 'menu', 'sale', 'link', 'image', 'store',
               'online']
stop_words = set(stop_words)


from key import API_KEY, api_key

PLACES_API_BASE = f'https://maps.googleapis.com/maps/api/place/findplacefromtext/json?key={API_KEY}&inputtype=textquery&'
DETAILS_API_BASE = f'https://maps.googleapis.com/maps/api/place/details/json?key={API_KEY}&'
TEXTSEARCH_API_BASE = f'https://maps.googleapis.com/maps/api/place/textsearch/json?key={API_KEY}&'
LOC_SEARCH_API_BASE = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?key={API_KEY}&'

ALLOWED_PLACE_TYPES = ['clothing_store', 'drugstore', 'department_store', 
                       'book_store', 'bar', 'liquor_store', 'electronics_store', 
                       'jewelry_store']

def get_latlong(code):
    res = r.get(f'http://api.postcodes.io/postcodes/{code}')
    # TODO what if res not good
    loc = res.json()

    long = loc['result']['longitude']
    lat = loc['result']['latitude']

    return lat, long

def get_placeid(code):
    lat, long = get_latlong(code)
    url = 'https://maps.googleapis.com/maps/api/geocode/json?latlng=' + str(lat) + ',' + str(long) + '&key=' + api_key
    res = r.get(url)

    return res.json()['results'][0]['place_id']

def get_details(place_id):
    params = {
      'place_id': place_id,
    }
    url = DETAILS_API_BASE + urlencode(params)
    res = r.get(url)
    response = res.json()

    return response['result']


PLACE_ATTRS =  ['name', 'types', 'website', 'formatted_address', 'rating', 'place_id']
def filter_(places):
    places_w_details = [get_details(p['place_id']) for p in places]
    places_w_website = [p for p in places_w_details if 'website' in p]
    return [{k: d[k] for k in PLACE_ATTRS if k in d} for d in places_w_website] 


def get_places_by_type(postcode=None, pagetoken=None, radius=1000, place_t='cafe'):
    assert postcode or pagetoken
    places = []

    if postcode:
        lat, long = get_latlong(postcode)
        params = {
            'location': f'{lat},{long}',
            'radius': radius,
            'type': place_t, 
        }
    else:
        params = {
            'pagetoken': pagetoken, 
        }

    url = LOC_SEARCH_API_BASE + urlencode(params)

    res = r.get(url)
    response = res.json()

    results = filter_(response['results'])
    places.extend(results)

    if 'next_page_token' in response:
        places.extend(get_places_by_type(pagetoken=response['next_page_token'], place_t=place_t))

    return places


def get_places(postcode):
    places = []
    for t in tqdm(ALLOWED_PLACE_TYPES):
        print
        places.extend(get_places_by_type(postcode=postcode, place_t=t))
    return places


flatten = lambda t: [item for sublist in t for item in sublist]


class PlacesSearchFn:
    def __init__(self, postcode):
        self.postcode = postcode
        
    def __call__(self, t):
        return get_places_by_type(postcode=self.postcode, place_t=t)


def get_places_mp(postcode):
    with mp.Pool(processes=len(ALLOWED_PLACE_TYPES)) as pool:
        results = pool.map(PlacesSearchFn(postcode), ALLOWED_PLACE_TYPES)
    return flatten(results)


def get_dist(source, top_places):
    """
    gets distance and duration between source and destination (driving mode)
    source: source place_id (not sure why using coordinates from postcodes gives no result)
    top_places: output from Sultan's code for now
    """
    num_dest = len(top_places)
    dest  = [top_places[i]['place_id'] for i in range(num_dest)]

    # if source is a postcode (but when converted to coordinates gives "no result") pretty sure i'm doing sth wrong somewhr
    # long, lat = get_latlong(source)
    #url = dist_url + 'origins=' + str(long) + ',' + str(lat) + '&destinations=place_id:' + dest[0]


    dist_url ='https://maps.googleapis.com/maps/api/distancematrix/json?'
    url = dist_url + '&origins=place_id:' + source + '&destinations=place_id:' + dest[0]

    if num_dest > 1:
        for i in range(1, num_dest):
            url += '|place_id:' + dest[i]

    url += '&key=' + api_key
    #print(url)

    res = r.get(url) 				
    res_json = res.json() 

    # by default driving mode considered 
    return res_json


def get_shops_sorted(dist_json, top_places, mode):
    """
    sorts destination by distance/ duration from source
    dist_json: output from get_dist
    top_places: output from Sultan's code
    mode: order shops by "distance" or "duration"
    """
    sorted_shops = []
    shops_dict = {}

    if mode == 'distance':
        dist_to_dest = [dist_json['rows'][0]['elements'][i]['distance']['value'] for i in range(len(top_places))]
        sorted_dist_by_index = np.argsort(dist_to_dest)
        for i in sorted_dist_by_index:
            #sorted_shops.append((top_places[i]['name'], dist_to_dest[i]))
            shops_dict[top_places[i]['name']] = dist_to_dest[i]
    
    if mode == 'duration':
        time_to_dest = [dist_json['rows'][0]['elements'][i]['duration']['value'] for i in range(len(top_places))]
        sorted_time_by_index = np.argsort(time_to_dest)
        for i in sorted_time_by_index:
            #sorted_shops.append((top_places[i]['name'], time_to_dest[i]))
            shops_dict[top_places[i]['name']] = time_to_dest[i]
    
    return sorted_shops, shops_dict         # returning sorted list and dict as I'm not sure what we need for now


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    # this also get the text in the "tabs" that pop up when you hover your mouse over a word
    # so might be a bit weird
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts) 
    return u" ".join(t.strip() for t in visible_texts)

def get_word_count(search_word, url):
    """
    Count the freq of a word in a webpage
    search_word: word that we are searching for
    url: webpage url
    """
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        text = text_from_html(html)
        split_text = text.split()
        split_text = [i.lower() for i in split_text]

        lemmatizer = WordNetLemmatizer() 
        text_filtered = [w for w in split_text if w.isalpha()]
        text_filtered = [w for w in text_filtered if not w in stop_words]
        text_filtered = [lemmatizer.lemmatize(w) for w in text_filtered]
        #print(sorted(text_filtered))


        # counter occurences of word
        counter = Counter(text_filtered)

        counter_res = []                        # result for indiv words
        counts = 0                              # total count if search word is more than one
        search_word = search_word.split()       # for the case where there is more than 1 search word
        search_word = [lemmatizer.lemmatize(i) for i in search_word]
        for word in search_word:
            counter_res.append((counter[word], word))
            counts += counter[word]
        
        return counter_res, counts
    
    except:
        
        return [('error', search_word)], 0


def order_by_word_freq(search_word, top_places):
    """
    orders the shop by frequency in which the search word appears in the website
    search_word: keyword that user enters
    top_places: output from Sultan's code
    """
    num_dest = len(top_places)
    web_urls  = [top_places[i]['website'] for i in range(num_dest)]

    word_freqs = []
    # get freq of search_word in each web
    for url in web_urls:
        counter_res, counts = get_word_count(search_word, url)
        word_freqs.append(counts)
    
    # sort in desc order
    sorted_freq_by_index = np.argsort(word_freqs)[::-1]
    sorted_shops = []
    shops_freq_dict = {}
    for i in sorted_freq_by_index:
        # if word not found in website, do not return shop
        if word_freqs[i] != 0:
            #sorted_shops.append((top_places[i]['name'], word_freqs[i]))
            shops_freq_dict[top_places[i]['name']] = word_freqs[i]
            
    return sorted_shops, shops_freq_dict


def get_similarity(url, search_word):
    """
    Count the freq of a word in a webpage
    search_word: word that we are searching for
    url: webpage url
    """
    try:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        text = text_from_html(html)
        split_text = text.split()
        split_text = [i.lower() for i in split_text]

        # preprocess web text
        lemmatizer = WordNetLemmatizer() 
        text_filtered = [w for w in split_text if w.isalpha()]
        text_filtered = [w for w in text_filtered if w not in stop_words]
        text_filtered = [lemmatizer.lemmatize(w) for w in text_filtered]
        text_filtered = ' '.join(word for word in text_filtered)
        text_nlp = nlp(text_filtered)

        # lemmatize search_word
        search_word = search_word.split()       
        search_word = [lemmatizer.lemmatize(i) for i in search_word]
        search_word = [' '.join(search_word)][0]
        search_word = nlp(search_word)
        
        return text_nlp.similarity(search_word)
    
    except:
        return -999


def order_by_sim(search_word, top_places):

    num_dest = len(top_places)
    web_urls  = [top_places[i]['website'] for i in range(num_dest)]

    sim_score = []
    # get freq of search_word in each web
    for url in web_urls:
        score = get_similarity(url, search_word)
        sim_score.append(score)
    
    # sort in desc order
    sorted_score_by_index = np.argsort(sim_score)[::-1]
    sorted_shops = []
    shops_sim_dict = {}
    for i in sorted_score_by_index:
        # if no similarity score, do not rank
        if sim_score[i] != -999:
            #sorted_shops.append((top_places[i]['name'], sim_score[i]))
            shops_sim_dict[top_places[i]['name']] = sim_score[i]

    return sorted_shops, shops_sim_dict

def order_dist_freq(d_freq, d_dist, top_places, w=0.5):
    """
    combines results from both freq of words in the website and distance of shop from source for final result
    d_freq: dict of shop name, search_word_freq
    d_dist: dict of shop name, distance of shop from source
    w: weighting given to freq for final score calculation
    """
    # quick fix to get shop_name, web pair
    num_dest = len(top_places)
    name_web_map  = {top_places[i]['name']: top_places[i]['website'] for i in range(num_dest)}

    # get shops that are in both dicts
    d_intersection = d_dist.keys() & d_freq.keys()

    max_freq = max(d_freq.values())
    max_dist = max(d_dist.values())

    # normalise the dist/ freq
    # so max score is 1
    d_freq_norm = {k: v/max_freq for k,v in d_freq.items() if k in d_intersection}
    d_dist_norm = {k: 1- v/max_dist for k,v in d_dist.items() if k in d_intersection}

    # final score
    res = {}
    for shop in d_freq_norm:
        res[shop] = w * d_freq_norm[shop] + (1-w) *d_dist_norm[shop]

    sorted_res = []
    for shop in sorted(res, key=res.get, reverse=True):
        sorted_res.append((shop, res[shop]))
    
    return res, sorted_res

def order_dist_freq_sim(d_freq, d_dist, d_sim, top_places, wf=0.3, wd=0.3, ws=0.4):
    """
    combines results from both freq of words in the website and distance of shop from source, and word vectors for final result
    d_freq: dict of shop name, search_word_freq
    d_dist: dict of shop name, distance of shop from source
    d_sim: dict of shop name, similarity score of shop with search_word
    wf, wd, ws: weighting on final score given to freq, dist and sim
    """
    # quick fix to get shop_name, web pair
    num_dest = len(top_places)
    name_web_map  = {top_places[i]['name']: top_places[i]['website'] for i in range(num_dest)}

    # get shops that are in both dicts
    d_intersection = d_dist.keys() & d_freq.keys() & d_sim.keys()

    max_freq = max(d_freq.values())
    max_dist = max(d_dist.values())
    max_sim = max((d_sim.values()))

    # normalise the dist/ freq/ sim
    # so max score is 1
    d_freq_norm = {k: v/max_freq for k,v in d_freq.items() if k in d_intersection}
    d_dist_norm = {k: 1- v/max_dist for k,v in d_dist.items() if k in d_intersection}
    d_sim_norm = {k: v/max_sim for k,v in d_sim.items() if k in d_intersection}

    # final score
    res = {}
    for shop in d_freq_norm:
        res[shop] = wf * d_freq_norm[shop] + wd * d_dist_norm[shop] + ws * d_sim_norm[shop]

    sorted_res = []
    for shop in sorted(res, key=res.get, reverse=True):
        sorted_res.append((shop, res[shop], name_web_map[shop]))
    
    return res, sorted_res


def order_shops_final(source_place_id, search_word, top_places, mode='distance'):
    res_json = get_dist(source_place_id, top_places)
    _, shops_dist_dict = get_shops_sorted(res_json, top_places, mode)
    _, shops_freq_dict = order_by_word_freq(search_word, top_places)
    _, shops_sim_dict = order_by_sim(search_word, top_places)
    return order_dist_freq_sim(shops_freq_dict, shops_dist_dict, shops_sim_dict, top_places)
    #return order_dist_freq(shops_freq_dict, shops_dist_dict, top_places)