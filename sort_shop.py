from algo import *
from key import *

postcode= 'CB2 1RF'
search_keyword = 'shoes'

#top_places = get_places_mp(postcode)

source_id = get_placeid(postcode)
final_res_dict, final_sorted_res = order_shops_final(source_id, search_keyword, top_places[:20])

print(final_sorted_res)
print(final_res_dict)