import numpy as np
import pandas as pd
import json
import io
import os
import requests
import pickle
import tqdm
import time
import math
import matplotlib.pyplot as plt
import tensorly as tl
from scipy.misc import face, imresize
from tensorly.decomposition import non_negative_parafac
from tensorly.decomposition import tucker
from tensorly.decomposition import non_negative_tucker
from tensorly.decomposition import robust_pca
from math import ceil
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import difflib
from get_recommendation import get_recommendation
from is_food_in_menu import in_biz_menu
from matrix_factorization import matrix_factorization

'''
********* 
Collect the menu for restaurant from FourSquare 
*********

'''

count = 0

#Importing file containing restaurant data
f = open("dataset/business.json","r")
#Extracting Restaurant menus using Foursquare's venue and menu APIs
master_dict = {}
p = 0
p1 = 0
for eachLine in f:
    p1 += 1
    if p1%1000 == 0:
        print(p1)
    d = json.loads(eachLine)
    flag = 0
    small_dict = {}
    if 'Food' in d['categories']:
        biz_id = d['business_id']
        if biz_id in b_ids:
            continue
        b_ids.append(biz_id)
        name = str(d['name'])
        name = name.replace(' ','%20')

        # Obtain Latitude Longitude to identify the restaurant
        lat = d['latitude']
        long = d['longitude']
        time.sleep(0.5)

        # Connect and obtain menu data from FourSquare API
        url = 'https://api.foursquare.com/v2/venues/search?ll={}&query={}&client_id=G2J41SNYXH5NV2WYWLJEIF3ZV3GQYMLIOYNFX1XWY1FE50YC&client_secret=0AETTLIUOVPSPKDL4J0GVHGZEFSH311RNVRYKPLAVVO2S33D&v=20180410'.format(str(lat)+','+str(long),name)
        try:
            release_response = (requests.get(url)).json()
            count += 1
            if len(release_response['response']['venues']) != 0:
                venue_id = release_response['response']['venues'][0]['id']
                time.sleep(0.5)
                venue_menu_url = 'https://api.foursquare.com/v2/venues/'+str(venue_id)+'/menu?client_id=G2J41SNYXH5NV2WYWLJEIF3ZV3GQYMLIOYNFX1XWY1FE50YC&client_secret=0AETTLIUOVPSPKDL4J0GVHGZEFSH311RNVRYKPLAVVO2S33D&v=20180410'
                release_response_1 = (requests.get(venue_menu_url)).json()
                count += 1
                small_dict['venue_id'] = venue_id
                small_dict['name'] = name
                if release_response_1['response']['menu']['menus']['count'] != 0:
                    for items in release_response_1['response']['menu']['menus']['items']:
                        dict1 = {}
                        for items1 in items['entries']['items']:
                            if 'name' in items1:
                                dict1[items1['name']] = []
                                for items2 in items1['entries']['items']:
                                    dict1[items1['name']].append(items2['name'])
                            else:
                                flag = 1
                    small_dict['menu'] = dict1
                else:
                    small_dict['menu'] = []
                if flag == 0:
                    master_dict[biz_id] = small_dict
                if release_response_1['response']['menu']['menus']['count'] != 0 and flag == 0:
                    print(master_dict[biz_id])
            else:
                if flag == 0:
                    master_dict[biz_id] = {}
        except:
            continue

# Save the values to a pickle file
with open("./rest_menu.pickle", "rb") as output_file:
    menu = pickle.load(output_file)


count = 0
for k,v in menu.items():
    if 'menu' in v and len(v['menu']) != 0:
        count += 1
print(count)


'''
********* 
Cluster Level 1 and Level 2 Food Items to obtain relevant food item list 
*********

'''

#Extracting food items - Level 2 items are restaurant menu headings and level 1 entities are specific food items available  under menu headings
list1 = []
list2 = []
heading_list_1 = []
heading_list_2 = []
for k,v in m.items():
    for k1,v1 in v.items():
        if k1 == 'menu':
            if len(v1) > 0:
                for k2,v2 in v1.items():
                    list1.append(k2)
                    heading_list_1.append([k2,k,0])
                    for vals in v2:
                        list2.append(vals)
                        heading_list_2.append([vals,k2,k,0])



with open('level1.pickle', 'wb') as handle:
    pickle.dump(list1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('level2.pickle', 'wb') as handle:
    pickle.dump(list2, handle, protocol=pickle.HIGHEST_PROTOCOL)


api_key = 'b94ef970b1bbd7a16e64e58809cfd252'
file_type = 'json'
api_domain = 'https://api.foursquare.com/v2/venues/search?'
releaseURL = fred_series_url + 'release?series_id=' + series_code + url_payload
release_response = (requests.get(releaseURL)).json()

# Clustering of level 1 entities followed by clustering of level 2 entities using fuzzy string matching
cluster = []
ans = []
for k in range(len(list1)):
    if heading_list_1[k][2] != 1:
        temp = difflib.get_close_matches(list1[k],list1,n=1000,cutoff=0.75)
        temp1 = []
        for i in range(len(temp)):
            indices = [index for index, value in enumerate(list1) if value == temp[i]]
            for j in range(len(indices)):
                if heading_list_1[indices[j]][2] == 0:
                    temp1.append([temp[i],heading_list_1[indices[j]][1]])
                    heading_list_1[indices[j]][2] = 1
        if len(temp1) > 0:
            print(list1[k],len(temp1))
            cluster.append(temp1)
            ans.append([list1[k],len(temp1)])


m1 = {}
for i in range(len(cluster)):
    m1[ans[i][0]] = []
    for j in range(len(cluster[i])):
        for k in range(len(m[cluster[i][j][1]]['menu'][cluster[i][j][0]])):
            m1[ans[i][0]].append(m[cluster[i][j][1]]['menu'][cluster[i][j][0]][k])



m2 = {}
for k,v in m1.items():
    temp3 = []
    m2[k] = []
    print(k)
    for i in range(len(v)):
        temp3.append([v[i],0])
    for i in range(len(v)):
        if temp3[i][1] == 0:
            temp = difflib.get_close_matches(v[i],v,n=1000,cutoff = 0.7)
            temp1 = []
            for j in range(len(temp)):
                indices = [index for index, value in enumerate(v) if value == temp[j]]
                for l in range(len(indices)):
                    if temp3[indices[l]][1] == 0:
                        temp1.append(v[i])
                        temp3[indices[l]][1] = 1
            if len(temp1) > 0:
                #print(v[i],len(temp1))
                m2[k].append(v[i])


# Store the final clustered list of food items
with open('final_clusters.pickle', 'wb') as handle:
    pickle.dump(m2, handle)


l1 = []
l2 = []
for k,v in m2.items():
    l1.append(k)
    for values in v:
        l2.append(values)

with open('cluster_level_1.pickle', 'wb') as handle:
    pickle.dump(l1, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('cluster_level_2.pickle', 'wb') as handle:
    pickle.dump(l2, handle, protocol=pickle.HIGHEST_PROTOCOL)


list3 = []
ans1 = []
heading_list1 = []
for i in range(len(ans)):
    if ans[i][1] < 5:
        list3.append(ans[i][0])
        heading_list1.append([ans[i][0],0])


for k in range(len(list3)):
    if heading_list1[k][1] != 1:
        temp = difflib.get_close_matches(list3[k],list3,n=1000,cutoff=0.65)
        temp1 = []
        for i in range(len(temp)):
            indices = [index for index, value in enumerate(list3) if value == temp[i]]
            for j in range(len(indices)):
                if heading_list1[indices[j]][1] == 0:
                    temp1.append(temp[i])
                    heading_list1[indices[j]][1] = 1
        if len(temp1) > 0:
            print(list3[k],len(temp1))
            print(temp1)
            ans1.append([list3[k],len(temp1)])


for vals in list2:
    print(vals)
    print(difflib.get_close_matches(vals, list2))

user_biz = []

#Loading review data 
f = open("dataset/review.json","r")
for eachLine in f:
    d = json.loads(eachLine)
    user_biz.append([d['business_id'],d['user_id'],d['stars']])


with open('user_biz.pickle', 'wb') as handle:
    pickle.dump(user_biz, handle, protocol=pickle.HIGHEST_PROTOCOL)


count = 0
for k,v in m2.items():
    count += len(m2[k])

with open("./rest_menu.pickle", "rb") as output_file:
    m = pickle.load(output_file)


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


with open("./visionData.p", "rb") as output_file:
    pic_menu = pickle.load(output_file)


keys = []
for k,_ in pic_menu.items():
    keys.append(k)

#Clustering photo_menu to condense the data to a smaller size
for i in range(len(keys)):
    for j in range(len(photo_menu[keys[i]])):
        already_used.append(0)
    for j in range(len(photo_menu[keys[i]])):
        if already_used[j] == 0:
            temp = difflib.get_close_matches(photo_menu[keys[i]][j],photo_menu[keys[i]],n=1000,cutoff=0.75)
            temp1 = []
            for k in range(len(temp)):
                indices = [index for index, value in enumerate(photo_menu[keys[i]]) if value == temp[i]]
                for l in range(len(indices)):
                    if already_used[indices[l]] == 0:
                        temp1.append()
                        already_used[indices[l]] = 1

#Adding food items extracted from photos in photo_menu

photo_menu = {}
for i in range(len(keys)):
    photo_menu[keys[i]] = []
    print(i)
    for j in range(len(m[keys[i]])):
        if len(m[keys[i]][j]) > 1:
            for k in range(len(m[keys[i]][j])):
                photo_menu[keys[i]].append(m[keys[i]][j][k][0]['label'])

with open("./results104000.p", "rb") as output_file:
    m = pickle.load(output_file)

with open("./user_biz_rating_matrix.p", "rb") as output_file:
    mean_user_business_rating_dict = pickle.load(output_file)

#Food Item, Restaurant and Users lists for 3-way tensor
biz_list = []
user_list = []
food_list = []
for k,v in m.items():
    if len(v) != 0:
        if(len(user_list)) >= 100:
            break
        biz_list.append(k)
        for review in v:
            if review[0] in user_id_final and len(user_list) < 100:
                user_list.append(review[0])
                for food_items in review[1]:
                    food_list.append(food_items)
                for food_items in review[2]:
                    food_list.append(food_items)
            else:
                break


food_list = set(food_list)
biz_list = set(biz_list)
user_list = set(user_list)


biz_dict = {}
user_dict = {}
food_dict = {}
biz_count = 0
user_count = 0
food_count = 0
for i in biz_list:
    biz_dict[i] = biz_count
    biz_count += 1
for i in user_list:
    user_dict[i] = user_count
    user_count += 1
for i in food_list:
    food_dict[i] = food_count
    food_count += 1
reverse_biz_dict = {v:k for k,v in biz_dict.items()}
reverse_user_dict = {v:k for k,v in user_dict.items()}
reverse_food_dict = {v:k for k,v in food_dict.items()}


final_tensor = np.zeros((len(biz_list),len(user_list),len(food_list)))

#Making the final tensor and adding data
temp_user_list = []
for k,v in m.items():
    if len(v) != 0:
        if len(user_list) >= 100:
            break
        for review in v:
            if review[0] in user_id_final and len(temp_user_list)<100:
                temp_user_list.append(review[0])
                for food_items in review[1]:
                    for biz_id in biz_list:
                        if in_biz_menu(biz_id, food_items):
                            final_tensor[biz_dict[biz_id]][user_dict[review[0]]][food_dict[food_items]] = mean_user_business_rating_dict[(k,review[0])]        
                for food_items in review[2]:
                    for biz_id in biz_list:
                        if in_biz_menu(biz_id, food_items):
                            final_tensor[biz_dict[biz_id]][user_dict[review[0]]][food_dict[food_items]] = mean_user_business_rating_dict[(k,review[0])]
            else:
                break


final_tensor = tl.tensor(final_tensor, dtype='float64')

#Training Tucker and PARAFAC tensor decomposition models
core, factors = non_negative_tucker(final_tensor, ranks=[120,20,220], n_iter_max=100, init='random', tol=0.00001, random_state=None, verbose=True)
factors = non_negative_parafac(final_tensor, rank=70, n_iter_max=100, init='random', tol=0.00001, random_state=None, verbose=True)

reconstructed_tensor = tl.kruskal_to_tensor(factors)

nonzero_mat = np.nonzero(final_tensor)

#Computing RMSE
error = 0
for i in range(len(nonzero_mat[0])):
    error += final_tensor[nonzero_mat[0][i]][nonzero_mat[1][i]][nonzero_mat[2][i]]-reconstructed_tensor[nonzero_mat[0][i]][nonzero_mat[1][i]][nonzero_mat[2][i]]

#Making User - Food, User - restaurant and Restuarant - Food matrices
user_food_mat = np.zeros((len(user_list),len(food_list)))
biz_food_mat = np.zeros((len(biz_list),len(food_list)))
biz_food_count = np.zeros((len(biz_list),len(food_list)))
user_biz_mat = np.zeros((len(user_list),len(biz_list)))

for k,v in m.items():
    if len(v) != 0:
#         print(k,v)
        for review in v:
            user_biz_mat[user_dict[review[0]]][biz_dict[k]] = mean_user_business_rating_dict[(k,review[0])]
            for food_items in review[1]:
                if food_items in food_list:
                    user_food_mat[user_dict[review[0]]][food_dict[food_items]] = 1
                    biz_food_mat[biz_dict[k]][food_dict[food_items]] = (biz_food_mat[biz_dict[k]][food_dict[food_items]]*biz_food_count[biz_dict[k]][food_dict[food_items]]+review[3])/(biz_food_count[biz_dict[k]][food_dict[food_items]]+1)
                    biz_food_count[biz_dict[k]][food_dict[food_items]] += 1
            for food_items in review[2]:
                if food_items in food_list:
                    user_food_mat[user_dict[review[0]]][food_dict[food_items]] = 1
                    biz_food_mat[biz_dict[k]][food_dict[food_items]] = (biz_food_mat[biz_dict[k]][food_dict[food_items]]*biz_food_count[biz_dict[k]][food_dict[food_items]]+review[3])/(biz_food_count[biz_dict[k]][food_dict[food_items]]+1)
                    biz_food_count[biz_dict[k]][food_dict[food_items]] += 1


for i in range(len(user_list)):
    for j in range(len(biz_list)):
        user_biz_mat[i][j] = mean_user_business_rating_dict[(reverse_biz_dict[i],reverse_user_dict[j])]

user_id = {}
for k,v in m.items():
    for review in v:
        user_id[review[0]] = 0


for k,v in m.items():
    for review in v:
        user_id[review[0]] += 1

#Filtering out users with less than 15 reviews
user_id_plenty = {}
for k,v in user_id.items():
    if v >= 15:
        user_id_plenty[k] = v



user_id_final = []
for k,v in user_id_plenty.items():
    user_id_final.append(k)


user_id_mat = {}

#Segregating restaurants and food items based on Users
user_id_food_list = {}
user_id_biz_list = {}
for k,v in m.items():
    for review in v:
        if review[0] in user_id_plenty:
            user_id_biz_list[review[0]] = []
            for food_item in review[1]:
                user_id_food_list[review[0]] = []
            for food_item in review[2]:
                user_id_food_list[review[0]] = []

for k,v in m.items():
    for review in v:
        if review[0] in user_id_plenty:
            user_id_biz_list[review[0]].append(k)
            for food_item in review[1]:
                user_id_food_list[review[0]].append(food_item)
            for food_item in review[2]:
                user_id_food_list[review[0]].append(food_item)
                



for k,_ in user_id_plenty.items():
    user_id_food_list[k] = set(user_id_food_list[k])
    user_id_biz_list[k] = set(user_id_biz_list[k])



user_id_biz_map = {}
user_id_food_map = {}
for k,_ in user_id_plenty.items():
    user_id_biz_map[k] = {}
    user_id_food_map[k] = {}

for k,_ in user_id_plenty.items():
    for index,item in enumerate(user_id_biz_list[k]):
        user_id_biz_map[k][item] = index
    for index,item in enumerate(user_id_food_list[k]):
        user_id_food_map[k][item] = index



user_id_food_size = []
user_id_biz_size = []
for k,v in user_id_food_list.items():
    for vals in v:
        user_id_food_size.append(vals)
for k,v in user_id_biz_list.items():
    for vals in v:
        user_id_biz_size.append(vals)
user_id_food_size = set(user_id_food_size)
user_id_biz_size = set(user_id_biz_size)


user_id_data = {}
for k,_ in user_id_plenty.items():
    user_id_data[k] = []


#Forming 2-D matrices for all the users 
for k,v in m.items():
    for review in v:
        if review[0] in user_id_plenty:
            #print([user_id_biz_map[review[0]][k],user_id_food_map[review[0]][food_item],mean_user_business_rating_dict[(k,review[0])]])
            for food_item in review[1]:
                user_id_data[review[0]].append([user_id_biz_map[review[0]][k],user_id_food_map[review[0]][food_item],mean_user_business_rating_dict[(k,review[0])]])
            for food_item in review[2]:
                user_id_data[review[0]].append([user_id_biz_map[review[0]][k],user_id_food_map[review[0]][food_item],mean_user_business_rating_dict[(k,review[0])]])



from sklearn.model_selection import train_test_split


#Splits data in 2-D matrices into test and train
user_id_test_data = {}
user_id_train_data = {}
for k,_ in user_id_plenty.items():
    user_id_train_data[k] = []
    user_id_test_data[k] = []
    for i in range(len(user_id_data[k])):
        user_id_train_data[k], user_id_test_data[k] = train_test_split(user_id_data[k],test_size=0.2)


#Running NMF on all the 2-D matrices and recording the RMSE values
P_dict = {}
Q_dict = {}
b_i_dict = {}
b_u_dict = {}
b_dict = {}
user_count = 0
RMSE = []
for k,_ in user_id_plenty.items():
    user_count += 1
    if user_count%5 == 0:
        print('User Number : ' + str(user_count))
    P = np.random.rand(len(user_id_biz_list[k]),2)/2
    Q = np.random.rand(len(user_id_food_list[k]),2)/2
    b_i = np.random.rand(len(user_id_food_list[k]))
    b_u = np.random.rand(len(user_id_biz_list[k]))
    mean = 0
    mean_count = 0
    for i in range(len(user_id_train_data[k])):
        mean += user_id_train_data[k][i][2]
        mean_count += 1
    b = mean/mean_count
    b,b_u,b_i,P,Q,e = matrix_factorization(b,b_u,b_i,P,Q,2,30,0.01,0.02,user_id_train_data[k], user_id_test_data[k])
    P_dict[k] = P
    Q_dict[k] = Q
    b_i_dict[k] = b_i
    b_u_dict[k] = b_u
    b_dict[k] = b
    RMSE.append(e)

