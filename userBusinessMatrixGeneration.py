'''
Generate the 'User - Business' Matrix, by calculating the average ratings of user for a particular business
'''

import json
import pickle

# Fetch the list of valid business IDs
valid_business_list = pickle.load(open('business_ids_2.0.p', 'r' ))

# Open the JSON containing the user reviews
review_data = open('review.json', 'r')

# Dictionary to store all the ratings
user_business_rating_dict = dict()

# Dictionary to store mean ratings
mean_user_business_rating_dict = dict()

# Iterate over all the reviews
for curr_review in review_data:
    # Get the current user review
    review = json.loads(curr_review)
    # If the review belongs to a valid business ID
    if review['business_id'] in valid_business_list:
        if (review['business_id'], review['business_id']) not in user_business_rating_dict:
            # Store the rating and count for each
            user_business_rating_dict[(review['business_id'], review['user_id'])] = (review['stars'],1)
        else:
            # Update the existing businesses, with the new rating - average over all ratings
            curr_rating = user_business_rating_dict.get((review['business_id'], review['user_id']))
            mean_rating = ((int(curr_rating[1]) * float(curr_rating[0])) + float(review['stars']))/int(curr_rating[1])+1
            user_business_rating_dict[(review['business_id'], review['user_id'])] = (mean_rating, curr_rating[1]+1)

# Create a dictionary of final business IDs and ratings
for key, value in user_business_rating_dict.items():
    mean_user_business_rating_dict[key] = value[1]

# Store all the data in a pickle file
pickle.dump(mean_user_business_rating_dict, open("user_biz_rating_matrix.p", "wb"))
