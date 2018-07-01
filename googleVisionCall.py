'''
Obtain all the food labels by using the photos uploaded by users.
These photographs are used as an input to the Google's Vision API and the labels denoting to the photographs obtained.
'''


from google.cloud import vision
from google.cloud.vision import types
import io
import json
import pickle
from tqdm import tqdm


# String Literals
image_dir_path = './yelp_photos/photos/'
photos_json_path = './dataset/photos.json'
business_id_key = 'business_id'
label_key = 'label'
__FOOD__ = 'food'
__PhotoID__ = 'photoID'
__Photo_ID__ = 'photo_id'
__Label__ = 'label'
__Score__ = 'score'
__CAPTION__ = 'caption'

# Function to get the labels from Vision API
def getFromVisionAPI(imageID):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Generate the image file path
    file_name = image_dir_path + imageID

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    # Annotate the image and retrieve results
    image = types.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    return labels

# Function to get label details
def getLabelDetails(labels):
    label_details = []
    # Iterate over all labels and create its map
    for label in labels:
        # Dictionaries to store description and score
        label_dict = dict()
        score_dict = dict()
        # List to store current Labels
        curr_label = []

        label_dict[__Label__] = label.description
        score_dict[__Score__] = label.score

        # Append to the local dictionary
        curr_label.append(label_dict)
        curr_label.append(score_dict)

        #Append to the label details
        label_details.append(curr_label)

    return label_details

# Driver function to create a map of Business and Food Photos and Captions
def createBusinessFoodMap():
    # Create the data and photo dictionary
    data_dict = dict()
    photo_dict = dict()
    caption_dict = dict()
    # List with all the photo details
    photo_details = []

    # Open the photos JSON file
    photos_json = open(photos_json_path, 'r')

    # Parse through each JSON entry of photos
    for line in tqdm(photos_json):
        label_details = []
        # Parse the line in JSON format
        parsed_line = json.loads(line)
        # Execute only if the photo is of Food
        if (parsed_line[label_key] == __FOOD__):

            # Add the photo caption
            caption_dict[__CAPTION__] = parsed_line[__CAPTION__]

            # Add the photo ID
            photo_dict[__PhotoID__] = parsed_line[__Photo_ID__]

            # Get the response and labels for the current image
            labels = getFromVisionAPI(parsed_line[__Photo_ID__] + '.jpg')

            label_details = getLabelDetails(labels)

            # Append to the business list dictionary
            photo_details.append(photo_dict)
            photo_details.append(caption_dict)
            photo_details.append(label_details)

            # See if the key exists in dictionary
            if(parsed_line[business_id_key] in data_dict):
                data_dict[parsed_line[business_id_key]].append(photo_details)
            else:
                data_dict[parsed_line[business_id_key]] = photo_details

            #print(data_dict)
            #print(json.dumps(data_dict))

    # Convert the whole info into JSON
    final_data = data_dict
    print(data_dict)
    pickle.dump(data_dict, open("visionData.p", "wb"))


# Main driver function
if __name__ == '__main__':
    createBusinessFoodMap()
