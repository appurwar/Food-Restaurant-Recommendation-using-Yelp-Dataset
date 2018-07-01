#Outputs recommendations for a given user 
def recommendation(user_number):
    
    example_mat = reconstructed_tensor[:, user_number, :]
    index_matrix = []
    for i in range(example_mat.shape[0]):
        index_matrix.append([])
        for j in range(example_mat.shape[1]):
            index_matrix[i].append(str((i,j)))
    index_matrix = np.array(index_matrix)
    index_matrix.shape = (example_mat.shape[0] * example_mat.shape[1])
    top_200_indices = index_matrix[np.argsort(np.matrix.flatten(example_mat))[-200:]]
    used_food = []
    used_biz = []
    for i in range(len(top_200_indices)):
        indices = top_200_indices[i].split(',')
        if in_biz_menu(reverse_biz_dict[int(indices[0][1:])],reverse_food_dict[int(indices[1][0:-1])]):
            if (int(indices[1][0:-1]) not in used_food) and (int(indices[0][1:]) not in used_biz):
                used_food.append(int(indices[1][0:-1]))
                used_biz.append(int(indices[0][1:]))
#             print(reconstructed_tensor[int(indices[0][1:])][user_number][int(indices[1][0:-1])])
                print('Restaurant Name : ' + str(menu[reverse_biz_dict[int(indices[0][1:])]]['name']))
                print('Food Name : ' + str(reverse_food_dict[int(indices[1][0:-1])]))