#Checks whether a food item is offered by the Restaurant or not
def in_biz_menu(biz_id, food):
    level1 = []
    level2 = []
    for k,v in menu[biz_id]['menu'].items():
        level1.append(k)
        for food_item in v:
            level2.append(food_item)
    temp1 = difflib.get_close_matches(food,level1,n=3,cutoff=0.7)
    temp2 = difflib.get_close_matches(food,level2,n=3,cutoff=0.7)
    if len(temp1) > 0 or len(temp2) > 0:
#         print('Food Found')
        return True
    else:
        return False