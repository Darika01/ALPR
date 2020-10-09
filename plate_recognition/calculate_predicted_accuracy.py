from itertools import zip_longest

def calculate_predicted_accuracy(actual_plate, predict_plate):
    accuracy = "0 %"
    num_matches = 0
    if actual_plate == predict_plate: 
        accuracy = "100 %"
    else: 
        if len(actual_plate) == len(predict_plate): 
            for a, p in zip_longest(actual_plate, predict_plate, fillvalue='?'): 
                if a == p:
                    num_matches += 1
            if num_matches == 0:
                accuracy = "0 %"
            accuracy = str(round((num_matches / len(actual_plate)), 2) * 100)
            accuracy += "%"
    print("     ", actual_plate, "\t\t\t", predict_plate, "\t\t  ", accuracy)