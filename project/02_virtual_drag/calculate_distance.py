import math


def calculate_distance_btween_two_fingers(index_finger, middle_finger):
    x = (index_finger[0]- middle_finger[0])**2
    y = (index_finger[1] - middle_finger[1])**2
    # print(math.sqrt(x + y))
    return math.sqrt(x + y)