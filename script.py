# # Find the Missing Number in a Sequence
# input: [1, 2, 4, 6, 3, 7, 8]
# output: 5
 
# input: [-3, 0, 1, -1]
# output: -2
 
# input: [1, 3]
# output: 2

def find_missing_value(arr):
    if len(arr) == 0:
        print("\nEmpty array")
        return 
    
    new_arr = [-999] * (len(arr)+1)
    
    min_value = min(arr)
    new_arr[0] = min_value
    for i in range(0, len(arr)):
        new_arr[arr[i]-min_value] = arr[i]

    for i in range(0, len(new_arr)):
        if new_arr[i] == -999:
            if i != len(new_arr)-1:
                print(f"\nMissing value is: {new_arr[i-1]+1}")
            else:
                print("\nNo missing values")


arr = [1, 2, 4, 6, 3, 7, 8]
find_missing_value(arr)
find_missing_value([-3, 0, 1, -1])
find_missing_value([1, 3])
find_missing_value([3, 4, 5])
find_missing_value([])


# arr = [0] * len(input) + 1
# [0, 0, 0, 0, 0, 0, 0, 0]  

# [1, 2, 3, 3, 0, 6, 7, 8]

# [1, 2, 3, 4, 0, 6, 7, 8]