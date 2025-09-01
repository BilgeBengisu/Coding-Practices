'''
    * Activity Selection Problem using Greedy Algorithm
    * Time Complexity: O(n log n) due to sorting
    * Space Complexity: O(1)
    We start choosing the first activity and then choosing 
    the next activity which has the start time greater than 
    or equal to the finish time of the previously selected activity.
'''


data = {
  "start_time": [2 , 6 , 4 , 10 , 13 , 7],
  "finish_time": [5 , 10 , 8 , 12 , 14 , 15],
  "activity": ["Homework" , "Presentation" , "Term paper" , "Volleyball practice" , "Biology lecture" , "Hangout"]
}


