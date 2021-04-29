# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 08:54:36 2021

@author: uros
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

iteration_score = []
iterations = []

f= open('attack_no_defense_output.txt', "r")

for line in f: 

    words = line.split()
    #print(words)
    if 'Pred' in words: #Start/end of an iteration 
        iterations.append(iteration_score)
        iteration_score = []
    
    if '--' in words:
        iteration_score.append(float(words[2]))        


iteration_score = []
f= open('attack_no_defense_2.txt', "r")

for line in f: 

    words = line.split()
    #print(words)
    if 'Pred' in words: #Start/end of an iteration 
        iterations.append(iteration_score)
        iteration_score = []
    
    if '--' in words:
        iteration_score.append(float(words[2]))        




        
iteration_defense_score = []
iterations_defense = []

f= open('attack_with_defense_output.txt', "r")

for line in f: 

    words = line.split()
    #print(words)
    if 'Pred' in words: #Start/end of an iteration 
        iterations_defense.append(iteration_defense_score)
        iteration_defense_score = []
    
    if '--' in words:
        iteration_defense_score.append(float(words[2]))          



f= open('attack_with_defense_2.txt', "r")
iteration_defense_score = []
for line in f: 

    words = line.split()
    #print(words)
    if 'Pred' in words: #Start/end of an iteration 
        iterations_defense.append(iteration_defense_score)
        iteration_defense_score = []
    
    if '--' in words:
        iteration_defense_score.append(float(words[2]))      


scoressss = [0.9956801
,0.96671313
,0.9614875
,0.9897536
,0.98677516
,0.97823024
,0.79454595
,0.9710632
,0.9835601
,0.92207515
,0.984585
,0.98830926
,0.8492267
,0.996567
,0.96671313
,0.9379066
,0.87708837
,0.59590554
,0.9548803,
0.65169835]

#plt.title('Attacks score for each iteration', fontsize=18)

for iteration, s in zip(iterations, scoressss):
    
    plt.plot(list(range(len(iteration)+1)),[s] +iteration, color="#DA7575")
    
for iteration, s in zip(iterations_defense, scoressss):
    print(iteration)
    plt.plot(list(range(len(iteration)+1)),[s]+ iteration, color="#73C388")
    
plt.axhline(y = 0.5, color = 'b', linestyle = '-')
#plt.plot(xstep_trainy_u[0], label="u=5, w=1 b=0", color="blue")
#plt.plot(x,y_u2[0], label="u=10, w=1 b=0", color="green")
plt.ylabel('score', fontsize=16)
plt.xlabel('iteration', fontsize=16)
plt.xticks(np.arange(0, 21, 1))

red_patch = mpatches.Patch(color='#DA7575', label='Original')
green_patch = mpatches.Patch(color='#73C388', label='With defense')
threshold = mpatches.Patch(color='b', label='Success threshold')
plt.legend(handles=[red_patch, green_patch, threshold])
#plt.legend()
plt.show()





iteration_score = []
iterations = []


f= open('attack_no_defense_output.txt', "r")

for line in f: 

    words = line.split()
    #print(words)
    if 'Pred' in words: #Start/end of an iteration 
        print(words)
        iterations.append(iteration_score)
        iteration_score = []
    
    if '--' in words:
        iteration_score.append(float(words[2]))  
        
length = [len(i) for i in iterations]
length2 = sum(length)/ len(length)        
import numpy as np 
np.std(length) 
        
iteration_defense_score = []
iterations_defense = []
attack = []
defense = []

f= open('attack_with_defense_2.txt', "r")

iteration_defense_score = []
for line in f: 

    words = line.split()
    #print(words)
    if 'Pred' in words: #Start/end of an iteration 
        iterations_defense.append(iteration_defense_score)
        iteration_defense_score = []
        attack.append(float(words[4]))
        defense.append(float(words[3]))
        
    
    if '--' in words:
        iteration_defense_score.append(float(words[2]))          

average = [i / j for i, j in zip(defense, attack)]
average_2 = sum(average)/len(average)
import numpy as np
np.std(average)

attackkkk = [0.016025641, 0.016025641, 0.01201923, 0.25641025641]

average = np.std(attackkkk)



length_attack = [len(i) for i in iterations_defense]
length_attack2 = sum(length_attack)/ len(length_attack)        
import numpy as np 
np.std(length_attack) 
        


scoressss = [0.9956801
,0.96671313
,0.9614875
,0.9897536
,0.98677516
,0.97823024
,0.79454595
,0.9710632
,0.9835601
,0.92207515
,0.984585
,0.98830926
,0.8492267
,0.996567
,0.96671313
,0.9379066
,0.87708837
,0.59590554
,0.9548803,
0.65169835]

plt.hist(scoressss, bins=19)
