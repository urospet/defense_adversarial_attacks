import numpy as np

index = 0
number_pred = 0 
number_attack_detected = 0 
queries = np.ones((100,300),dtype=np.uint8)*-1
outputs = np.ones((100,2),dtype=np.float32)*-1