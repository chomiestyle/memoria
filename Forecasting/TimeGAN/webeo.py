
import numpy as np

arreglo=np.array([[2,3,5,6],[2,6,4,8],[4,6,1,3]])
index=np.random.randint(low=0, high=len(arreglo)-1, size=(2))
number_of_rows = arreglo.shape[0]

random_indices = np.random.choice(number_of_rows, size=3, replace=False)
print(random_indices)
random_rows = arreglo[random_indices, :]

print(random_rows)


