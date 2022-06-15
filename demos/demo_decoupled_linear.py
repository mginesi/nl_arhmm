import numpy as np
from dataset.jigsaw import jigsaw
from nl_arhmm.arhmm import Decoupled_Linear_ARHMM
import matplotlib.pyplot as plt

model = Decoupled_Linear_ARHMM(3, 3, 2)
data = jigsaw()
batch = data.give_random_batch('np', 10)
cart_batch = [data.give_cartesian(batch[_n]) for _n in range(len(batch))]

model.initialize(cart_batch)
model.em_algorithm(cart_batch)
