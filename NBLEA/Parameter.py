from ast import Num

#Model parameter
T_VEL = 0.58#mile per minute 35/60
D_VEL = 0.83

TECHNICAN_NUMS = 2
TECHNICAN_CAN_WAIT = True


#Constrain
WORK_TIME = 120 
T = 30


#GA parameter 
POP_SIZE = 30
ELITE_SIZE = 6  # even int
MUTATION_RATE = 0.01
GENERATIONS = 100


#START LOW 
INIT_PHEROMONE = 0
MAX_ITERATION = 50
NUM_LANTS = 50   


#ANT 
LOCAL_EVAPORATION_RATE = 2 / NUM_LANTS
DEPOSIT_RATE = 3 / MAX_ITERATION
GLOBAL_EVAPORATION_RATE = 0.7 / MAX_ITERATION

#=====END LOW 



INFINITY = 9999999.0