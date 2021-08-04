from NBLEA.Low import *
import pprint
import matplotlib.pyplot
import pandas as pd 

graph = Map("Instances/6.10.4.txt")

print(graph.d_time)
print(graph.t_time)

df = pd.DataFrame(graph.d_time)

## save to xlsx file

filepath = '10.4.xlsx'

df.to_excel(filepath, index=False)

if 0:
    graph.draw()
    plt.show()
    sys.exit()

t_route = [5, 3, 1, 4, 6, 7, 2] 

cost, best_route_details, best_u_tour = solver(graph, t_route)  
pprint.pprint(best_route_details)

#fitness = 1 - (cost khi có drone hỗ trợ / max_cost)
#max_cost: Tổng thời gian chờ của mẫu ứng với hành trình t_route khi không có drone hỗ trợ 
#best_u_tour: hành trình của uav 

print("=====================================")
print(f't route: {t_route}')
print(f'uav_tour: {best_u_tour}')
print(cost)


#plt.show()