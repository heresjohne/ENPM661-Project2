import queue
import math
from math import pi
import numpy as np
import pygame
from collections import defaultdict
import decimal

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

 


np.set_printoptions(threshold=np.inf)

global rows, columns, size_m, row_moves, column_moves, start_r, start_s, goal_r, goal_c, nodes_in_next, nodes_in_layer

# globals
row_moves = queue.Queue()
column_moves = queue.Queue()
vari = 300
#------------------------------------------------------------------------------
# Pygame stuff
#------------------------------------------------------------------------------
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
 # This sets the WIDTH and HEIGHT of each grid location
WIDTH = 2
HEIGHT = 2
MARGIN = 1
grid = []

for row in range(vari):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(400):
        grid[row].append(0)  # Append a cell

stuff = []

# Initialize pygame
pygame.init()

# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [1200, 900]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set title of screen
pygame.display.set_caption("Array Backed Grid")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()
#------------------------------------------------------------------------------
# Pygame stuff
#------------------------------------------------------------------------------


# function here

state_dict = defaultdict(list)

rows = 300
columns = 400

stuff = []
# Other vars
move_counter = 0
nodes_in_layer = 1
nodes_in_next = 0
reached_goal = False

visited = np.full((rows+1, columns+1), False)

#------------------------------------------------------------------------------
# Direction vectors and exploration function
#------------------------------------------------------------------------------

# [-1,0]-> west, [1,0]-> east, [0,1]-> north, [0,-1]-> south, [1,1]-> north-east, [1,-1]-> south-east, [-1,-1]-> south-west, [-1,1]-> north-west
dcol = [-1, +1, 0, 0, 1, 1, -1, -1]  # dr
drow = [0, 0, +1, -1, 1, -1, -1, 1]  # dc

# 8 Directions total

def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += decimal.Decimal(step)

def findFirstAndLast(arr, n, x):
    junk_array = []
    pos_array = []
    first = -1
    last = -1
    for i in range(0, n):
        if (x != arr[i]):
            continue
        if (first == -1):
            first = i

        last = i
        junk_array.append(i)
    if (first != -1):
        pos_array = [first, junk_array[1]]
        return pos_array


def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(round(x1), round(x2 + 1)):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


def points_on_circumference(center=(0, 0), r=50, n=100):
    return [
        (
            round(center[1]+(math.cos(2 * pi / n * x) * r)),  # x
            round(center[0] + (math.sin(2 * pi / n * x) * r))  # y

        ) for x in list(float_range(0, 100, '0.2'))]

def points_in_ellipse(u, v, a, b, n=100):

    return [
        (
            round(u+a*math.cos(t)),  # x
            round(v+b*math.sin(t))  # y

        ) for t in list(float_range(0, 100, '0.2'))]

def graph_maker(rows, columns):
    size_m = np.full((rows, columns), '.')


    #Elipsoids: 1 - Elipse
    data = np.array(points_on_circumference(center=(70, 90), r=35))
    # Lines: 1 - C Shape 
    point1 = (get_line(200, 230, 200, 280))
    point2 = (get_line(200, 280, 230, 280))
    point3 = (get_line(230, 280, 230, 270))
    point4 = (get_line(230, 270, 210, 270))
    point5 = (get_line(210, 270, 210, 240))
    point6 = (get_line(210, 240, 230, 240))
    point7 = (get_line(230, 240, 230, 230))
    point8 = (get_line(230, 230, 200, 230))
    
    # Lines: 2 - Tilted Rectangle 
    # point1 = (get_line(48,108,36.5,124.4))
    # point2 = (get_line(37,124,159.4,210.4))


    rec1 = (get_line(48, 108, 36, 124))
    rec2 = (get_line(36, 124, 159, 210))
    rec3 = (get_line(159, 210, 171, 194))
    rec4 = (get_line(171, 194, 48, 108))

    shape1 = (get_line(330, 63, 288, 105))
    shape2 = (get_line(288, 105, 328, 146))
    shape3 = (get_line(328, 146, 354, 148))
    shape4 = (get_line(354, 148, 383, 171))
    shape5 = (get_line(383, 171, 383, 116))
    shape6 = (get_line(383, 116, 330, 63))

    rect = rec1+rec2+rec3+rec4

    shape = shape1+shape2+shape3+shape4+shape5+shape6
    
    point = point1+point2+point3+point4+point5+point6+point7+point8
    data2 = np.array(point)
    data4 = np.array(rect)
    data5 = np.array(shape)

    #Elipsoids: 2 - Elipse
    data3 = np.array(points_in_ellipse(u = 246, v = 145 , a= 120 , b = 60))

    # print(data3)
    b, z = data.T
    x, y = data2.T
    a, q = data3.T
    j, k = data4.T
    l, m = data5.T

    for i in range(len(x)):
        blank = []
        blank = data2[i]
        size_m[[blank[0]],[blank[1]]] = '#'
    for i in range(len(b)):
        blank = []
        blank = data[i]
        size_m[[blank[1]], [blank[0]]] = '#'
    for i in range(len(a)):
        blank = []
        blank = data3[i]
        size_m[[blank[1]], [blank[0]]] = '#'
    for i in range(len(j)):
        blank = []
        blank = data4[i]
        size_m[[blank[1]], [blank[0]]] = '#'
    for i in range(len(l)):
        blank = []
        blank = data5[i]
        size_m[[blank[1]], [blank[0]]] = '#'

    return size_m
# data3 = np.array(points_in_ellipse(u = 246, v = 145 , a= 120 , b = 60))
# round(u+a*math.cos(t)),  # x
# round(v+b*math.sin(t))  # y
def in_ellipse( h, k, x, y, a, b): 
    checks = ((math.pow((x - h), 2) // math.pow(a, 2)) + 
         (math.pow((y - k), 2) // math.pow(b, 2))) 
    return checks 

def in_circle(center_x, center_y, radius, x, y):
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2
# data = np.array(points_on_circumference(center=(70, 90), r=35))

#------------------------------------------------------------------------------
# Initialize Graph and Goal and Start
#------------------------------------------------------------------------------
n = columns
x = '#'
size_m = graph_maker(rows+1, columns+1)

for i in range(rows):
    stuff.append(findFirstAndLast(size_m[i], n, x))
for i in range(len(stuff)):
    blank = []
    blank = stuff[i]
    # print(blank)
    if blank is not None:
        for b in range(blank[0], abs(blank[0]-blank[1])+1):
            size_m[i, b] = '#'
        for x in range(blank[0], abs(blank[0]-blank[1])+1):
            size_m[i, x] = '#'
        for a in range(blank[0], abs(blank[0]-blank[1])+1):
            size_m[i, a] = '#'
        for j in range(blank[0], abs(blank[0]-blank[1])+1):
            size_m[i, j] = '#'
        for l in range(blank[0], abs(blank[0]-blank[1])+1):
            size_m[i, l] = '#'
    else: continue
    
    # point1 = (get_line(200, 230, 200, 280))
    # point2 = (get_line(200, 280, 230, 280))
    # point3 = (get_line(230, 280, 230, 270))
    # point4 = (get_line(230, 270, 210, 270))
    # point5 = (get_line(210, 270, 210, 240))
    # point6 = (get_line(210, 240, 230, 240))
    # point7 = (get_line(230, 240, 230, 230))
    # point8 = (get_line(230, 230, 200, 230))

    # rec1 = (get_line(48, 108, 36, 124))
    # rec2 = (get_line(36, 124, 159, 210))
    # rec3 = (get_line(159, 210, 171, 194))
    # rec4 = (get_line(171, 194, 48, 108))

    # shape1 = (get_line(330, 63, 288, 105))
    # shape2 = (get_line(288, 105, 328, 146))
    # shape3 = (get_line(328, 146, 354, 148))
    # shape4 = (get_line(354, 148, 383, 171))
    # shape5 = (get_line(383, 171, 383, 116))
    # shape6 = (get_line(383, 116, 330, 63))

polygon = Polygon([(200, 230), (200, 280), (230, 280), (230, 270), (210, 270), (210, 240), (230, 240), (230, 230), (200, 230)])
polygon1 = Polygon([(48, 108), (36, 124), (159, 210), (171, 194), (48, 108)])
polygon2 = Polygon([(330, 63), (288, 105), (328, 146), (354, 148), (383, 171), (383, 116), (330, 63)])

goal_c = int(input("Enter Column Goal:"))
goal_r = int(input("Enter Row Goal:"))


start_c = int(input("Enter Column Start:"))
start_r = int(input("Enter Row Start:"))

goal = Point(goal_c, goal_r)
start = Point(start_c, start_r)
# Check if obstacles are in chosen start and goal locations
if goal_c == start_c and goal_r == start_r:
  print('Can not start at goal')
  quit()
if size_m[goal_r][goal_c] == '#':
  print('Obstacle is in chosen goal location')
  quit()
if size_m[start_r][start_c] == '#':
  print('Obstacle is in chosen start location')

check_dist1 = in_circle(90,70,35,goal_c,goal_r)
check_dist2 = in_circle(90,70,35,start_c,start_r)
check_dist3 = in_ellipse(246,145,goal_c,goal_r,120,60)
check_dist4 = in_ellipse(246,145,start_c,start_r,120,60)

if polygon.contains(goal):
  print('Can not have goal in boundary')
  quit()
if polygon1.contains(goal):
  print('Can not have goal in boundary')
  quit()
if polygon2.contains(goal): 
  print('Can not have goal in boundary')
  quit()

if polygon.contains(start):
  print('Can not have start in boundary')
  quit() 
if polygon1.contains(start):
  print('Can not have start in boundary')
  quit() 
if polygon2.contains(start): 
  print('Can not have start in boundary')
  quit() 
if check_dist1:
  print('Can not have goal in boundary')
  quit()
if check_dist2:
  print('Can not have start in boundary')
  quit()    
if check_dist3 <= 1:
  print('Can not have goal in boundary')
  quit()
if check_dist4 <= 1:
  print('Can not have start in boundary')
  quit()    



size_m[goal_r][goal_c] = '$'


#------------------------------------------------------------------------------
# Main Function
#------------------------------------------------------------------------------
def search(graph, start, end):
    # maintain a queue of paths
    queue = []
    # push the first path into the queue
    queue.append([start])
    while queue:
        # get the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        # path found
        if node == end:
            return path
        # enumerate all adjacent nodes, construct a new path and push it into the queue
        for adjacent in graph.get(node, []):
            new_path = list(path)
            new_path.append(adjacent)
            queue.append(new_path)


def convert(blah):
    res = (''.join(["{:03d}".format(blah)]))
    return res

def movement(row_var, column_var):
  global nodes_in_layer, nodes_in_next, move_counter
  for i in range(0, 8):
    column_var_temp = column_var+dcol[i]
    row_var_temp = row_var+drow[i]

    # If trying to move outside of bounds, don't
    if row_var_temp < 0 or column_var_temp < 0: continue
    if row_var_temp >= rows or column_var_temp >= columns: continue

    # If trying to move to visited node or obstacle
    if visited[row_var_temp][column_var_temp]: continue
    if size_m[row_var_temp][column_var_temp] == '#': continue

    row_moves.put(row_var_temp)
    # print(f"{row_moves.qsize()}row moves are")
    column_moves.put(column_var_temp)
    # print(f"{column_moves.qsize()} column moves are")
    visited[row_var_temp][column_var_temp] = True
    grid[row_var_temp][column_var_temp]=1
    state_dict[str(convert(row_var))+str(convert(column_var))].append(str(convert(row_var_temp))+str(convert(column_var_temp)))
    nodes_in_next = nodes_in_next + 1


def solve():
  global nodes_in_layer, move_counter, reached_goal, nodes_in_next
  row_moves.put(start_r)
  column_moves.put(start_c)
  visited[start_c][start_r] = True
#   print(visited, size_m)

  while row_moves.qsize() > 0:

    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop

    row_var = row_moves.get()
    column_var = column_moves.get()

    if size_m[row_var][column_var] == '$':
        print(row_var, column_var)
        reached_goal = True
        break
    movement(row_var,column_var)

            # Set the screen background
    screen.fill(BLACK)

    # Draw the grid
    for row in range(vari):
        for column in range(400):
            color = WHITE
            if grid[row][column] == 1:
                color = GREEN
            pygame.draw.rect(screen,
                            color,
                            [(MARGIN + WIDTH) * column + MARGIN,
                            (MARGIN + HEIGHT) * row + MARGIN,
                            WIDTH,
                            HEIGHT])

    # Limit to 60 frames per second
    clock.tick(120)

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

    nodes_in_layer = nodes_in_layer - 1 
        
    if nodes_in_layer == 0:
        nodes_in_layer = nodes_in_next
        nodes_in_next = 0
        move_counter = move_counter + 1
        print(move_counter,row_var,column_var)
  if reached_goal == True:
    return move_counter
  return -1, visited


# # Driver code

print(f"{str(convert(start_r))+str(convert(start_c))} is here")
print(f"{str(convert(goal_c))+str(convert(goal_r))} is here")


bread = solve()
print(bread)

stuffe = (search(state_dict, str(convert(start_r))+str(convert(start_c)), str(convert(goal_r))+str(convert(goal_c))))

print(stuffe)
grid = []
for row in range(vari):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(400):
        grid[row].append(0)  # Append a cell


for i in range(len(stuffe)):
    temp = [stuffe[i][x:x+3] for x in range(0, len(stuffe[i]), 3)]
    i_temp = (str(temp[0]).lstrip('0'))
    j_temp = (str(temp[1]).lstrip('0'))
    if i_temp == '':
        i_temp = 0
    if j_temp == '':
        j_temp = 0

    print(f"{i_temp, j_temp} is here")

    grid[int(i_temp)][int(j_temp)] = 1

pygame.quit()

# print(grid)

pygame.init()
 
# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [1200, 900]
screen = pygame.display.set_mode(WINDOW_SIZE)
 
# Set title of screen
pygame.display.set_caption("Array Backed Grid")
 
# Loop until the user clicks the close button.
done = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
 

while not done:

    screen.fill(BLACK)

    for row in range(vari):
        for column in range(400):
            color = WHITE
            if grid[row][column] == 1:
                color = GREEN
            pygame.draw.rect(screen,
                            color,
                            [(MARGIN + WIDTH) * column + MARGIN,
                            (MARGIN + HEIGHT) * row + MARGIN,
                            WIDTH,
                            HEIGHT])

    clock.tick(10)

    pygame.display.flip()

solve2()

pygame.quit()

# 8 Directions total
