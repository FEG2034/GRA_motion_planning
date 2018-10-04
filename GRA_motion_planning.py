import numpy
import math
import tkinter
import pygame
import os
import platform
import matplotlib.pyplot as plt

pygame.init()

#background_initialization------------------------------------------------------
scale = 400/128 # display / model
display_width = int(128*scale)
display_height = int(128*scale)
box_edge = numpy.array([[0,0], [0,127], [127,127], [127,0]])

#function_setup-----------------------------------------------------------------

#--convert planning vertices into display vertices (x, y) => (x*scale, H-y*scale)
#----matrix <- numpy.array (1D/2D)
def planning_to_display(matrix):
    return matrix.dot(numpy.array([[scale,0],[0,-scale]])) + numpy.array([0, display_height])

#--convert display vertices into planning vertices (x, y) => (x/scale, (H-y)/scale)
#----matrix <- numpy.matrix (1D/2D)
def display_to_planning(matrix):
    return (matrix - numpy.array([0, display_height])).dot(numpy.array([[1/scale,0],[0,-1/scale]]))

#--Translation and Rotation; matrix is 2D; xytheta = (dx, dy, theta in degrees)
#----matrix <- numpy.array (2D); xytheta <- list / numpy.array (1D)
def TR(matrix, xytheta):
    temp = numpy.ones(matrix.shape[0], dtype=int).reshape((matrix.shape[0],1))
    temp = numpy.concatenate((matrix, temp), axis=1)
    (dx, dy, theta) = (xytheta[0], xytheta[1], math.radians(xytheta[2]))
    temp = temp.dot(numpy.array([[math.cos(theta), math.sin(theta),0], [-math.sin(theta), math.cos(theta),0], [dx, dy, 1]]))
    return temp[:,:2]

#--standarize the angle into (-180, 180)
#----angle <- degrees
def angle_standarize(angle):
    if -180<=angle<=180:
        return angle
    elif angle>180:
        angle %= 360
        return angle-360
    elif angle<-180:
        angle %= 360
        return angle+360

#--angle_compute
#----vector1 <- list / numpy.array (1D); vector2 <- list / numpy.array (1D)
def get_angle(vector1, vector2):
    vector1 = vector1.reshape((2,))
    vector2 = vector2.reshape((2,))
    angle = math.degrees(math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0]))
    angle = angle_standarize(angle)
    return angle

#--assign the point in obstacles or not; point as matrix, edge as xytheta
#----vector_of_point <- list / numpy.array (1D); vector_of_edge <- list / numpy.array(1D)
def point_left(vector_of_point, vector_of_edge):
    temp = TR(numpy.array(vector_of_point).reshape((1,2)), [0,0,-math.degrees(math.atan2(vector_of_edge[1], vector_of_edge[0]))])
    return temp[0,1] >= 0

#--detect the intersection of two line segment
#----segment1, segment2 <- numpy.array (2D)
def intersect_segment(segment1, segment2):
    temp1 = numpy.array([segment1[0], segment1[0]])
    temp1 = (segment1[1,:] - segment1[0,:]).dot(numpy.array([[0,-1], [1,0]])).dot(segment2.T - temp1.T).prod()
    temp2 = numpy.array([segment2[0], segment2[0]])
    temp2 = (segment2[1,:] - segment2[0,:]).dot(numpy.array([[0,-1], [1,0]])).dot(segment1.T - temp2.T).prod()
    if temp1 < 0 and temp2 < 0:
        return True
    else:
        return False

#--detect the intersection of two polygon
#----polygon1, polygon2 <- numpy.array (2D)
def intersect_polygon(polygon1, polygon2):
    polygon1 = numpy.append(polygon1, polygon1[0].reshape(1,2), axis=0)
    polygon2 = numpy.append(polygon2, polygon2[0].reshape(1,2), axis=0)
    for i in range(len(polygon1)-1):
        for j in range(len(polygon2)-1):
            if intersect_segment(polygon1[i:i+2], polygon2[j:j+2]) == True:
                return True
            else:
                pass
    return False

#--Best First Search OPEN OPEN[i] = list of tuple(conf)
class BFS_OPEN:
    def __init__(self):
        self.OPEN = {i: [] for i in range(255)}

    def insert(self, conf, potential):
        self.OPEN[potential].insert(0, tuple(conf))

    def first(self): # return tuple or none
        FIRST_var = None
        for i in range(255):
            if self.OPEN[i] == []:
                pass
            elif self.OPEN[i]:
                FIRST_var = self.OPEN[i][0]
                self.OPEN[i] = self.OPEN[i][1:]
                return FIRST_var
        return FIRST_var

#--Best First Search Tree T[i] = list of [tuple(conf), (potential_of_source, index_of_source_in_T[potential])]
class BFS_T:
    def __init__(self):
        self.T = {i: [] for i in range(261)}
        self.path = []

    def insert_root(self, conf, potential):
        self.T[potential].insert(0, [tuple(conf)])

    def insert(self, conf, potential, source): # source = (potential, index of source in T[i])
        self.T[potential].insert(len(self.T[potential])+1 , [tuple(conf), source])

    def search(self, conf, potential):
        index = list(map(lambda x: x[0]==tuple(conf), self.T[potential]))
        if sum(index)==0:
            return False
        elif index[0]==1 :
            return (int(potential), 0)
        else:
            return (int(potential), (numpy.array(index)*numpy.arange(len(index))).sum() )

    def trace(self, where_in_T): # where_in_T = (potential, index in T[i]) of goal
        self.path.insert(0, self.T[where_in_T[0]] [where_in_T[1]] [0])
        if len(self.T[where_in_T[0]] [where_in_T[1]]) > 1:
            self.trace(self.T[where_in_T[0]] [where_in_T[1]] [1])

#--Best First Search NF1
def NF1():
    global display_objects

    U = {0: numpy.ones(128*128).reshape(128,128) * 255 } #initial potential = 255
    for obstacle in display_objects[2:]: #obstacle potential = 260
        for x in range(obstacle.planning_bounding_box[0,0], obstacle.planning_bounding_box[1,0]+1):
            for y in range(obstacle.planning_bounding_box[0,1], obstacle.planning_bounding_box[1,1]+1):
                if obstacle.point_inside([x,y]):
                    U[0][127-y,x] = 260
    for n in range(display_objects[1].n_control):
        U[n] = U[0].copy()

    for n in range(display_objects[1].n_control):
        U[n] = U[0].copy()
        U[n][127-display_objects[1].planning_control[n][1], display_objects[1].planning_control[n][0]] = 0
        L = {0: [numpy.array([0,0,0])], 1: []} #(dx,dy, delta theta) based on display_objects[1].planning_control[n,:]
        order = 0

        while L[0]: # L[0] is not empty --> return True
            L[1] = []
            for q in L[0]:

                for dx in (1,-1):
                    control = TR(display_objects[1].world_control[n].reshape((1,2)), (display_objects[1].planning_conf+q+(dx,0,0))).reshape((2,)).astype(int)
                    if 0<=control[0]<=127 and 0<=control[1]<=127: #bounded
                        if (U[n][127-control[1], control[0]] == 255):
                            U[n][127-control[1], control[0]] = order + 1
                            L[1] += [(q+(dx,0,0)).astype(int)]

                for dy in (1,-1):
                    control = TR(display_objects[1].world_control[n].reshape((1,2)), (display_objects[1].planning_conf+q+(0,dy,0))).reshape((2,)).astype(int)
                    if 0<=control[0]<=127 and 0<=control[1]<=127: #bounded
                        if (U[n][127-control[1], control[0]] == 255):
                            U[n][127-control[1], control[0]] = order + 1
                            L[1] += [(q+(0,dy,0)).astype(int)]

                for theta in (5,-5):
                    control = TR(display_objects[1].world_control[n].reshape((1,2)), (display_objects[1].planning_conf+q+(0,0,theta))).reshape((2,)).astype(int)
                    if 0<=control[0]<=127 and 0<=control[1]<=127: #bounded
                        if (U[n][127-control[1], control[0]] == 255):
                            U[n][127-control[1], control[0]] = order + 1
                            L[1] += [(q+(0,0,theta)).astype(int)]
            L[0] = L[1]
            order += 1

    display_objects[1].BFS_U = U
    print("NF1 success")

def NF1_show():
    global display_objects
    X = numpy.full((128,128), numpy.arange(128))
    Y = X.T
    Z = numpy.flip(display_objects[1].BFS_U[0].reshape(128,128), axis=0)
    plt.pcolormesh(X,Y,Z)
    plt.savefig("NF1")
    print(display_objects[1].BFS_U[0])

#--Best First Search BFS
def BFS():
    global display_objects

    U = display_objects[1].BFS_U
    def conf_potential(conf):
        potential = [tuple(TR(display_objects[0].world_control[i].reshape((1,2)), numpy.array(conf)).reshape((2,)).astype(int)) for i in range(display_objects[0].n_control)]
        for control in potential:
            if 0<=control[0]<=127 and 0<=control[1]<=127:
                pass
            else: # control point run out of bound
                return 260
        potential = [U[i][127-potential[i][1], potential[i][0]] for i in range(display_objects[0].n_control)]
        potential = int(sum(potential)/display_objects[0].n_control)
        return potential

    def collision(conf):
        polygon_robot = [TR(display_objects[0].world_polygon[i], numpy.array(conf)) for i in range(display_objects[0].n_polygon)]
        bounding_box = numpy.array(polygon_robot).flatten().astype(int)
        bounding_box = bounding_box.reshape((int(bounding_box.size/2), 2))
        bounding_box = numpy.array([bounding_box.min(0), bounding_box.max(0)])

        for obstacle in display_objects[2:]:
            #bounding_box intersect or not
            if (bounding_box[0,0] < obstacle.planning_bounding_box[0,0] < bounding_box[1,0] and \
            bounding_box[0,1] < obstacle.planning_bounding_box[0,1] < bounding_box[1,1]) or \
            (obstacle.planning_bounding_box[0,0] < bounding_box[0,0] < obstacle.planning_bounding_box[1,0] and \
            obstacle.planning_bounding_box[0,1] < bounding_box[0,1] < obstacle.planning_bounding_box[1,1]):
                #polygon intersect or not
                for polygon_obstacle_element in obstacle.planning_polygon:
                    for polygon_robot_element in polygon_robot:
                        if intersect_polygon(polygon_robot_element, polygon_obstacle_element):
                            print("robot and obstacle intersect")
                            return True
                        else:
                            pass
            else:
                pass
        return False

    OPEN = BFS_OPEN()
    T = BFS_T()

    delta = []
    for theta in (0,10,-10):
        for dx in (0,1,-1):
            for dy in (0,1,-1):
                    delta.insert(0, (dx,dy,theta))
    delta.remove((0,0,0))

    FIRST_conf = tuple(display_objects[0].planning_conf.astype(int)) #insure the conf in OPEN and T are integer
    potential = conf_potential(FIRST_conf)
    OPEN.insert(FIRST_conf, potential)
    T.insert_root(FIRST_conf, potential)

    SUCCESS = False
    while not SUCCESS:
        FIRST_conf = OPEN.first()
        if FIRST_conf == None:
            SUCCESS = False
            break
        source = T.search(FIRST_conf, conf_potential(FIRST_conf))
        print(FIRST_conf)
        print(conf_potential(FIRST_conf))
        for neighbor in delta:
            neighbor_conf = [FIRST_conf[i] + neighbor[i] for i in range(3)]
            neighbor_conf[-1] = angle_standarize(neighbor_conf[-1])
            neighbor_conf = tuple(neighbor_conf)
            neighbor_potential = conf_potential(neighbor_conf)
            visited = T.search(neighbor_conf, neighbor_potential)
            if neighbor_potential<260 and visited == False and -180 <= neighbor_conf[-1] <= 180:
                if collision(neighbor_conf) == False:
                    T.insert(neighbor_conf, neighbor_potential, source)
                    OPEN.insert(neighbor_conf, neighbor_potential)
            if neighbor_potential <= 1:
                SUCCESS = True
                T.trace(where_in_T = (neighbor_potential, -1))
                T.path.insert(len(T.path)+1, tuple(display_objects[1].planning_conf))
                break

    if SUCCESS:
        display_objects[1].BFS_path = T.path
        print("BFS success")
    else:
        print("BFS fail")

def BFS_show():
    global display_objects
    global polygon_buffer
    global polygon_buffer_var

    polygon_buffer = display_objects[1].BFS_path
    polygon_buffer = list(map(lambda x: [TR(display_objects[1].world_polygon[i], numpy.array(x)) for i in range(display_objects[1].n_polygon)], polygon_buffer))
    polygon_buffer = list(map(lambda x: [planning_to_display(segment) for segment in x], polygon_buffer))
    polygon_buffer_var = True

def CLEAR():
    global polygon_buffer_var
    polygon_buffer_var = False

#objects_setup------------------------------------------------------------------

#--objects
#----conf = configuration (x, y, angle) <- list (1D)
#----vertices <- list (3D) and counterclockwise ordered
#----control = control point (x, y) <- list (2D)
class robots:
    def __init__(self, conf, n_polygon, vertices, n_control, control):
        self.n_polygon = n_polygon
        self.n_control = n_control
        self.world_conf = numpy.array(conf).astype(float) #<----flexible
        self.world_polygon = [numpy.array(vertices[i]) for i in range(self.n_polygon)] #<----fixed
        self.world_control = numpy.array(control) #<----fixed

        self.planning_conf = self.world_conf #<----flexible
        self.planning_conf[-1] = angle_standarize(self.planning_conf[-1])
        self.planning_polygon = [TR(self.world_polygon[i], self.planning_conf) for i in range(self.n_polygon)]
        self.planning_control = TR(self.world_control, self.planning_conf).astype(int)

        self.planning_bounding_box = numpy.array(self.planning_polygon).flatten().astype(int)
        self.planning_bounding_box = self.planning_bounding_box.reshape((int(self.planning_bounding_box.size/2), 2))
        self.planning_bounding_box = numpy.array([self.planning_bounding_box.min(0), self.planning_bounding_box.max(0)])

        self.display_polygon = [planning_to_display(polygon) for polygon in self.planning_polygon]

        self.display_bounding_box = numpy.array(self.display_polygon).flatten().astype(int)
        self.display_bounding_box = self.display_bounding_box.reshape((int(self.display_bounding_box.size/2), 2))
        self.display_bounding_box = numpy.array([self.display_bounding_box.min(0), self.display_bounding_box.max(0)])

        self.BFS_U = {}
        self.BFS_path = []

    def mouse_inside(self, mouse_position): #mouse_position <- list (1D)
        if mouse_position[0] in range(self.display_bounding_box[0,0], self.display_bounding_box[1,0])\
        and mouse_position[1] in range(self.display_bounding_box[0,1], self.display_bounding_box[1,1]):
            return True
        else:
            return False

    def display_change(self, position1, position2, mouse_left=True): #position1,2 <- numpy.array (1D) on the display map
        if mouse_left == True: #move<----(dx, dy)
            move = display_to_planning(position2) - display_to_planning(position1)
            self.planning_conf[:2] += move

        else: #move<----theta
            position1 = (display_to_planning(position1) - self.planning_conf[:2])
            position2 = (display_to_planning(position2) - self.planning_conf[:2])
            move = get_angle(position1, position2)
            self.planning_conf[-1] += move
            self.planning_conf[-1] = angle_standarize(self.planning_conf[-1])

        self.planning_polygon = [TR(self.world_polygon[i], self.planning_conf) for i in range(self.n_polygon)]
        self.planning_control = TR(self.world_control, self.planning_conf).astype(int)
        self.planning_bounding_box = numpy.array(self.planning_polygon).flatten().astype(int)
        self.planning_bounding_box = self.planning_bounding_box.reshape((int(self.planning_bounding_box.size/2), 2))
        self.planning_bounding_box = numpy.array([self.planning_bounding_box.min(0), self.planning_bounding_box.max(0)])

        self.display_polygon = [planning_to_display(polygon) for polygon in self.planning_polygon]
        self.display_bounding_box = numpy.array(self.display_polygon).flatten().astype(int)
        self.display_bounding_box = self.display_bounding_box.reshape((int(self.display_bounding_box.size/2), 2))
        self.display_bounding_box = numpy.array([self.display_bounding_box.min(0), self.display_bounding_box.max(0)])

    def display_draw(self, game, color, width=0):
        for i in range(self.n_polygon):
            pygame.draw.polygon(game, color, numpy.array(self.display_polygon[i]), width)

class obstacles:
    def __init__(self, conf, n_polygon, vertices):
        self.n_polygon = n_polygon
        self.world_conf = numpy.array(conf).astype(float) #<----flexible
        self.world_polygon = [numpy.array(vertices[i]) for i in range(self.n_polygon)] #<----fixed

        self.planning_conf = self.world_conf
        self.planning_conf[-1] = angle_standarize(self.planning_conf[-1])
        self.planning_polygon = [TR(self.world_polygon[i], self.planning_conf) for i in range(self.n_polygon)]
        
        self.planning_bounding_box = numpy.array(self.planning_polygon).flatten().astype(int)
        self.planning_bounding_box = self.planning_bounding_box.reshape((int(self.planning_bounding_box.size/2), 2))
        self.planning_bounding_box = numpy.array([self.planning_bounding_box.min(0), self.planning_bounding_box.max(0)])

        self.display_polygon = [planning_to_display(polygon) for polygon in self.planning_polygon]

        self.display_bounding_box = numpy.array(self.display_polygon).flatten().astype(int)
        self.display_bounding_box = self.display_bounding_box.reshape((int(self.display_bounding_box.size/2), 2))
        self.display_bounding_box = numpy.array([self.display_bounding_box.min(0), self.display_bounding_box.max(0)])

    def mouse_inside(self, mouse_position): #mouse_position <- list (1D)
        if mouse_position[0] in range(self.display_bounding_box[0,0], self.display_bounding_box[1,0])\
        and mouse_position[1] in range(self.display_bounding_box[0,1], self.display_bounding_box[1,1]):
            return True
        else:
            return False

    def point_inside(self, point): #point <- list (1D)
        in_or_not = [True] * self.n_polygon
        for i in range(self.n_polygon):
            vector = (-1) * numpy.identity(self.planning_polygon[i].shape[0]) + numpy.eye(self.planning_polygon[i].shape[0], k=1)
            vector[-1,0] = 1
            vector = vector.dot(self.planning_polygon[i])
            points = numpy.full_like(self.planning_polygon[i], point) - self.planning_polygon[i]
            for j in range(self.planning_polygon[i].shape[0]):
                in_or_not[i] = in_or_not[i] & point_left(points[j,:], vector[j,:])
        return sum(in_or_not) > 0

    def display_change(self, position1, position2, mouse_left=True): #position1,2 <- numpy.array (1D) on the display map
        if mouse_left == True: #move<----(dx, dy)
            move = display_to_planning(position2) - display_to_planning(position1)
            self.planning_conf[:2] += move

        else: #move<----theta
            position1 = (display_to_planning(position1) - self.planning_conf[:2])
            position2 = (display_to_planning(position2) - self.planning_conf[:2])
            move = get_angle(position1, position2)
            self.planning_conf[-1] += move
            self.planning_conf[-1] = angle_standarize(self.planning_conf[-1])

        self.planning_polygon = [TR(self.world_polygon[i], self.planning_conf) for i in range(self.n_polygon)]
        self.planning_bounding_box = numpy.array(self.planning_polygon).flatten().astype(int)
        self.planning_bounding_box = self.planning_bounding_box.reshape((int(self.planning_bounding_box.size/2), 2))
        self.planning_bounding_box = numpy.array([self.planning_bounding_box.min(0), self.planning_bounding_box.max(0)])

        self.display_polygon = [planning_to_display(polygon) for polygon in self.planning_polygon]
        self.display_bounding_box = numpy.array(self.display_polygon).flatten().astype(int)
        self.display_bounding_box = self.display_bounding_box.reshape((int(self.display_bounding_box.size/2), 2))
        self.display_bounding_box = numpy.array([self.display_bounding_box.min(0), self.display_bounding_box.max(0)])

    def display_draw(self, game, color, width=0):
        for i in range(self.n_polygon):
            pygame.draw.polygon(game, color, numpy.array(self.display_polygon[i]), width)

#data_input--------------------------------------------------------------------

n_robots = 2
n_obstacles = 3

robots0_recent = robots(conf = [64, 64, 90], n_polygon = 2, \
                vertices = [[[15,4], [-3,4], [-3,-4], [15,-4]], [[7,4], [11,4], [11,8], [7,8]]], \
                n_control = 2, control = [[12,10], [-2,0]])

robots0_goal = robots(conf = [80,80,0], n_polygon = 2, \
                vertices = [[[15,4], [-3,4], [-3,-4], [15,-4]], [[7,4], [11,4], [11,8], [7,8]]], \
                n_control = 2, control = [[12,10], [-2,0]])

robots1_recent = robots(conf = [20, 20, 90], n_polygon = 1, \
                vertices = [[[-5,-5], [5,-5], [0,5]]], \
                n_control = 2, control = [[0,-4], [0,4]])

robots1_goal = robots(conf = [30,100,0], n_polygon = 1, \
                vertices = [[[-5,-5], [5,-5], [0,5]]], \
                n_control = 2, control = [[0,-4], [0,4]])

obstacles0 = obstacles(conf = [40, 30, 300], n_polygon = 1, \
            vertices = [[[9,-7], [13,0], [9,6], [-11,6], [-14,0], [-11,-7]]])

obstacles1 = obstacles(conf = [90, 51, 3.75], n_polygon = 1, \
            vertices = [[[17,6], [-17,6], [-17,-7], [25,-7]]])

obstacles2 = obstacles(conf = [56,30,90], n_polygon = 2, \
            vertices = [[[9,-3], [9,6], [-11,6], [-11,-3]],[[1,6], [1,10], [-2,10], [-2,6]]])

#global variable
display_objects = [robots0_recent, robots0_goal, obstacles0, obstacles1, obstacles2]
polygon_buffer = []
polygon_buffer_var = False

#tkinter_initialization--------------------------------------------------------
root = tkinter.Tk()
root.title("Motion Planning - Control Board")

##Label, Checkbotton, Botton
option_var = tkinter.StringVar(root)
option_var.set("robot #0")
Option = tkinter.OptionMenu(root, option_var, "robot #0", "robot #1")

Button_NF1 = tkinter.Button(root, text="Run NF1", command = NF1, fg='white', bg='black')
Button_NF1_show = tkinter.Button(root, text="Show NF1", command = NF1_show, fg='yellow', bg='black')
Button_BFS = tkinter.Button(root, text="Run BFS", command = BFS, fg='white', bg='black')
Button_BFS_show = tkinter.Button(root, text="Show BFS", command = BFS_show, fg='yellow', bg='black')
Button_clear = tkinter.Button(root, text="CLEAR", command = CLEAR, fg='yellow', bg='blue')

Option.pack(fill=tkinter.BOTH, expand=1)
Button_NF1.pack(fill=tkinter.BOTH, expand=1)
Button_NF1_show.pack(fill=tkinter.BOTH, expand=1)
Button_BFS.pack(fill=tkinter.BOTH, expand=1)
Button_BFS_show.pack(fill=tkinter.BOTH, expand=1)
Button_clear.pack(fill=tkinter.BOTH, expand=1)

##embed pygame into tkinter
os.environ['SDL_WINDOWID'] = str(root.winfo_id())
if platform.system() == 'Darwin': # <- for Mac OS
    os.environ['SDL_VIDEODRIVER'] = 'Quartz' 
elif platform.system() == 'Windows': # <- for Windows
    os.environ['SDL_VIDEODRIVER'] = 'windib'

root.update()

#pygame_main-------------------------------------------------------------------
black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
blue = (0,0,139)

gameDisplay = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption('Motion Planning - Displaying board')
clock = pygame.time.Clock()

def main():
    global display_objects
    global polygon_buffer
    global polygon_buffer_var

    while True:

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            root.quit()
            quit()

        if option_var.get() == "robot #0":
            display_objects = [robots0_recent, robots0_goal, obstacles0, obstacles1, obstacles2]
        else:
            display_objects = [robots1_recent, robots1_goal, obstacles0, obstacles1, obstacles2]

#mouse_control-----------------------------------------------------------------
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: #mouse_left
            mouse_position1 = numpy.array(pygame.mouse.get_pos())
            
            while event.type != pygame.MOUSEBUTTONUP:
                event = pygame.event.poll()
                mouse_position2 = pygame.mouse.get_pos()

            mouse_position2 = numpy.array(mouse_position2)
            
            for object in display_objects:
                if object.mouse_inside(mouse_position1):
                    object.display_change(mouse_position1, mouse_position2, mouse_left=True)
                else:
                    pass

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3: #mouse_right
            mouse_position1 = numpy.array(pygame.mouse.get_pos())

            while event.type != pygame.MOUSEBUTTONUP:
                event = pygame.event.poll()
                mouse_position2 = pygame.mouse.get_pos()

            mouse_position2 = numpy.array(mouse_position2)
            
            for object in display_objects:
                if object.mouse_inside(mouse_position1):
                    object.display_change(mouse_position1, mouse_position2, mouse_left=False)
                else:
                    pass

#display section---------------------------------------------------------------
        gameDisplay.fill(white)

        pygame.draw.polygon(gameDisplay, blue, numpy.array(planning_to_display(box_edge)), 1)

        display_objects[0].display_draw(gameDisplay, black) #draw robot_recent
        display_objects[1].display_draw(gameDisplay, black, width=2) #draw robot_goal

        for object in display_objects[2:]: #draw obstacle
            object.display_draw(gameDisplay, red)

        if polygon_buffer_var == True:
            for segment in polygon_buffer:
                for i in range(len(segment)):
                    pygame.draw.polygon(gameDisplay, blue, numpy.array(segment[i]), 1)

        pygame.display.update()
        root.update()
        clock.tick(30)

main()