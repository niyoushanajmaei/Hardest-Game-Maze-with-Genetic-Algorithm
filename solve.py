import hardest_game
import copy
import random
import math
import time
import matplotlib.pyplot as plt

'''
Mutation Based (Asexual) genetic algorithm is used
For each generation, a parent is selected from the population considering its fitness 
The child of the parent will be a mutated version of the parent
The size of the population is 400, and the initial number of moves is 5
every 5 generations, each path is expanded by 10 random moves
'''

'''
The fitness function maximizes if the path leads to obtaining the nearest goal and moving
towards the end region when all points have been obtained
Fitness of paths that lead to death by blue dots is 0.81 times the ones that don't
Mutation probability for the last 5 moves before death by blue dot is 0.5, otherwise 0.01
When a random move is done, the probability of getting a move towards the goal (not considering walls)
is 2 times the probability of getting other moves (in order to speed up convergence).
'''


class Path:

    def __init__(self):
        self.length = 0  # the length of the path
        self.arr = []  # a list of moves of the path chosen from {w,a,s,d}
        self.won = False  # Whether the path wins the game
        self.points_obtained = 0  # The number of points obtained by this path
        self.all_points = False  # Whether the path collects all points
        self.initial_length = 5 # start from paths with length 100
        self.lost = False  # Whether the player has been eaten by the blue point
        self.step_lost = -1 # The step at which the red rect lost
        self.out_of_start = False # Whether the red rect has gotten out of the starting region
        self.fitness = 0 # fitness which will be maximized
        self.player_coords = [-1,-1] # Last coordinates of the red_rect
        self.goal_coords = [-1,-1] # Coordinates of the goal, could be the nearest goal or end region

    # a function to calculate the fitness of the path and update some attributes
    def get_fitness(self, player, points, game_params):
        fitness = 0  # minimum value of fitness
        goals_coords = game_params[0] # coordinates of yellow circles
        end_coords = game_params[2] # coordinates of the end region
        start_coords = game_params[3] # coordinates of the starting region
        window_coords = game_params[4] # coordinates of the middle of the slit of the starting region
        end_window_coords = game_params[5] # coordinates of the middle of the slit of the end region
        player_coords = [player[0].x, player[0].y] # last coordinates of the red rect
        self.won = player[2] # whether the red rect won the game
        self.points_obtained = len([b for b in points if b]) # the number of yellow circles that were obtained
        if player[1] != -1:
            self.lost = True
            self.step_lost = player[1] # if the player lost, get the step at which it happened
        distance_to_end = self.distance_to_window(end_window_coords, player_coords) # distance from end_window
        distance_to_goal, index = self.nearest_goal(points, player_coords, goals_coords) # distance from the nearest yellow goal
        if self.points_obtained == len(goals_coords):
            self.all_points = True # all yellow goals have been obtained
        if self.nearest_distance(start_coords, player_coords) > 0:  # if player got out of start
            self.out_of_start = True
        if self.points_obtained and self.out_of_start:
            g_coords = end_coords  # goal is end if all points were obtained and the red rect got out of the stating region
        elif self.points_obtained:
            g_coords = window_coords # if all goals were obtained but hasn't escaped from start, move towards slit
        else:
            g_coords = goals_coords[index] # otherwise, move towards the nearest yello goal
        self.set_player_goal(player_coords,g_coords)

        # setting distance to current goal
        if self.all_points:
            if not self.out_of_start:
                distance = self.distance_to_window(window_coords, player_coords)
            else:
                distance = distance_to_end
        else:
            distance = distance_to_goal

        # calculate the fitness
        if self.out_of_start :
            fitness = (1 + (2 ** (20 * (1 / 5 + 4 * self.points_obtained)))) / (distance) ** 4
        else:
            fitness = (1 + (2 ** (20 * self.points_obtained))) / (distance) ** 4
        if self.lost:
            fitness = 0.9 * fitness  # fitness 0 for those who have lost
        self.fitness = fitness * fitness

        return self.fitness

    # a function to set the coordinates of the player and current goal
    def set_player_goal(self,player_coords,g_coords):
        self.player_coords = copy.deepcopy(player_coords)
        self.goal_coords = copy.deepcopy(g_coords)

    # checks whether a point lies inside a rectangle
    @staticmethod
    def in_rectangle(start_coords, goals_coords):
        res = False
        for goal in goals_coords:
            if start_coords[0] <= goal[0] <= start_coords[2] \
                    and start_coords[1] <= goal[1] <= start_coords[3]:
                res = True
        return res

    # gets the coordinates and the index of the nearest yellow goal
    @staticmethod
    def nearest_goal(points, player_coords, goals_coords):
        distance = 100000
        index = -1
        for i in range(len(goals_coords)):
            if not points[i]:
                d = math.sqrt(
                    (player_coords[0] - goals_coords[i][0]) ** 2 + (player_coords[1] - goals_coords[i][1]) ** 2)
                if d < distance:
                    distance = d
                    index = i
        return distance, index

    # gets the distance to the slit on the start region
    @staticmethod
    def distance_to_window(window_coords, player_coords):
        return math.sqrt((player_coords[0] - window_coords[0]) ** 2 + (player_coords[1] - window_coords[1]) ** 2)

    # returns the nearest distance of a point from a rectangle if the point is outside of the rectangle
    @staticmethod
    def nearest_distance(rect, point):
        dx = max(rect[0] - point[0], 0, point[0] - rect[2]);
        dy = max(rect[1] - point[1], 0, point[1] - rect[3]);
        distance = math.sqrt(dx * dx + dy * dy)
        return distance

    # a function to change each element of "self.arr" to a random move with probability "prob"
    def mutate(self, iter):
        prob = 0.01
        for i in range(0, self.length):
            rand = random.uniform(0, 1)
            if self.lost and self.step_lost - i < 5:
                rand = rand * prob * 2 # if the agent lost, last 5 moves have probability 0.5
            if rand < prob:
                self.arr[i] = self.get_random_move()
        return self

    #  returns a random move, choosing a move toward the current goal is more probable
    def get_random_move(self):
        moves = ['w', 'a', 's', 'd']
        if self.player_coords[1] < self.goal_coords[1]-10:
            moves.append('s')
        if self.player_coords[1] - 10 > self.goal_coords[1]:
            moves.append('w')
        if self.player_coords[0] < self.goal_coords[0] - 10:
            moves.append('d')
        if self.player_coords[0] - 10 > self.goal_coords[0]:
            moves.append('a')
        rand = random.randint(0, len(moves) - 1)
        return moves[rand]

    # add a random move to "self.path"
    def add_random_move(self):
        self.arr.append(self.get_random_move())
        self.length += 1

    # add defined move to "self.path"
    def add_move(self, i):
        moves = ['w', 'a', 's', 'd']
        self.arr.append(moves[i])
        self.length += 1

    # This path wins
    def set_won(self):
        self.won = True

    # increment the number of obtained points
    def increment_points_obtained(self):
        self.points_obtained += 1

    # resetting the attributes (for children)
    def reset(self):
        self.points_obtained = 0  # The number of points obtained by this path
        self.all_points = False  # Whether the path collects all points
        self.lost = False  # Whether the player has been eaten by the blue point
        self.step_lost = -1
        self.out_of_start = False
        self.fitness = 0


class Population:
    def __init__(self):
        self.size = 400 # population size
        self.paths = []  # a list of all the paths
        self.path_length = 0  # the length of each path
        self.iter = 1  # iter-th generation
        self.initiate() # randomly choose initial paths
        self.best_fitness = 0  # The best fitness of the population
        self.cum_fitness = None  # a dictionary where key values are indices and values are cumulative fitness
        self.fitness_arr = None  # an array of fitness values
        self.won = False # whether at least one path of the population won

    # get 400 random paths with 5 moves
    def initiate(self):
        # generating 400 random numbers of length 5 in base 5
        # equivalently, between 0 and 5**5 - 1
        for i in range(self.size):
            rand = random.randint(0, (4 ** 5) - 1)
            temp = Path()
            for i in range(5):
                temp.add_move(rand % 4)
                rand = int(rand / 4)
            self.paths.append(copy.deepcopy(temp))
        self.path_length = self.paths[0].length

    # a function to increment path_length by 10 every 5 generations
    def increment_path_length(self):
        by = 10
        every = 5
        if self.iter % every == 0:
            self.path_length += by
            for path in self.paths:
                for i in range(by):
                    path.add_random_move()

    # make the next generation and replace the population with the new one
    #returns the fitness of self.best_fitness
    def next_generation(self):
        self.calc_cum_fitness()
        new_paths = []
        for i in range(self.size):
            parent = self.get_parent()
            child = copy.deepcopy(parent).mutate(self.iter)
            # reset attributes
            child.reset()
            new_paths.append(child)
        self.paths = copy.deepcopy(new_paths)
        self.iter += 1
        self.increment_path_length()
        if not self.won:
            self.update_won()
        return self.best_fitness

    # checks whether at least one path of the population has won
    def update_won(self):
        for path in self.paths:
            if path.won:
                self.won = True
                break

    # get a parent considering the fitness of each path
    # a random number in [0,max(cum_fitness)] is generated
    # the parent is the first path that has cumulative fitness more than the generated number
    def get_parent(self):
        potential = list()
        rand = random.uniform(min(self.cum_fitness.values()), max(self.cum_fitness.values()))
        for (i, cumulative) in self.cum_fitness.items():
            if cumulative >= rand and (len(potential)==0 or cumulative == self.cum_fitness[potential[0]]):
                potential.append(i)
        rand = random.randint(0,len(potential)-1)
        return self.paths[potential[rand]]

    # returns the path with largest fitness
    def get_best(self):
        return self.get_best

    # sum of fitness values are computed, fitness values are normalized
    # the cumulative normalized fitness value are computed
    # The function also sets best path of the
    def calc_cum_fitness(self):
        cum_fitness = {}
        for fitness in self.fitness_arr:
            if len(cum_fitness) == 0:
                cum_fitness[0] = fitness
            else:
                cum_fitness[len(cum_fitness)] = fitness + cum_fitness[len(cum_fitness) - 1]
        self.cum_fitness = cum_fitness

    # calculates fitness for all paths of the generation
    def calc_population_fitness(self, players, goal_matrix, game_params):
        fitness_arr = []
        goal_matrix = [[goal_matrix[j][i] for j in range(len(goal_matrix))] for i in range(len(goal_matrix[0]))]
        for (i, path) in enumerate(self.paths):
            fitness_arr.append(path.get_fitness(players[i], goal_matrix[i], game_params))
        self.fitness_arr = fitness_arr
        self.best_fitness = max(self.fitness_arr)


def play_game_AI(str, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AI').run_AI_moves_graphic(moves=str)
    return game


def simulate(str, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AI').run_AI_moves_no_graphic(moves=str)
    return game


def run_whole_generation(list_of_strs, N, map_name='map1.txt'):
    game = hardest_game.Game(map_name=map_name, game_type='AIS').run_generation(list_of_moves=list_of_strs, move_len=N)
    return game


def play_human_mode(map_name='map1.txt'):
    hardest_game.Game(map_name=map_name, game_type='player').run_player_mode()

# returns the coordinates of the slit of the starting region
def get_window_coords(vlines, start_coords):
    x_o = start_coords[2]
    y_min = start_coords[1]
    y_max = start_coords[3]
    found = False
    for l in vlines:
        if l.x1 == x_o and l.y1 == y_min:
            found = True
            break
    if not found:
        return [x_o, (y_max + y_min) / 2]
    return [x_o, (y_max - l.y2) / 2 + l.y2]

# returns the coordinates of the slit of the end region
def get_end_window_coords(vlines, end_coords):
    x_o = end_coords[0]
    y_min = end_coords[1]
    y_max = end_coords[3]
    found = False
    for l in vlines:
        if l.x1 == x_o and l.y2 == y_max:
            found = True
            break
    if not found:
        return [x_o, (y_max + y_min) / 2]
    return [x_o, (l.y1 - y_min) / 2 + y_min]


def run_genetic_algorithm(game_params, map_name,start_time):
    population = Population()
    done = False
    best_arr = list()
    while not done:
        if population.iter % 10 == 0:
            print(f"Generation: {population.iter}")
        list_of_paths = list(map(lambda path: path.arr, population.paths))
        game = run_whole_generation(list_of_paths, population.path_length, map_name=map_name)
        population.calc_population_fitness(game.players, game.goal_player, game_params)
        best_arr.append(population.next_generation())
        if population.won:
            done = True
    print(f"Generation {population.iter} won after {(time.time()-start_time)/60} minutes !!!")
    return best_arr

def plot_fitness(best_arr):
    plt.semilogy(list(range(1,len(best_arr)+1)),best_arr,'o')
    plt.show()
    return


if __name__ == "__main__":
    #play_human_mode()
    map_name = "map1.txt"
    start_time = time.time()
    game = simulate("", map_name=map_name)  # running with an empty string to get the position of goals and and
    goals_coords = [[goal[0].x, goal[0].y] for goal in game.goals]
    player_init_coords = [game.player_x, game.player_y]
    #init_space_coords holds the coordinates of top left and bottom right corners of the end space
    end_coords = [game.end.x, game.end.y, game.end.x + game.end.w, game.end.y + game.end.h]
    start_coords = [game.start.x, game.start.y, game.start.x + game.start.w, game.start.y + game.start.h]
    window_coords = get_window_coords(game.Vlines, start_coords)
    end_window_coords = get_end_window_coords(game.Vlines, end_coords)
    game_params = [goals_coords, player_init_coords, end_coords, start_coords, window_coords, end_window_coords]
    best_arr = run_genetic_algorithm(game_params, map_name,start_time)
    plot_fitness(best_arr)
