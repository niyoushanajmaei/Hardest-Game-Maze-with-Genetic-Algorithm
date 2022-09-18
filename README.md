# Hardest-Game-Maze-with-Genetic-Algorithm
## Solving the Hardest Game Mazes using Genetic Algorithm
Genetic Algorithm is used to successfully solve three maps of the hardest game maze series. 
Videos of a sample run of the maps can be found in the maps folder, and the plots of the evolution of fitness funtions through generations can be found in folder result plots.

### The Game 
The goal of the game is to capture all of the yellow dots first, and then escape the maze by entering the green area. The player must avoid being eaten by the blue dots throughout the game. In case the player is eaten, the game is over.

### Fitness Function
Current goal:
  - If all yellow goals are obtained and the red rectangle is outside the green start region
    - Current goal: End region
  - If all yellow goals are obtained and the red rectangle is inside the green start region
    - Current goal: The middle point of an opening in the starting green region
  - If yellow points remain
    - Current goal: The nearest yellow point
    
Distance: Euclidean distance of player from current goal

Points: number of yellow points obtained 


<img width="463" alt="image" src="https://user-images.githubusercontent.com/20246676/190924929-c7810c8b-abc6-4706-b107-b2cba9e260dd.png">


### Results
The following results were obtained during a sample run of each map:

- Map 2: Generation 66 wins after 4.6 minutes
- Map 1: Generation 91 wins after 8.2 minutes
- Map 3: Generation 308 after 86.1 minutes
