import cv2
import numpy as np
import heapq

image =cv2.imread('maze.png')

# Function to check if a point is valid (not in an obstacle)
def is_valid_point(point, maze):
    x,y= point
    rows, cols = maze.shape
    return 0 <= y< rows and 0 <=x< cols and maze[y,x] == 255

# Function to check if an edge between two points is valid (not intersecting with obstacles)
def is_valid_edge(p1, p2, maze):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    steps = max(abs(dx), abs(dy))
    if steps == 0:
        return False
    for i in range(1, steps + 1):
        x = int(x1 + float(i) * dx / steps)
        y = int(y1 + float(i) * dy / steps)
        if not is_valid_point((x, y), maze):
            return False
    return True

# Function to sample random points using Gaussian distribution
def sampling(maze):
    while True:
        x = int(np.random.randint(0, maze.shape[1] - 1))
        y = int(np.random.randint(0, maze.shape[0] - 1))
        point = (x, y)
        if is_valid_point(point, maze):
            return point

# Function to perform PRM algorithm
def prm(maze, start, goal, num_nodes):
    roadmap = [start, goal]
    while len(roadmap) < num_nodes:
        random_point = sampling(maze)
        if is_valid_point(random_point, maze):
            roadmap.append(random_point)
    return roadmap

# Function to perform Dijkstra's algorithm
def dijkstra(graph,maze, start, goal):
    pq = [(0, start)]
    heapq.heapify(pq)
    visited = set()
    parent = {}
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        if current_node == goal:
            break
        if current_node in visited:
            continue
        visited.add(current_node)
        for neighbor, weight in graph[current_node]:
            if neighbor in visited:
                continue
            new_dist = current_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                parent[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))
    path = []
    while goal != start:
        path.append(goal)
        if goal not in parent:
            print("No path found from start to goal.")
            return []
        goal = parent[goal]
    path.append(start)
    return path[::-1]

# Function to plot the maze, roadmap, and shortest path
def plot_maze(img, start, goal, roadmap, path):
    for point in roadmap:
        img = cv2.circle(img, tuple(point), 1, (255, 255, 0), -1)
    for i in range(len(path)-1):
        cv2.line(img, tuple(path[i]), tuple(path[i+1]), (255, 0, 0), 2)
    img = cv2.circle(img, tuple(start), 3, (0, 255, 0), -1)
    img = cv2.circle(img, tuple(goal), 3, (0, 0, 255), -1)
    cv2.imshow('maze', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def implement_prm(maze,start,goal,num_nodes):
    # Perform PRM algorithm
    roadmap = prm(maze, start, goal, num_nodes)

    # Construct graph from the roadmap for Dijkstra's algorithm
    graph = {node: [] for node in roadmap}
    for i, node in enumerate(roadmap):
        for j in range(i + 1, len(roadmap)):
            neighbor = roadmap[j]
            if is_valid_edge(node, neighbor, maze):
                dist = np.linalg.norm(np.array(node) - np.array(neighbor))
                graph[node].append((neighbor, dist))
                graph[neighbor].append((node, dist))

    # Find shortest path using Dijkstra's algorithm
    shortest_path = dijkstra(graph, maze, start, goal)

    # Plot the maze, roadmap, and shortest path
    plot_maze(image, start, goal, roadmap, shortest_path)

# Main function
def main():
    # Read the maze image and convert it to grayscale
    img=image.copy()
    img=cv2.line(img,(11,340),(128,340),(0,0,0),10)
    img=cv2.line(img,(135,20),(190,20),(0,0,0),10)
    img=cv2.line(img,(445,280),(445,330),(0,0,0),10)
    maze_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    ret,maze=cv2.threshold(maze_gray,243,255,cv2.THRESH_BINARY)

    # Define the start and goal points
    start_easy=(30,330)
    end_easy=(110,330)
    start_hard=(160,30)
    end_hard=(435,305)
    # Define the maximum number of nodes for the PRM algorithm
    num_nodes = 300

    for i in range (0,2):
        if(i==0):
            start=start_easy
            goal=end_easy
            implement_prm(maze,start,goal,num_nodes)
        else:
            start=start_hard
            goal=end_hard
            implement_prm(maze,start,goal,num_nodes)

if __name__ == "__main__":
    main()
