from collections import deque
import heapq

def dfs(graph, start, goal, path=None, visited=None):
    if path is None:
        path = [start]
    if visited is None:
        visited = set([start])
    if start == goal:
        return path
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            visited.add(neighbor)
            res = dfs(graph, neighbor, goal, path + [neighbor], visited)
            if res:
                return res
            
    
    return None




def dls(graph, start, goal, depth ,path=None, visited=None):
    if depth == 0 and start == goal:
        return path
    if path is None:
        path = [start]
    if visited is None:
        visited = set([start])

    if depth > 0:
        if start == goal:
            return path
    
        for neighbor in graph[start]:
            if neighbor not in visited:
                visited.add(neighbor)
                res = dls(graph, neighbor, goal,depth - 1 ,path + [neighbor], visited)
                if res:
                    return res
                
        
    return None

def bfs(graph, start, goal):
    queue = deque([start])
    visited= set([start])

    while queue:
        path = queue.pop()
        node = path[-1]
        
        if node == goal:
            return path
        
        for n in graph[node]:
            if n not in visited:
                visited.add(n)
                new_path = list(path)
                new_path.append(n)
                queue.append(new_path)

    return None


def ucs_path(graph, start, goal):
    queue = [(0, [start])]
    visited = set()

    while queue:
        cost, path = heapq.heappop(queue)
        node = path[-1]
        if node == goal:
            return path, cost
        
        if node not in visited:
            visited.add(node)
            for n, w in graph[node]:
                if n not in visited:
                    new_path = path + [n]
                    heapq.heappush(queue, (cost + w, new_path))
    return None


def iddfs(graph, max_depth, start, goal):
    for i in range(max_depth):
        res = dls(graph, start, goal, i)
        if res:
            return res
    return None




def A_star(graph, start, goal, heuristic):
    queue = [( 0 + heuristic[start], 0, [start])]
    visited = set()

    while queue:
        f,cost, path = heapq.heappop(queue)
        node = path[-1]
        if node == goal:
            return path, cost
        
        if node not in visited:
            visited.add(node)
            for n, w in graph[node]:
                if n not in visited:
                    new_path = path + [n]
                    heapq.heappush(queue, (cost + w + heuristic[n], cost + w,new_path))
    return None




def hill_climbing_path(graph, start, goal, h):
    current = start
    path = [current]  # Initialize path with the start node

    while True:
        neighbors = graph[current]
        if not neighbors:
            return None  # No path if we are stuck
        
        next_node = min(neighbors, key=lambda x: h[x[0]])  # Find neighbor with lowest heuristic
        if h[next_node[0]] >= h[current]:
            return path if current == goal else None  # Return the path if goal is reached
        
        current = next_node[0]
        path.append(current)  # Append the best neighbor to the path
0



graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}



graph_weighted = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 3), ('E', 2)],
    'C': [('F', 5)],
    'D': [],
    'E': [('F', 1)],
    'F': []
}


# heuristic = {
#     'A': 7, 'B': 6, 'C': 2, 'D': 1, 'E': 0, 'F': 0
# }


graph_c = {
    'S': [('A', 1), ('B', 5)],  # S to A has low cost, S to B is higher cost
    'A': [('S', 1), ('C', 2), ('D', 8)],  # A to D has high cost
    'B': [('S', 5), ('D', 2)],
    'C': [('A', 2), ('G', 8)],  # Path C to G is longer
    'D': [('A', 8), ('B', 2), ('G', 3)],  # Path D to G is optimal
    'G': [('C', 8), ('D', 3)]  # Goal node
}
heuristic_c  = {
    'S': 7,
    'A': 6,
    'B': 2,  # B is closer to G
    'C': 8,
    'D': 1,  # D is very close to G
    'G': 0   # Goal node has heuristic 0
}

tree = {'S': [['A', 1], ['B', 5], ['C', 8]],
        'A': [['S', 1], ['D', 3], ['E', 7], ['G', 9]],
        'B': [['S', 5], ['G', 4]],
        'C': [['S', 8], ['G', 5]],
        'D': [['A', 3]],
        'E': [['A', 7]]}

"""2. Initialize Heuristic values of each node""" 
heuristic = {'S': 8, 'A': 8, 'B': 4, 'C': 3, 'D': 5000, 'E': 5000, 'G': 0}



graph_m = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 1), ('E', 3)],
    'C': [('F', 5)],
    'D': [('G', 2)],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}


print(ucs_path(graph_m,'A', 'G'))
