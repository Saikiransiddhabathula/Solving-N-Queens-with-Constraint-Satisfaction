import time
import random
import math
import constraint
from collections import deque
from constraint import Problem, AllDifferentConstraint
from itertools import count

# Forward Checking
def is_safe_FC(board, row, col, n, slash_diag, backslash_diag, col_set):
    return col not in col_set and \
           slash_diag[row - col] == 0 and \
           backslash_diag[row + col] == 0

def update_diagonals(row, col, n, slash_diag, backslash_diag, value):
    slash_diag[row - col] += value
    backslash_diag[row + col] += value

#@profile
def forward_checking_nqueens(n):
    start_time = time.time()

    solutions = 0
    col_set = set()
    slash_diag = [0] * (2 * n - 1)
    backslash_diag = [0] * (2 * n - 1)

    def solve_util(row):
        nonlocal solutions

        if row == n:
            solutions += 1
            return

        for col in range(n):
            if is_safe_FC(board, row, col, n, slash_diag, backslash_diag, col_set):
                board[row] = col
                col_set.add(col)
                update_diagonals(row, col, n, slash_diag, backslash_diag, 1)

                solve_util(row + 1)

                col_set.remove(col)
                update_diagonals(row, col, n, slash_diag, backslash_diag, -1)

    for i in range(n):
        board = [-1] * n
        board[0] = i
        col_set.add(i)
        update_diagonals(0, i, n, slash_diag, backslash_diag, 1)
        solve_util(1)
        col_set.remove(i)
        update_diagonals(0, i, n, slash_diag, backslash_diag, -1)

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"\nNumber of solutions for {n}-Queens problem using Forward Checking: {solutions}")
    print(f"Time taken to solve: {time_taken:.6f} seconds")

# ARC Consistency
def is_consistent(board, row, col):
    for i in range(row):
        if board[i] == col or \
           board[i] - i == col - row or \
           board[i] + i == col + row:
            return False
    return True

def arc_consistency(board, row, n):
    for i in range(row + 1, n):
        for j in range(n):
            if board[i] == j or \
               board[i] - i == j - row or \
               board[i] + i == j + row:
                board[i] = -1

#@profile
def solve_n_queens_arc_consistency(n):
    solutions = 0
    start_time = time.time()

    def backtrack(board, row, n):
        nonlocal solutions

        if row == n:
            solutions += 1
            return

        for col in range(n):
            if is_consistent(board, row, col):
                board[row] = col
                arc_consistency(board, row, n)
                backtrack(board, row + 1, n)
                board[row] = -1

    board = [-1] * n
    backtrack(board, 0, n)

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"\nNumber of solutions for {n}-Queens problem using ARC Consistency: {solutions}")
    print(f"Time taken to solve the problem: {time_taken:.6f} seconds")

# Minimum Remaining Values (MRV)
def is_consistent(board, row, col):
    for i in range(row):
        if board[i] == col or \
           board[i] - i == col - row or \
           board[i] + i == col + row:
            return False
    return True

def mrv_heuristic(board, row, n):
    candidates = []
    for col in range(n):
        if is_consistent(board, row, col):
            count = sum(1 for i in range(row + 1, n) if is_consistent(board, i, col))
            candidates.append((col, count))
    return sorted(candidates, key=lambda x: x[1])

#@profile
def solve_n_queens_mrv(n):
    solutions = 0
    start_time = time.time()

    def backtrack(board, row, n):
        nonlocal solutions

        if row == n:
            solutions += 1
            return

        candidates = mrv_heuristic(board, row, n)

        for col, _ in candidates:
            board[row] = col
            if is_consistent(board, row, col):
                backtrack(board, row + 1, n)
            board[row] = -1

    board = [-1] * n
    backtrack(board, 0, n)

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"\nNumber of solutions for {n}-Queens problem using Minimum Remaining Values (MRV): {solutions}")
    print(f"Time taken to solve the problem: {time_taken:.6f} seconds")

    
# Backtracking Algorithm
def is_safe(board, row, col, n):
    # Check if there is no queen in the same column, left diagonal, and right diagonal
    for i in range(row):
        if board[i] == col or \
           board[i] - i == col - row or \
           board[i] + i == col + row:
            return False
    return True

#@profile        
def solve_n_queens_backtracking(n):
    start_time = time.time()
    board = [-1] * n
    solutions = 0

    def solve_util(row):
        nonlocal solutions

        if row == n:
            solutions += 1
            return

        for col in range(n):
            if is_safe(board, row, col, n):
                board[row] = col
                solve_util(row + 1)

    solve_util(0)
    end_time = time.time()
    time_taken = end_time - start_time

    print("Solution using Backtracking Algorithm:")
    print(f"\nNumber of solutions for {n}-Queens using Backtracking Algorithm: {solutions}")
    print(f"Time taken to solve: {time_taken:.6f} seconds")

# Least Constraining Value Heuristic Algorithm
def is_safe(board, row, col, n):
    for i in range(row):
        if board[i] == col or \
           board[i] - i == col - row or \
           board[i] + i == col + row:
            return False
    return True

def count_conflicts(board, row, col, n):
    conflicts = 0
    for i in range(row + 1, n):
        if not is_safe(board, i, col, n):
            conflicts += 1
    return conflicts

def get_lcv_value(board, row, n):
    lcv_values = []

    for col in range(n):
        if is_safe(board, row, col, n):
            conflicts = count_conflicts(board, row, col, n)
            lcv_values.append((col, conflicts))

    lcv_values.sort(key=lambda x: x[1])
    return [col for col, _ in lcv_values]

#@profile        
def solve_n_queens_lcv(n):
    start_time = time.time()
    board = [-1] * n
    solutions = 0

    def solve_util(row):
        nonlocal solutions

        if row == n:
            solutions += 1
            return

        lcv_values = get_lcv_value(board, row, n)

        for col in lcv_values:
            board[row] = col
            solve_util(row + 1)

    solve_util(0)
    end_time = time.time()
    time_taken = end_time - start_time

    print(f"\nNumber of solutions for {n}-Queens using LCV: {solutions}")
    print(f"Time taken to solve: {time_taken:.6f} seconds\n")

# Simulated Annealing Algorithm
#@profile
def simulated_annealing_n_queens(n, initial_temperature=1000, cooling_rate=0.95, min_temperature=1e-6):
    def generate_initial_solution():
        return [random.randint(0, n-1) for _ in range(n)]

    def accept_solution(current_attacks, new_attacks, temperature):
        if new_attacks < current_attacks:
            return True
        probability = math.exp((current_attacks - new_attacks) / temperature)
        return random.random() < probability

    def apply_random_move(solution):
        new_solution = solution.copy()
        rand_col = random.randint(0, n-1)
        new_solution[rand_col] = random.randint(0, n-1)
        return new_solution

    def temperature_schedule(initial_temperature, cooling_rate, min_temperature):
        temperature = initial_temperature
        while temperature > min_temperature:
            yield temperature
            temperature *= cooling_rate

    def calculate_attacks(solution, n):
        attacks = 0
        for i in range(n):
            for j in range(i + 1, n):
                if solution[i] == solution[j] or abs(i - j) == abs(solution[i] - solution[j]):
                    attacks += 1
        return attacks

    def print_solution(solution):
        board = [["." for _ in range(n)] for _ in range(n)]
        for col, row in enumerate(solution):
            board[row][col] = "Q"

        for row in board:
            print(" ".join(row))
        print()

    def solve_n_queens_util(initial_solution, temperature, n):
        current_solution = initial_solution
        current_attacks = calculate_attacks(current_solution, n)

        solutions = 0

        for temp in temperature_schedule(initial_temperature, cooling_rate, min_temperature):
            new_solution = apply_random_move(current_solution)
            new_attacks = calculate_attacks(new_solution, n)

            if accept_solution(current_attacks, new_attacks, temp):
                current_solution = new_solution
                current_attacks = new_attacks

        if current_attacks == 0:
            solutions += 1
            #print_solution(current_solution)

        return current_solution, solutions

    start_time = time.time()
    initial_solution = generate_initial_solution()
    final_solution, num_solutions = solve_n_queens_util(initial_solution, initial_temperature, n)
    end_time = time.time()

    print(f"Number of solutions for {n}-Queens using Simulated Annealing Algorithm: {num_solutions}")
    print(f"Time taken: {end_time - start_time:.6f} seconds\n")

# Genetic Algorithm
def initialize_population(population_size, board_size):
    return [[random.randint(0, board_size - 1) for _ in range(board_size)] for _ in range(population_size)]

def fitness(board):
    conflicts = 0
    for i in range(len(board)):
        for j in range(i + 1, len(board)):
            if board[i] == board[j] or abs(i - j) == abs(board[i] - board[j]):
                conflicts += 1
    return conflicts

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def mutate(board, mutation_rate):
    if random.random() < mutation_rate:
        index = random.randint(0, len(board) - 1)
        board[index] = random.randint(0, len(board) - 1)

#@profile
def genetic_algorithm(board_size, population_size=100, generations=1000, mutation_rate=0.01):
    population = initialize_population(population_size, board_size)
    solution_count = 0

    for generation in range(generations):
        population = sorted(population, key=lambda x: fitness(x))
        best_solution = population[0]

        if fitness(best_solution) == 0:
            solution_count += 1

        new_population = [best_solution]

        for _ in range(population_size - 1):
            parent1 = random.choice(population[:population_size // 2])
            parent2 = random.choice(population[:population_size // 2])
            child = crossover(parent1, parent2)
            mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

    return solution_count, generations

#@profile
def solve_nqueens(N):
    problem = Problem()

    # Define variables for each row representing the column of the queen
    for i in range(N):
        problem.addVariable(i, range(N))

    # Ensure queens are in different columns (AllDifferentConstraint)
    problem.addConstraint(AllDifferentConstraint(), range(N))

    # Ensure queens are not in the same diagonal
    for i in range(N):
        for j in range(i + 1, N):
            problem.addConstraint(lambda x, y, i=i, j=j: abs(x - y) != abs(i - j), (i, j))

    solutions = problem.getSolutions()
    return len(solutions)


def main():
    n = int(input("Enter the size of the chessboard (N): "))

    forward_checking_nqueens(n)

    solve_n_queens_arc_consistency(n)

    solve_n_queens_mrv(n)

    solve_n_queens_backtracking(n)

    solve_n_queens_lcv(n)

    simulated_annealing_n_queens(n)

    genetic_algorithm(n)

    start_time = time.time()
    num_solutions, generations = genetic_algorithm(n)
    end_time = time.time()
    print(f"Number of solutions for {n}-Queens using Genetic Algorithm: {num_solutions}")
    print(f"Time taken: {end_time - start_time:.4f} seconds\n")

    start_time = time.time()
    num_solutions = solve_nqueens(n)
    end_time = time.time()
    print(f"Number of solutions for {n}-Queens using Constraint Programming: {num_solutions}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")   


if __name__ == "__main__":
    main()
    
    
