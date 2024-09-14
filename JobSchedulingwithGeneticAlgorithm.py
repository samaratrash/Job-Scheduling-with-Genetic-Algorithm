import copy
import pandas as pd
import matplotlib.pyplot as plt
import random
import queue
import re

# Job1: M1[10] -> M2[5] -> M4[12]
# Job2: M2[7] -> M3[15] -> M1[8]
# Job3: M1[7] -> M4[15] -> M2[7]

# Job1: M1[5] -> M2[10] -> M4[7]
# Job2: M1[2] -> M3[15] -> M1[8]
# Job3: M1[5] -> M4[10] -> M1[7]

# Job_1: M1[10] -> M2[5] -> M3[12]
# Job_2: M2[7] -> M3[15] -> M1[8]

class job:
    def __init__(self, name, duration, ID):
        self.name = name
        self.duration = duration
        self.start = dict()
        self.ID = ID

    def setstart(self, machine_name, current_time):
        self.start[machine_name] = current_time

    def getstart(self, machine):
        return self.start[machine]


population_size = 50  # number of solution each generation
population_list = list()  # list of solutions(dict)
num_generations = 50
mutation_probability = 0.01

# Function to parse user input
def input_jobs():
    temp_jobs = {}
    print("Enter the job name followed by job sequence (enter an empty line to finish):")

    while True:
        try:
            user_input = input()
            if not user_input:
                break
            # Split the input string into job name and tasks string using ':' as the delimiter
            job_name, tasks_str = user_input.split(':')
            # Strip any leading or trailing whitespace from job name
            job_name = job_name.strip()
            # Split the tasks string into individual tasks using '->' as the delimiter
            tasks = tasks_str.strip().split('->')
            # regex used to manipulate the input string
            temp_jobs[job_name] = [(re.search(r'(M\d+)', task).group(1), int(re.search(r'\[(\d+)\]', task).group(1)))
                                   for task in tasks]
        except:
            # Print an error message if the input format is incorrect
            print("Enter a valid job input format somthing like  'Job_1: M1[10] -> M2[5] -> M4[12]'")
    return temp_jobs


jobs = input_jobs()


def create_gen():
    for i in range(population_size):
        temp = {}
        queue_dictionary = {}
        queue_current_time_dictionary = {}

        for job_name, machine_sequence in jobs.items():
            q = queue.Queue()
            for current_index, (machine, duration) in enumerate(machine_sequence):
                frequency_of_machine = [m[0] for m in machine_sequence[current_index + 1:]].count(machine)
                q.put((machine, duration, frequency_of_machine))
            queue_dictionary[job_name] = q
            queue_current_time_dictionary[q] = 0

        job_names = list(queue_dictionary.keys())
        while any(not q.empty() for q in queue_dictionary.values()):
            job_names = list(queue_dictionary.keys())
            random.shuffle(job_names)  # shuffle the jobs to pick jobs randomly
            job_name = random.choice(job_names)
            for job_name in job_names:
                q = queue_dictionary[job_name]

                if not q.empty():
                    machine, duration, ID = q.get()

                    if machine not in temp:
                        temp[machine] = []
                        initial_start_time = 0
                        start_time = max(queue_current_time_dictionary[q], initial_start_time)
                    else:
                        last_job = temp[machine][-1]
                        start_time = max(queue_current_time_dictionary[q],
                                         (last_job.getstart(machine) + last_job.duration[machine]))

                    temp[machine].append(job(job_name, {machine: duration}, ID))
                    temp[machine][-1].setstart(machine, start_time)
                    queue_current_time_dictionary[q] = start_time + duration

        population_list.append(temp)


def check_validation(solution):
    if solution is None:
        return False

    # Check for intersection between jobs on different machines
    for m1, m1_list in solution.items():
        for m2, m2_list in solution.items():
            if m1 != m2:
                for j1 in m1_list:
                    for j2 in m2_list:
                        if j1.name == j2.name:
                            start_j1_m1 = j1.start[m1]
                            start_j2_m2 = j2.start[m2]
                            end_j1_m1 = start_j1_m1 + j1.duration[m1]
                            end_j2_m2 = start_j2_m2 + j2.duration[m2]

                            # Check for overlap
                            if start_j1_m1 < end_j2_m2 and start_j2_m2 < end_j1_m1:
                                return False

    # Check for correct sequence of jobs on machines
    for job, machine_sequence in jobs.items():
        previous_end = -1
        for current_index, (machine, duration) in enumerate(machine_sequence):
            frequency_of_machine = [m[0] for m in machine_sequence[current_index + 1:]].count(machine)
            if previous_end == -1:
                for i in range(len(solution[machine])):
                    if solution[machine][i].name == job and duration == solution[machine][i].duration[machine] and frequency_of_machine == solution[machine][i].ID:
                        previous_end = solution[machine][i].start[machine] + solution[machine][i].duration[machine]
                        break
            else:
                for i in range(len(solution[machine])):
                    if solution[machine][i].name == job and duration == solution[machine][i].duration[machine] and frequency_of_machine == solution[machine][i].ID:
                        if previous_end > solution[machine][i].start[machine]:
                            return False
                        previous_end = solution[machine][i].start[machine] + solution[machine][i].duration[machine]
                        break

    return True


def fitness(population_list):
    probabilities = []
    total_fitness = 0

    for chrom_index, chrom in enumerate(population_list):
        max_completion_time = 0
        is_valid_chrom = check_validation(chrom)  # Check if the chromosome is valid

        if is_valid_chrom:
            # Calculate the maximum completion time for valid chromosomes
            for key, value_list in chrom.items():
                current_time = 0
                for obj in value_list:
                    current_time += obj.duration[key]
                max_completion_time = max(max_completion_time, current_time)

            # Calculate fitness score for valid chromosomes
            fitness_score = 1 / max_completion_time
            total_fitness += fitness_score

        else:
            # For invalid chromosomes, set fitness score to zero
            fitness_score = 0

        # Append the probability (which may be zero for invalid chromosomes)
        probabilities.append(fitness_score)

    # Calculate probabilities for each chromosome
    if total_fitness > 0:
        probabilities = [(fitness_score / total_fitness) * 100 for fitness_score in probabilities]

    return probabilities


def crossover(parent1, parent2):
    # Select a crossover point that ensures all machines are included in both children
    crossover_point = random.randint(1, len(parent1) - 1)

    # Initialize child chromosomes with copies of the parents' machine assignments
    child1 = {key: list(value) for key, value in parent1.items()}
    child2 = {key: list(value) for key, value in parent2.items()}

    # Swap the machine assignments after the crossover point between parents to create children
    for i in range(crossover_point, len(parent1)):
        key = list(parent1.keys())[i]
        child1[key] = parent2[key]
        child2[key] = parent1[key]

    return child1, child2


def mutation(chromosome):
    old_chromosome = copy.deepcopy(chromosome)

    if random.randint(1, 100) == (mutation_probability * 100):
        # print("muted chromosome in the following generation")
        machine_to_mutate = random.choice(list(chromosome.keys()))

        if len(chromosome[machine_to_mutate]) > 1:  # make sure the machine in chrom has more than one job
            position1 = random.randint(0, len(chromosome[machine_to_mutate]) - 1)
            position2 = random.randint(0, len(chromosome[machine_to_mutate]) - 1)

            while position1 == position2:
                position2 = random.randint(0, len(chromosome[machine_to_mutate]) - 1)

            chromosome[machine_to_mutate][position1], chromosome[machine_to_mutate][position2] = \
                chromosome[machine_to_mutate][position2], chromosome[machine_to_mutate][position1]

            start_time1 = 0
            for job in chromosome[machine_to_mutate]:
                job.setstart(machine_to_mutate, start_time1)
                start_time1 += job.duration[machine_to_mutate]
        else:
            print(f"muted chromosome but position failed on {machine_to_mutate} in the following generation")
            return chromosome

        if check_validation(chromosome):
            print("muted valid chromosome in the following generation")
            return chromosome
        else:
            print("mutation invalid chromosome in the following generation")
            return old_chromosome
    else:
        return chromosome

#function to seclect parents 
def select_parents(population, probabilities):
    return random.choices(population, probabilities, k=2)

#create generation 0
create_gen()

print("Generation 0")
for index, chromosome in enumerate(population_list):
    print(f"Chromosome {index + 1}:")
    for machine_name, value_list in sorted(chromosome.items(), key=lambda x: x[0][-1]):
        print(f" {machine_name}:  ", end='')
        for obj in value_list:
            print(obj.name, " ", obj.start[machine_name], "  ", end='')
        print()
    print()


try:
    for generation in range(num_generations):
        print(f"Generation {generation + 1}:")

        probabilities = fitness(population_list)

        new_population_list = []
        new_probabilities = []

        for _ in range((population_size // 2) - 1):
            #select parents
            parent1, parent2 = select_parents(population_list, probabilities)
            #perform crossover for the chosen parents
            child1, child2 = crossover(parent1, parent2)
            #perform mutation on children
            child1 = mutation(child1)
            new_population_list.append(child1)

            child2 = mutation(child2)
            new_population_list.append(child2)
        #chose the best two chrom from generation
        elite_chromosome1_index = probabilities.index(max(probabilities))

        elite_chromosome1 = population_list[elite_chromosome1_index]

        elite_chromosome2_index = probabilities.index(
            max(probabilities[i] for i in range(len(probabilities)) if probabilities[i] != elite_chromosome1_index))
        elite_chromosome2 = population_list[elite_chromosome2_index]

        # Add elite chromosome to the new population
        new_population_list.append(elite_chromosome1)
        new_population_list.append(elite_chromosome2)

        population_list = [chromosome for chromosome in new_population_list]

        for index, chromosome in enumerate(population_list):
            print(f"Chromosome {index + 1}:")
            for machine_name, value_list in sorted(chromosome.items(), key=lambda x: x[0][-1]):
                print(f" {machine_name}:  ", end='')
                for obj in value_list:
                    print(obj.name, " ", obj.start[machine_name], "  ", end='')
                print()
            print()

    # Prepare Gantt chart data
    gantt_data = []
    job_colors = {}
    color_index = 0

    for chromosome in population_list:
        for machine_name, value_list in chromosome.items():
            sorted_jobs = sorted(value_list, key=lambda x: x.start[machine_name])
            for obj in sorted_jobs:
                gantt_data.append({'Machine': machine_name, 'Job': obj.name,
                                   'Start Time': obj.start[machine_name],
                                   'End Time': obj.start[machine_name] + obj.duration[machine_name]})
                if obj.name not in job_colors:
                    job_colors[obj.name] = color_index
                    color_index += 1

    best_chromosome = min(population_list, key=lambda chrom: max(
        value_list[-1].duration[machine] + value_list[-1].start[machine] for machine, value_list in chrom.items() if
        len(value_list) > 0 and check_validation(chrom)))

    gantt_data = pd.DataFrame(gantt_data)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort machines by their names
    sorted_machines = sorted(gantt_data['Machine'].unique(), key=lambda x: int(re.findall(r'\d+', x)[0]))

    y_positions = {machine: idx + 1 for idx, machine in enumerate(sorted_machines)}

    for index, row in gantt_data.iterrows():
        machine = row['Machine']
        job = row['Job']
        y_pos = y_positions[machine]
        ax.barh(y=y_pos, left=row['Start Time'], width=row['End Time'] - row['Start Time'],
                color=plt.cm.tab10(job_colors[job]))
        ax.text(x=row['Start Time'] + (row['End Time'] - row['Start Time']) / 2, y=y_pos,
                s=row['Job'], ha='center', va='center', color='white', fontsize=8)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(sorted_machines)

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart')

    ax.grid(True)  # Add grid
    plt.show()
except Exception as e:
    print(f"no valid solution is found! the error is {e}")