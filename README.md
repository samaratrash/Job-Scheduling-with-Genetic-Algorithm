# Job Scheduling Optimization with Genetic Algorithm

## Overview

This project demonstrates a job scheduling optimization system utilizing a genetic algorithm to efficiently schedule jobs across multiple machines. The goal is to minimize the maximum completion time by determining the optimal sequence of job execution on each machine. The final schedule is visualized using a Gantt chart to provide a clear view of job assignments and timings.

## Features

- **Job Scheduling Optimization:** Uses a genetic algorithm to evolve and optimize job schedules.
- **Validation:** Ensures that the schedule is valid by checking for job overlap and correct sequencing.
- **Visualization:** Generates a Gantt chart to visualize job schedules and timings.

## Usage

1. **Input Jobs:**
   - Enter jobs in the format: `Job_Name: Machine[Duration] -> Machine[Duration] -> ...`
   - Example: `Job1: M1[10] -> M2[5] -> M4[12]`

2. **Run the Algorithm:**
   - The program initializes a population of schedules.
   - Evolves the population through generations using crossover and mutation operations.
   - Validates and selects the best schedules based on fitness.

3. **View Results:**
   - After running the algorithm, a Gantt chart will be displayed, showing the final job schedule across machines.

## Requirements

- Python 3.x
- Pandas
- Matplotlib
- NumPy (optional, depending on implementation details)
