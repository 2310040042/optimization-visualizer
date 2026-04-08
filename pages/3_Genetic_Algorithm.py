"""
Genetic Algorithm Page
Interactive genetic algorithm visualization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Genetic Algorithm", layout="wide")

# Title
st.markdown("# 🧬 Genetic Algorithm")
st.write("Evolutionary optimization using natural selection principles")

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Objective Function Selection
func_type = st.sidebar.selectbox(
    "Objective Function",
    ["Sphere", "Rosenbrock", "Rastrigin", "Schwefel", "Ackley"]
)

# GA Parameters
population_size = st.sidebar.slider("Population Size:", 20, 200, 50, step=10)
generations = st.sidebar.slider("Generations:", 10, 500, 100, step=10)
mutation_rate = st.sidebar.slider("Mutation Rate:", 0.01, 0.5, 0.1, step=0.01)
crossover_rate = st.sidebar.slider("Crossover Rate:", 0.5, 1.0, 0.8, step=0.05)
elitism_rate = st.sidebar.slider("Elitism Rate:", 0.0, 0.3, 0.1, step=0.05)

# Problem dimension
dimension = st.sidebar.radio("Dimension:", [2, 3])

# Define Objective Functions
def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rastrigin(x):
    A = 10
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

def schwefel(x):
    return 418.9829 * len(x) - sum(x * np.sin(np.sqrt(np.abs(x))))

def ackley(x):
    a, b, c = 20, 0.2, 2*np.pi
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c*x))
    return -a*np.exp(-b*np.sqrt(sum1/len(x))) - np.exp(sum2/len(x)) + a + np.exp(1)

# Map function names to functions
func_map = {
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
    "Schwefel": schwefel,
    "Ackley": ackley
}

objective_func = func_map[func_type]

# Genetic Algorithm Implementation
class GeneticAlgorithm:
    def __init__(self, objective, dimension, population_size, generations, 
                 mutation_rate, crossover_rate, elitism_rate):
        self.objective = objective
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        self.bounds = (-5.12, 5.12)
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'generation': []
        }
        
    def initialize_population(self):
        """Initialize random population"""
        return np.random.uniform(self.bounds[0], self.bounds[1], 
                                (self.population_size, self.dimension))
    
    def evaluate(self, population):
        """Evaluate fitness for all individuals"""
        fitness = np.array([self.objective(ind) for ind in population])
        return fitness
    
    def selection(self, population, fitness):
        """Tournament selection"""
        tournament_size = 3
        selected = []
        for _ in range(len(population)):
            indices = np.random.choice(len(population), tournament_size, replace=False)
            winner = indices[np.argmin(fitness[indices])]
            selected.append(population[winner].copy())
        return np.array(selected)
    
    def crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Gaussian mutation"""
        if np.random.rand() < self.mutation_rate:
            mutation_indices = np.random.rand(len(individual)) < 0.5
            individual[mutation_indices] += np.random.normal(0, 0.5, np.sum(mutation_indices))
            individual = np.clip(individual, self.bounds[0], self.bounds[1])
        return individual
    
    def run(self):
        """Run the genetic algorithm"""
        population = self.initialize_population()
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness = self.evaluate(population)
            
            # Record history
            self.history['best_fitness'].append(np.min(fitness))
            self.history['avg_fitness'].append(np.mean(fitness))
            self.history['worst_fitness'].append(np.max(fitness))
            self.history['generation'].append(gen)
            
            # Elitism - keep best individuals
            n_elite = max(1, int(self.population_size * self.elitism_rate))
            elite_indices = np.argsort(fitness)[:n_elite]
            elite_pop = population[elite_indices].copy()
            
            # Selection and reproduction
            selected = self.selection(population, fitness)
            
            # Create new population
            new_population = []
            for i in range(0, len(selected), 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < len(selected) else selected[0]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Replace with elite
            population = np.array(new_population[:self.population_size])
            for i, elite_idx in enumerate(elite_indices):
                if i < len(population):
                    population[i] = elite_pop[i]
        
        # Final evaluation
        final_fitness = self.evaluate(population)
        best_idx = np.argmin(final_fitness)
        
        return population[best_idx], final_fitness[best_idx], population

# Run GA
if st.sidebar.button("🚀 Run Genetic Algorithm", key="ga_button"):
    
    with st.spinner("Running Genetic Algorithm..."):
        ga = GeneticAlgorithm(objective_func, dimension, population_size, generations,
                             mutation_rate, crossover_rate, elitism_rate)
        
        best_solution, best_fitness, final_population = ga.run()
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Convergence Plot")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(ga.history['generation'], ga.history['best_fitness'], 
                   'g-', linewidth=2, label='Best Fitness')
            ax.plot(ga.history['generation'], ga.history['avg_fitness'], 
                   'b--', linewidth=2, label='Average Fitness')
            ax.plot(ga.history['generation'], ga.history['worst_fitness'], 
                   'r:', linewidth=2, label='Worst Fitness')
            
            ax.set_xlabel('Generation', fontsize=11)
            ax.set_ylabel('Fitness Value', fontsize=11)
            ax.set_title(f'GA Convergence - {func_type}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 📈 Results Summary")
            
            results = {
                "Metric": [
                    "Best Fitness Found",
                    "Average Final Fitness",
                    "Worst Final Fitness",
                    "Generations Run",
                    "Population Size",
                    "Final Population Std Dev"
                ],
                "Value": [
                    f"{best_fitness:.6e}",
                    f"{np.mean(np.array([objective_func(ind) for ind in final_population])):.6e}",
                    f"{np.max(np.array([objective_func(ind) for ind in final_population])):.6e}",
                    f"{generations}",
                    f"{population_size}",
                    f"{np.std(np.array([objective_func(ind) for ind in final_population])):.6e}"
                ]
            }
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Optimal Solution
        st.markdown("### 🎯 Optimal Solution Found")
        
        sol_data = {
            "Variable": [f"x{i+1}" for i in range(dimension)],
            "Value": best_solution
        }
        
        st.dataframe(pd.DataFrame(sol_data), use_container_width=True, hide_index=True)
        
        # Population Analysis
        st.markdown("---")
        st.markdown("### 👥 Final Population Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Fitness", f"{best_fitness:.6e}")
        
        with col2:
            st.metric("Population Diversity", 
                     f"{np.std(np.array([objective_func(ind) for ind in final_population])):.6e}")
        
        with col3:
            convergence_rate = (ga.history['worst_fitness'][0] - best_fitness) / (ga.history['worst_fitness'][0])
            st.metric("Convergence Rate", f"{convergence_rate*100:.1f}%")

# Information Panel
st.markdown("---")
st.markdown("### 📚 Genetic Algorithm Operations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Selection:**
    - Tournament selection picks best individuals from random groups
    - Higher fitness individuals more likely to reproduce
    
    **Crossover:**
    - Single-point crossover combines parent genes
    - Creates genetic diversity in offspring
    """)

with col2:
    st.markdown("""
    **Mutation:**
    - Gaussian mutation adds random perturbations
    - Explores new regions of search space
    
    **Elitism:**
    - Preserves best solutions between generations
    - Prevents loss of good solutions
    """)

# Function Information
with st.expander("🎯 Benchmark Functions"):
    funcs_info = {
        "Sphere": "Simple unimodal function. Global minimum at origin.",
        "Rosenbrock": "Valley-shaped, challenging for many algorithms.",
        "Rastrigin": "Highly multimodal with many local minima.",
        "Schwefel": "Complex landscape with deceptive structure.",
        "Ackley": "Many local minima but one global minimum."
    }
    
    for func_name, description in funcs_info.items():
        st.write(f"**{func_name}:** {description}")

# Help Section
with st.expander("💡 How to Use"):
    st.write("""
    1. **Select Function:** Choose an optimization benchmark function
    2. **Configure Parameters:**
       - Population Size: More individuals = better exploration but slower
       - Generations: More generations = more iterations
       - Mutation Rate: Higher = more exploration, lower = more exploitation
       - Crossover Rate: Probability of combining parent genes
       - Elitism Rate: Percentage of best solutions to preserve
    3. **Run Algorithm:** Click to start the optimization
    4. **Analyze Results:** View convergence patterns and optimal solutions
    """)
