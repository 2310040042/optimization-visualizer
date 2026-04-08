"""
Simulated Annealing Page
Stochastic optimization algorithm inspired by metallurgical annealing
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Simulated Annealing", layout="wide")

# Title
st.markdown("# 🌡️ Simulated Annealing")
st.write("Probabilistic technique for finding global optima in complex landscapes")

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Objective Function Selection
func_type = st.sidebar.selectbox(
    "Objective Function",
    ["Sphere", "Rosenbrock", "Rastrigin", "Beale", "Booth"]
)

# SA Parameters
initial_temp = st.sidebar.slider("Initial Temperature:", 1.0, 100.0, 10.0, step=1.0)
cooling_factor = st.sidebar.slider("Cooling Factor:", 0.8, 0.99, 0.95, step=0.01)
iterations = st.sidebar.slider("Iterations:", 100, 5000, 1000, step=100)
step_size = st.sidebar.slider("Step Size:", 0.1, 2.0, 0.5, step=0.1)

# Problem dimension
dimension = st.sidebar.radio("Dimension:", [2, 3])

# Cooling Schedule
cooling_schedule = st.sidebar.selectbox(
    "Cooling Schedule",
    ["Geometric", "Linear", "Exponential", "Logarithmic"]
)

# Define Objective Functions
def sphere(x):
    return np.sum(x**2)

def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rastrigin(x):
    A = 10
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

def beale(x):
    return ((1.5 - x[0] + x[0]*x[1])**2 + 
            (2.25 - x[0] + x[0]*x[1]**2)**2 + 
            (2.625 - x[0] + x[0]*x[1]**3)**2)

def booth(x):
    return ((x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2)

# Map function names to functions
func_map = {
    "Sphere": sphere,
    "Rosenbrock": rosenbrock,
    "Rastrigin": rastrigin,
    "Beale": beale,
    "Booth": booth
}

objective_func = func_map[func_type]

# Simulated Annealing Implementation
class SimulatedAnnealing:
    def __init__(self, objective, dimension, initial_temp, cooling_factor, 
                 iterations, step_size, cooling_schedule):
        self.objective = objective
        self.dimension = dimension
        self.initial_temp = initial_temp
        self.cooling_factor = cooling_factor
        self.iterations = iterations
        self.step_size = step_size
        self.cooling_schedule = cooling_schedule
        
        self.bounds = (-5.12, 5.12)
        self.history = {
            'current_fitness': [],
            'best_fitness': [],
            'temperature': [],
            'acceptance_rate': [],
            'iteration': []
        }
        
    def get_temperature(self, iteration):
        """Get temperature based on cooling schedule"""
        if self.cooling_schedule == "Geometric":
            return self.initial_temp * (self.cooling_factor ** (iteration / self.iterations))
        elif self.cooling_schedule == "Linear":
            return self.initial_temp * (1 - iteration / self.iterations)
        elif self.cooling_schedule == "Exponential":
            return self.initial_temp * np.exp(-iteration / (self.iterations / 10))
        elif self.cooling_schedule == "Logarithmic":
            return self.initial_temp / np.log(1 + iteration)
    
    def acceptance_probability(self, current_cost, neighbor_cost, temperature):
        """Calculate acceptance probability"""
        if neighbor_cost < current_cost:
            return 1.0
        else:
            return np.exp(-(neighbor_cost - current_cost) / temperature) if temperature > 0 else 0
    
    def generate_neighbor(self, current_solution):
        """Generate neighboring solution"""
        neighbor = current_solution + np.random.normal(0, self.step_size, len(current_solution))
        neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
        return neighbor
    
    def run(self):
        """Run the simulated annealing algorithm"""
        # Initialize
        current_solution = np.random.uniform(self.bounds[0], self.bounds[1], self.dimension)
        current_cost = self.objective(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        accepted_count = 0
        
        for iteration in range(self.iterations):
            temperature = self.get_temperature(iteration)
            
            # Generate and evaluate neighbor
            neighbor = self.generate_neighbor(current_solution)
            neighbor_cost = self.objective(neighbor)
            
            # Acceptance decision
            acceptance_prob = self.acceptance_probability(current_cost, neighbor_cost, temperature)
            
            if np.random.rand() < acceptance_prob:
                current_solution = neighbor
                current_cost = neighbor_cost
                accepted_count += 1
            
            # Update best solution
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
            
            # Record history
            self.history['current_fitness'].append(current_cost)
            self.history['best_fitness'].append(best_cost)
            self.history['temperature'].append(temperature)
            self.history['acceptance_rate'].append(accepted_count / (iteration + 1))
            self.history['iteration'].append(iteration)
        
        return best_solution, best_cost
    
    # Run the algorithm
    def run(self):
        """Run the simulated annealing algorithm"""
        # Initialize
        current_solution = np.random.uniform(self.bounds[0], self.bounds[1], self.dimension)
        current_cost = self.objective(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        accepted_count = 0
        
        for iteration in range(self.iterations):
            temperature = self.get_temperature(iteration)
            
            # Generate and evaluate neighbor
            neighbor = self.generate_neighbor(current_solution)
            neighbor_cost = self.objective(neighbor)
            
            # Acceptance decision
            acceptance_prob = self.acceptance_probability(current_cost, neighbor_cost, temperature)
            
            if np.random.rand() < acceptance_prob:
                current_solution = neighbor
                current_cost = neighbor_cost
                accepted_count += 1
            
            # Update best solution
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
            
            # Record history
            self.history['current_fitness'].append(current_cost)
            self.history['best_fitness'].append(best_cost)
            self.history['temperature'].append(temperature)
            self.history['acceptance_rate'].append(accepted_count / (iteration + 1))
            self.history['iteration'].append(iteration)
        
        return best_solution, best_cost

# Run SA
if st.sidebar.button("🚀 Run Simulated Annealing", key="sa_button"):
    
    with st.spinner("Running Simulated Annealing..."):
        sa = SimulatedAnnealing(objective_func, dimension, initial_temp, cooling_factor,
                               iterations, step_size, cooling_schedule)
        
        best_solution, best_cost = sa.run()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Convergence", "🌡️ Temperature", "📈 Acceptance Rate", "📋 Results"])
        
        with tab1:
            st.markdown("### Convergence Plot")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(sa.history['iteration'], sa.history['best_fitness'], 
                   'g-', linewidth=2.5, label='Best Cost Found', marker='o', markersize=4, markevery=100)
            ax.plot(sa.history['iteration'], sa.history['current_fitness'], 
                   'r--', linewidth=1.5, alpha=0.7, label='Current Cost', marker='x', markersize=3, markevery=100)
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Cost Value', fontsize=12)
            ax.set_title(f'SA Convergence - {func_type} ({cooling_schedule} Cooling)', 
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            st.pyplot(fig)
        
        with tab2:
            st.markdown("### Temperature Schedule")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(sa.history['iteration'], sa.history['temperature'], 
                   'b-', linewidth=2.5, marker='o', markersize=4, markevery=100)
            ax.fill_between(sa.history['iteration'], sa.history['temperature'], alpha=0.3)
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Temperature', fontsize=12)
            ax.set_title(f'Temperature Schedule - {cooling_schedule} Cooling', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with tab3:
            st.markdown("### Move Acceptance Rate")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(sa.history['iteration'], np.array(sa.history['acceptance_rate'])*100, 
                   'purple', linewidth=2.5, marker='s', markersize=4, markevery=100)
            ax.fill_between(sa.history['iteration'], np.array(sa.history['acceptance_rate'])*100, alpha=0.3, color='purple')
            
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Acceptance Rate (%)', fontsize=12)
            ax.set_title('Move Acceptance Rate Over Time', fontsize=13, fontweight='bold')
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Results Summary")
                
                results = {
                    "Metric": [
                        "Best Cost Found",
                        "Final Temperature",
                        "Total Iterations",
                        "Final Acceptance Rate",
                        "Initial Temperature"
                    ],
                    "Value": [
                        f"{best_cost:.6e}",
                        f"{sa.history['temperature'][-1]:.6e}",
                        f"{iterations}",
                        f"{sa.history['acceptance_rate'][-1]*100:.2f}%",
                        f"{initial_temp:.2f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### 🎯 Optimal Solution")
                
                sol_data = {
                    "Variable": [f"x{i+1}" for i in range(dimension)],
                    "Value": best_solution
                }
                
                st.dataframe(pd.DataFrame(sol_data), use_container_width=True, hide_index=True)

# Information Panel
st.markdown("---")
st.markdown("### 📚 How Simulated Annealing Works")

info_cols = st.columns(3)

with info_cols[0]:
    st.markdown("""
    **1. Initialization**
    - Start with random solution
    - Set initial temperature high
    - High temperature = high exploration
    """)

with info_cols[1]:
    st.markdown("""
    **2. Perturbation**
    - Generate neighbor solution
    - Accept if better
    - Accept worse with probability based on temperature
    """)

with info_cols[2]:
    st.markdown("""
    **3. Cooling**
    - Gradually decrease temperature
    - Reduce acceptance of worse solutions
    - Focus on exploitation over time
    """)

# Cooling Schedules
st.markdown("---")
st.markdown("### 🌡️ Cooling Schedules")

schedule_info = {
    "Geometric": "T(t) = T₀ × α^t - Most common, smooth cooling",
    "Linear": "T(t) = T₀ × (1 - t/tₘₐₓ) - Linear decrease",
    "Exponential": "T(t) = T₀ × exp(-t/τ) - Rapid cooling then slow",
    "Logarithmic": "T(t) = T₀ / ln(1+t) - Very slow cooling"
}

for schedule, equation in schedule_info.items():
    st.write(f"**{schedule}:** {equation}")

# Function Information
with st.expander("🎯 Benchmark Functions"):
    funcs_info = {
        "Sphere": "Global minimum at origin (0, 0, ...)",
        "Rosenbrock": "Valley-shaped, minimum at (1, 1, ...)",
        "Rastrigin": "Multimodal with 100+ local minima",
        "Beale": "Complex landscape with irregular structure",
        "Booth": "Global minimum at (1, 3)"
    }
    
    for func_name, description in funcs_info.items():
        st.write(f"**{func_name}:** {description}")

# Help Section
with st.expander("💡 How to Use"):
    st.write("""
    1. **Select Function:** Choose a benchmark function to optimize
    2. **Configure Parameters:**
       - Initial Temperature: Higher = more exploration initially
       - Cooling Factor: Lower = faster cooling, faster convergence to exploitation
       - Iterations: Total number of moves attempted
       - Step Size: Maximum distance of each random move
       - Cooling Schedule: Controls how temperature decreases
    3. **Run Algorithm:** Click to start the optimization
    4. **Analyze Results:**
       - Convergence plot shows how best solution improves
       - Temperature plot shows cooling schedule
       - Acceptance rate shows exploration vs exploitation ratio
    """)
