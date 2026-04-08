"""
Pareto Front Page
Multi-objective optimization and Pareto front analysis
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Pareto Front", layout="wide")

# Title
st.markdown("# 📈 Multi-Objective Optimization - Pareto Front")
st.write("Analyze trade-offs between conflicting objectives")

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Problem Selection
problem_type = st.sidebar.selectbox(
    "Problem Type",
    ["Product Design", "Portfolio Optimization", "Engineering Design", "Custom Functions"]
)

# Number of solutions to generate
n_solutions = st.sidebar.slider("Number of Solutions:", 50, 500, 200)

# Define problems and their objectives

def product_design_problem(n):
    """
    Objectives: Minimize Cost, Maximize Quality, Minimize Weight
    """
    np.random.seed(42)
    # Generate random solutions
    solutions = np.random.uniform(0, 10, (n, 3))
    
    costs = solutions[:, 0] * 100 + 50
    quality = 100 - (solutions[:, 1] - 5)**2
    weight = solutions[:, 2] * 5 + 10
    
    return solutions, costs, quality, weight

def portfolio_optimization_problem(n):
    """
    Objectives: Maximize Return, Minimize Risk
    """
    np.random.seed(42)
    weights = np.random.dirichlet(np.ones(5), n)
    
    # Asset returns and volatility
    returns = np.array([0.08, 0.10, 0.12, 0.06, 0.15])
    volatility = np.array([0.10, 0.15, 0.20, 0.08, 0.25])
    
    portfolio_return = weights @ returns
    portfolio_risk = np.sqrt(weights @ (volatility**2))
    
    return weights, portfolio_return, portfolio_risk

def engineering_design_problem(n):
    """
    Objectives: Minimize Material Usage, Maximize Strength
    """
    np.random.seed(42)
    solutions = np.random.uniform(0, 10, (n, 3))
    
    material_usage = np.sum(solutions, axis=1)
    strength = np.sum(solutions**2, axis=1) / 100
    cost = material_usage * 50
    
    return solutions, material_usage, strength, cost

# Generate based on problem type
st.sidebar.write("---")
st.sidebar.write("**Generating Pareto front...**")

if problem_type == "Product Design":
    solutions, costs, quality, weight = product_design_problem(n_solutions)
    objectives_data = np.column_stack([costs, quality, -weight])
    obj_names = ["Cost ($)", "Quality (↑)", "Weight (kg)"]
    n_obj = 3
elif problem_type == "Portfolio Optimization":
    weights, returns, risk = portfolio_optimization_problem(n_solutions)
    objectives_data = np.column_stack([-returns, risk])
    obj_names = ["Expected Return (%)", "Risk (%)"]
    n_obj = 2
elif problem_type == "Engineering Design":
    solutions, material, strength, cost = engineering_design_problem(n_solutions)
    objectives_data = np.column_stack([material, -strength, cost])
    obj_names = ["Material Use (units)", "Strength (↑)", "Cost ($)"]
    n_obj = 3
else:  # Custom - 2D functions
    x = np.random.uniform(-5, 5, n_solutions)
    y = np.random.uniform(-5, 5, n_solutions)
    f1 = x**2 + y**2  # Minimize distance from origin
    f2 = (x-2)**2 + (y-2)**2  # Minimize distance from (2,2)
    objectives_data = np.column_stack([f1, f2])
    obj_names = ["Objective 1", "Objective 2"]
    n_obj = 2

# Calculate Pareto Front
def is_dominated(solution, pareto_set):
    """Check if solution is dominated by any solution in pareto_set"""
    for p in pareto_set:
        if np.all(p <= solution) and np.any(p < solution):
            return True
    return False

def find_pareto_front(objectives):
    """Find Pareto front using non-dominated sorting"""
    pareto_front = []
    indices = []
    
    for i, obj in enumerate(objectives):
        dominated = False
        for j, p_obj in enumerate(objectives):
            if i != j:
                # Check if obj is dominated by p_obj
                if np.all(p_obj <= obj) and np.any(p_obj < obj):
                    dominated = True
                    break
        if not dominated:
            pareto_front.append(obj)
            indices.append(i)
    
    return np.array(pareto_front), np.array(indices)

pareto_front, pareto_indices = find_pareto_front(objectives_data)

# Create tabs for visualization
tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Analysis", "🎯 Solutions"])

with tab1:
    if n_obj == 2:
        # 2D Pareto Front
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # All solutions
        ax.scatter(objectives_data[:, 0], objectives_data[:, 1], 
                  alpha=0.3, s=30, c='lightblue', label='All Solutions', edgecolors='gray')
        
        # Pareto front
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], 
                  alpha=0.8, s=100, c='red', marker='*', 
                  label='Pareto Front', edgecolors='darkred', linewidths=2)
        
        # Sort and connect Pareto points for visualization
        sorted_indices = np.argsort(pareto_front[:, 0])
        ax.plot(pareto_front[sorted_indices, 0], pareto_front[sorted_indices, 1], 
               'r--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel(obj_names[0], fontsize=12)
        ax.set_ylabel(obj_names[1], fontsize=12)
        ax.set_title(f"Pareto Front - {problem_type}", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    else:
        # 3D Pareto Front
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # All solutions
        ax.scatter(objectives_data[:, 0], objectives_data[:, 1], objectives_data[:, 2],
                  alpha=0.2, s=20, c='lightblue', label='All Solutions')
        
        # Pareto front
        ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                  alpha=0.9, s=100, c='red', marker='*', 
                  label='Pareto Front', edgecolors='darkred', linewidths=2)
        
        ax.set_xlabel(obj_names[0], fontsize=10)
        ax.set_ylabel(obj_names[1], fontsize=10)
        ax.set_zlabel(obj_names[2], fontsize=10)
        ax.set_title(f"Pareto Front - {problem_type}", fontsize=12, fontweight='bold')
        ax.legend()
        
        st.pyplot(fig)

with tab2:
    st.markdown("### 📊 Analysis Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Solutions", n_solutions)
    
    with col2:
        st.metric("Pareto Front Size", len(pareto_front))
    
    with col3:
        st.metric("Percentage on Front", f"{(len(pareto_front)/n_solutions)*100:.1f}%")
    
    st.markdown("---")
    
    # Objective statistics
    st.markdown("### 📈 Objective Statistics")
    
    stats_data = []
    for i, obj_name in enumerate(obj_names):
        stats_data.append({
            "Objective": obj_name,
            "Min (All)": f"{objectives_data[:, i].min():.4f}",
            "Max (All)": f"{objectives_data[:, i].max():.4f}",
            "Min (Pareto)": f"{pareto_front[:, i].min():.4f}",
            "Max (Pareto)": f"{pareto_front[:, i].max():.4f}",
            "Std Dev": f"{pareto_front[:, i].std():.4f}"
        })
    
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # Trade-off analysis
    if n_obj == 2:
        st.markdown("### 🔄 Trade-Off Analysis")
        
        sorted_indices = np.argsort(pareto_front[:, 0])
        improvement = []
        for i in range(1, len(sorted_indices)):
            prev_idx = sorted_indices[i-1]
            curr_idx = sorted_indices[i]
            obj1_change = pareto_front[curr_idx, 0] - pareto_front[prev_idx, 0]
            obj2_change = pareto_front[curr_idx, 1] - pareto_front[prev_idx, 1]
            
            if obj1_change != 0:
                trade_off = obj2_change / obj1_change
                improvement.append({
                    "Move": f"{i}",
                    f"Δ {obj_names[0]}": f"{obj1_change:.4f}",
                    f"Δ {obj_names[1]}": f"{obj2_change:.4f}",
                    "Trade-off Ratio": f"{trade_off:.4f}"
                })
        
        if improvement:
            st.dataframe(pd.DataFrame(improvement), use_container_width=True)

with tab3:
    st.markdown("### 🎯 Pareto Optimal Solutions")
    
    # Create solutions table
    solutions_table = []
    for idx, pareto_idx in enumerate(pareto_indices):
        row = {"ID": idx + 1}
        for j, obj_name in enumerate(obj_names):
            row[obj_name] = f"{pareto_front[idx, j]:.4f}"
        solutions_table.append(row)
    
    st.dataframe(pd.DataFrame(solutions_table), use_container_width=True)
    
    # Selected solution details
    st.markdown("---")
    st.markdown("### 🔍 Select a Solution for Details")
    
    solution_id = st.slider("Solution ID:", 1, len(pareto_front), 1)
    selected_solution = pareto_front[solution_id - 1]
    
    cols = st.columns(len(obj_names))
    for i, (col, obj_name) in enumerate(zip(cols, obj_names)):
        with col:
            st.info(f"**{obj_name}**\n\n{selected_solution[i]:.4f}")

# Information Section
st.markdown("---")
st.markdown("### 📚 About Pareto Optimization")

info_text = """
**Pareto Front:** A set of solutions where no single objective can be improved without 
degrading at least one other objective. These solutions represent the best possible trade-offs.

**Key Concepts:**
- **Dominated Solution:** A solution that is worse than another in all objectives
- **Non-dominated Solution:** A solution not dominated by any other solution
- **Pareto Optimal:** Solutions on the Pareto front are Pareto optimal
- **Trade-off:** Moving along the Pareto front improves one objective while worsening another

**Applications:**
- Product design with multiple performance criteria
- Portfolio optimization (return vs. risk)
- Engineering design problems
- Resource allocation with competing objectives
"""

st.markdown(info_text)

# Help Section
with st.expander("💡 How to Use"):
    st.write("""
    1. **Select Problem Type:** Choose a pre-built problem or custom functions
    2. **Adjust Solutions:** Increase the number of candidate solutions for finer front visualization
    3. **View Visualization:** Explore the Pareto front in 2D or 3D
    4. **Analyze Trade-offs:** Examine the metrics and trade-off ratios
    5. **Select Solutions:** Click on individual solutions to review their objective values
    """)
