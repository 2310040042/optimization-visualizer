"""
Unconstrained Minimization Page
Visualizes various optimization methods for unconstrained minimization
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Unconstrained Minimization", layout="wide")

# Title
st.markdown("# 📊 Unconstrained Minimization")
st.write("Explore various methods to minimize objective functions")

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Objective Function Selection
objective_func = st.sidebar.selectbox(
    "Objective Function",
    ["Rosenbrock", "Sphere", "Rastrigin", "Beale", "Custom Quadratic"]
)

# Optimization Method Selection
opt_method = st.sidebar.selectbox(
    "Optimization Method",
    ["Nelder-Mead", "L-BFGS-B", "Powell", "CG", "Differential Evolution"]
)

# Problem Dimension
dimensions = st.sidebar.radio("Problem Dimension", ["2D", "3D"])

# Define Objective Functions
def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def sphere(x):
    return sum(x**2)

def rastrigin(x):
    A = 10
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

def beale(x):
    y = (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
    return y

def custom_quadratic(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

# Function mapper
func_map = {
    "Rosenbrock": rosenbrock,
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Beale": beale,
    "Custom Quadratic": custom_quadratic
}

selected_func = func_map[objective_func]

# Initial Point
x0 = st.sidebar.slider("Initial X:", -5.0, 5.0, -2.0)
if dimensions == "3D":
    y0 = st.sidebar.slider("Initial Y:", -5.0, 5.0, -2.0)
    z0 = st.sidebar.slider("Initial Z:", -5.0, 5.0, -2.0)
    initial_point = np.array([x0, y0, z0])
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
else:
    y0 = st.sidebar.slider("Initial Y:", -5.0, 5.0, -2.0)
    initial_point = np.array([x0, y0])
    bounds = [(-5, 5), (-5, 5)]

# Additional parameters
max_iter = st.sidebar.number_input("Max Iterations:", 100, 10000, 1000)

# Optimize
if st.sidebar.button("🚀 Run Optimization", key="opt_button"):
    
    with st.spinner("Optimizing..."):
        # For visualization purposes with 2D
        if dimensions == "2D":
            # Run optimization
            if opt_method == "Differential Evolution":
                result = differential_evolution(selected_func, bounds, maxiter=max_iter, seed=42)
            else:
                result = minimize(selected_func, initial_point, method=opt_method, 
                                 options={"maxiter": max_iter})
            
            # Create visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Contour plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                x_range = np.linspace(-5, 5, 300)
                y_range = np.linspace(-5, 5, 300)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros_like(X)
                
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i, j] = selected_func(np.array([X[i, j], Y[i, j]]))
                
                # Plot contours
                contour = ax.contour(X, Y, Z, levels=30, cmap="viridis", alpha=0.6)
                contourf = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.4)
                ax.clabel(contour, inline=True, fontsize=8)
                plt.colorbar(contourf, ax=ax, label="Function Value")
                
                # Plot initial and final points
                ax.plot(initial_point[0], initial_point[1], 'r*', markersize=15, label="Start")
                ax.plot(result.x[0], result.x[1], 'g*', markersize=15, label="Optimum")
                
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(f"{objective_func} Function - {opt_method}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Results Summary
                st.markdown("### 📊 Optimization Results")
                
                results_data = {
                    "Metric": [
                        "Initial Value",
                        "Final Value",
                        "Improvement",
                        "Iterations",
                        "Success"
                    ],
                    "Value": [
                        f"{selected_func(initial_point):.6e}",
                        f"{result.fun:.6e}",
                        f"{((selected_func(initial_point) - result.fun) / selected_func(initial_point) * 100):.2f}%",
                        f"{result.nit if hasattr(result, 'nit') else 'N/A'}",
                        "✅ Yes" if result.success else "❌ No"
                    ]
                }
                
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
                
                st.markdown("### 🎯 Optimal Point")
                opt_df = pd.DataFrame({
                    "Variable": [f"x{i+1}" for i in range(len(result.x))],
                    "Value": result.x
                })
                st.dataframe(opt_df, use_container_width=True)
                
                st.markdown(f"**Message:** {result.message}")
        
        else:  # 3D
            if opt_method == "Differential Evolution":
                result = differential_evolution(selected_func, bounds, maxiter=max_iter, seed=42)
            else:
                result = minimize(selected_func, initial_point, method=opt_method,
                                 options={"maxiter": max_iter})
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 3D Surface plot
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                x_range = np.linspace(-5, 5, 100)
                y_range = np.linspace(-5, 5, 100)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros_like(X)
                
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        Z[i, j] = selected_func(np.array([X[i, j], Y[i, j], initial_point[2]]))
                
                surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
                ax.plot([initial_point[0]], [initial_point[1]], [selected_func(initial_point)], 'r*', markersize=15, label="Start")
                ax.plot([result.x[0]], [result.x[1]], [result.fun], 'g*', markersize=15, label="Optimum")
                
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Function Value")
                ax.set_title(f"{objective_func} Function (3D)")
                fig.colorbar(surf, ax=ax, shrink=0.5)
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 📊 Optimization Results")
                
                results_data = {
                    "Metric": [
                        "Initial Value",
                        "Final Value",
                        "Improvement",
                        "Iterations",
                        "Success"
                    ],
                    "Value": [
                        f"{selected_func(initial_point):.6e}",
                        f"{result.fun:.6e}",
                        f"{((selected_func(initial_point) - result.fun) / selected_func(initial_point) * 100):.2f}%",
                        f"{result.nit if hasattr(result, 'nit') else 'N/A'}",
                        "✅ Yes" if result.success else "❌ No"
                    ]
                }
                
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
                
                st.markdown("### 🎯 Optimal Point")
                opt_df = pd.DataFrame({
                    "Variable": [f"x{i+1}" for i in range(len(result.x))],
                    "Value": result.x
                })
                st.dataframe(opt_df, use_container_width=True)
                
                st.markdown(f"**Message:** {result.message}")

# Information Panel
st.markdown("---")
st.markdown("### 📚 About Optimization Methods")

method_info = {
    "Nelder-Mead": "Derivative-free simplex method. Works well for non-smooth functions.",
    "L-BFGS-B": "Quasi-Newton method with bounds constraints. Fast for well-behaved functions.",
    "Powell": "Direction-set method. Good for non-smooth functions without gradients.",
    "CG": "Conjugate Gradient method. Efficient for large-scale problems.",
    "Differential Evolution": "Stochastic population-based algorithm. Good for global optimization."
}

with st.expander("📖 Method Details"):
    for method, description in method_info.items():
        st.write(f"**{method}:** {description}")

func_info = {
    "Rosenbrock": "Classic benchmark with a narrow valley. Difficult for gradient-based methods.",
    "Sphere": "Simple function. Global minimum at origin.",
    "Rastrigin": "Highly multimodal. Many local minima make this challenging.",
    "Beale": "Highly nonlinear. Sensitive to initial conditions.",
    "Custom Quadratic": "Modified Rosenbrock. Smooth with single global minimum."
}

with st.expander("🎯 Function Details"):
    for func, description in func_info.items():
        st.write(f"**{func}:** {description}")
