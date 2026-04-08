"""
Optimization Visualizer - Home Page
Professional tool for visualizing and exploring various optimization algorithms
"""

import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Optimization Visualizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        border-radius: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🎯 Optimization Visualizer</h1>", unsafe_allow_html=True)

# Introduction
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ## Welcome
    
    This interactive tool provides a comprehensive visualization of various optimization algorithms 
    and techniques. Explore different approaches to solving optimization problems, from classical 
    methods to modern metaheuristics.
    
    Each module offers interactive visualizations, customizable parameters, and real-time performance metrics.
    """)

with col2:
    st.info("""
    **📌 Quick Navigation**
    
    Use the sidebar to access different optimization algorithms:
    - 📊 Unconstrained Minimization
    - 📈 Pareto Front Analysis
    - 🧬 Genetic Algorithm
    - 🌡️ Simulated Annealing
    """)

# Features Section
st.markdown("<h2 class='section-header'>✨ Features</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="feature-box">
    <h4>🎨 Interactive Visualizations</h4>
    Real-time 2D and 3D plots with customizable parameters and dynamic updates.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
    <h4>⚙️ Flexible Parameters</h4>
    Adjust algorithm parameters and observe immediate effects on convergence behavior.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
    <h4>📊 Performance Metrics</h4>
    Track convergence rates, iterations, and fitness values throughout optimization.
    </div>
    """, unsafe_allow_html=True)

# Algorithms Overview
st.markdown("<h2 class='section-header'>🔧 Available Algorithms</h2>", unsafe_allow_html=True)

algorithms = {
    "📊 Unconstrained Minimization": {
        "description": "Minimize functions using gradient-based and gradient-free methods.",
        "methods": ["Gradient Descent", "Nelder-Mead Simplex", "Powell's Method"],
        "best_for": "Smooth, differentiable functions"
    },
    "📈 Pareto Front": {
        "description": "Explore multi-objective optimization by finding Pareto-optimal solutions.",
        "methods": ["Pareto Front Visualization", "Multi-Objective Optimization"],
        "best_for": "Trade-off analysis and decision making"
    },
    "🧬 Genetic Algorithm": {
        "description": "Evolutionary computation approach inspired by natural selection.",
        "methods": ["Selection", "Crossover", "Mutation", "Elitism"],
        "best_for": "Non-convex, black-box optimization problems"
    },
    "🌡️ Simulated Annealing": {
        "description": "Probabilistic technique for approximating global optima.",
        "methods": ["Temperature Control", "Acceptance Probability", "Cooling Schedule"],
        "best_for": "Complex landscapes with multiple local minima"
    }
}

for algo_name, algo_info in algorithms.items():
    with st.expander(algo_name, expanded=False):
        st.write(f"**Description:** {algo_info['description']}")
        st.write(f"**Methods:** {', '.join(algo_info['methods'])}")
        st.write(f"**Best for:** {algo_info['best_for']}")

# Getting Started
st.markdown("<h2 class='section-header'>🚀 Getting Started</h2>", unsafe_allow_html=True)

steps = """
1. **Navigate**: Use the sidebar menu to select an optimization algorithm
2. **Configure**: Adjust the objective function and algorithm parameters
3. **Visualize**: Watch the optimization process in real-time
4. **Analyze**: Review convergence metrics and performance data
5. **Experiment**: Try different settings to understand algorithm behavior
"""

st.markdown(steps)

st.markdown("""
    <div class="highlight">
    <strong>💡 Tip:</strong> Start with Unconstrained Minimization to understand basic concepts, 
    then explore more advanced algorithms like Genetic Algorithm and Simulated Annealing.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <p style="text-align: center; color: #666; font-size: 0.9rem;">
    Optimization Visualizer v1.0 | Professional Optimization Tool
    </p>
    """, unsafe_allow_html=True)
