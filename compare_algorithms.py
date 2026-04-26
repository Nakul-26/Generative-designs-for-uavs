import matplotlib.pyplot as plt
import time
from genetic_optimizer import run_ga
from pso_optimizer import run_pso

def run_comparison():
    print("Starting Comparative Study: GA vs PSO")
    
    # Run GA
    print("\n[Executing Genetic Algorithm]")
    start_ga = time.time()
    ga_best, ga_history = run_ga()
    ga_runtime = time.time() - start_ga
    
    # Run PSO
    print("\n[Executing Particle Swarm Optimization]")
    start_pso = time.time()
    pso_best, pso_history = run_pso()
    pso_runtime = time.time() - start_pso
    
    # 1. Convergence Plot
    plt.figure(figsize=(10, 6))
    plt.plot(ga_history, label=f"GA (Final Fitness: {ga_history[-1]:.2f})", marker='o', markersize=4)
    plt.plot(pso_history, label=f"PSO (Final Fitness: {pso_history[-1]:.2f})", marker='s', markersize=4)
    
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Mission Fitness")
    plt.title("Algorithm Convergence: GA vs PSO")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = "comparison_convergence.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved convergence plot to {plot_path}")
    
    # 2. Performance Summary
    summary_path = "comparison_results.txt"
    with open(summary_path, "w") as f:
        f.write("COMPARATIVE STUDY: GA vs PSO\n")
        f.write("============================\n\n")
        
        f.write("GA Results:\n")
        f.write(f"- Best Fitness: {ga_history[-1]:.4f}\n")
        f.write(f"- Runtime: {ga_runtime:.2f}s\n")
        f.write(f"- Best Design: {ga_best['airfoil']} | Span: {ga_best['wing_span']:.2f}m | Area: {ga_best['wing_area']:.2f}m^2\n\n")
        
        f.write("PSO Results:\n")
        f.write(f"- Best Fitness: {pso_history[-1]:.4f}\n")
        f.write(f"- Runtime: {pso_runtime:.2f}s\n")
        f.write(f"- Best Design: {pso_best['airfoil']} | Span: {pso_best['wing_span']:.2f}m | Area: {pso_best['wing_area']:.2f}m^2\n\n")
        
        f.write("Observation:\n")
        if ga_history[-1] > pso_history[-1]:
            f.write("GA achieved a higher fitness solution in this run.\n")
        else:
            f.write("PSO achieved a higher fitness solution in this run.\n")
            
    print(f"Saved comparison summary to {summary_path}")

if __name__ == "__main__":
    run_comparison()
