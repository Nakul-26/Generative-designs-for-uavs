import matplotlib.pyplot as plt
import time
import numpy as np
from genetic_optimizer import run_ga
from pso_optimizer import run_pso

NUM_RUNS = 5

def summarize(name, results):
    print(f"\n{name} Results:")
    print(f"Mean: {np.mean(results):.2f}")
    print(f"Best: {np.max(results):.2f}")
    print(f"Std Dev: {np.std(results):.2f}")

def run_comparison():
    print(f"Starting Comparative Study: GA vs PSO ({NUM_RUNS} runs each)")

    ga_best_fitnesses = []
    pso_best_fitnesses = []
    
    ga_histories = []
    pso_histories = []

    for i in range(NUM_RUNS):
        print(f"\n--- Run {i+1}/{NUM_RUNS} ---")
        
        # Run GA
        print("[Executing GA]")
        start_ga = time.time()
        ga_best, ga_history = run_ga()
        ga_runtime = time.time() - start_ga
        ga_best_fitnesses.append(ga_best["mission_fitness"])
        ga_histories.append(ga_history)
        
        # Run PSO
        print("[Executing PSO]")
        start_pso = time.time()
        pso_best, pso_history = run_pso()
        pso_runtime = time.time() - start_pso
        pso_best_fitnesses.append(pso_best["fitness"])
        pso_histories.append(pso_history)

    # Summarize results
    summarize("GA", ga_best_fitnesses)
    summarize("PSO", pso_best_fitnesses)

    # 1. Convergence Plot (averaging histories)
    plt.figure(figsize=(10, 6))
    
    # Pad histories to same length if necessary
    max_len = max(len(h) for h in ga_histories + pso_histories)
    def pad(h, length):
        return h + [h[-1]] * (length - len(h))
    
    ga_avg = np.mean([pad(h, max_len) for h in ga_histories], axis=0)
    pso_avg = np.mean([pad(h, max_len) for h in pso_histories], axis=0)

    plt.plot(ga_avg, label=f"GA (Avg Final: {ga_avg[-1]:.2f})", marker='o', markersize=4)
    plt.plot(pso_avg, label=f"PSO (Avg Final: {pso_avg[-1]:.2f})", marker='s', markersize=4)

    plt.xlabel("Iteration / Generation")
    plt.ylabel("Mean Best Mission Fitness")
    plt.title(f"Algorithm Convergence: GA vs PSO (Average over {NUM_RUNS} runs)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plot_path = "comparison_convergence_multirun.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved multirun convergence plot to {plot_path}")

    # 2. Performance Summary
    stats_path = "comparison_stats.txt"
    with open(stats_path, "w") as f:
        f.write("STATISTICAL COMPARISON: GA vs PSO\n")
        f.write("================================\n\n")
        f.write(f"Number of runs: {NUM_RUNS}\n\n")
        
        f.write("GA Statistics:\n")
        f.write(f"- Mean Fitness: {np.mean(ga_best_fitnesses):.4f}\n")
        f.write(f"- Best Fitness: {np.max(ga_best_fitnesses):.4f}\n")
        f.write(f"- Std Dev: {np.std(ga_best_fitnesses):.4f}\n\n")

        f.write("PSO Statistics:\n")
        f.write(f"- Mean Fitness: {np.mean(pso_best_fitnesses):.4f}\n")
        f.write(f"- Best Fitness: {np.max(pso_best_fitnesses):.4f}\n")
        f.write(f"- Std Dev: {np.std(pso_best_fitnesses):.4f}\n\n")

    print(f"Saved statistical summary to {stats_path}")

if __name__ == "__main__":
    run_comparison()
