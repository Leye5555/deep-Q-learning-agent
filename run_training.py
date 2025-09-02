import torch
from src import train_dqn
from src import plot_rewards, plot_policy_map, plot_heatmap_with_obstacles, run_lr_experiment
if __name__ == "__main__":

    # --- Main Execution ---

    # 1. Run a full training session to get a good model and reward log
    print("--- Starting Full Training Session for Final Model ---")
    final_dqn_model, final_rewards_log = train_dqn(num_episodes=200, early_stop_reward=15)

    # periodically save dqn training results
    torch.save(final_dqn_model.state_dict(), "results/trained_dqn.pt")

    # 2. Generate the visuals based on the trained model
    print("\n--- Generating Visualizations ---")

    # Visual 1: Plot of rewards over time
    plot_rewards(final_rewards_log, filename='rewards_plot.png')

    # Visual 2: Q-table with arrows (Policy Map)
    plot_policy_map(final_dqn_model, filename='policy_map.png')

    # Visual 4: Heatmap with blacked-out obstacles
    plot_heatmap_with_obstacles(final_dqn_model, filename='heatmap.png')

    # 3. Run the learning rate experiment
    print("\n--- Running Learning Rate Comparison Experiment ---")
    run_lr_experiment(filename='learning_rate_comparison.png')

    print("\n--- All tasks complete. Check the 'results' folder for output images. ---")