#!/usr/bin/env python3
"""
Visualization tools for tuning results analysis.
Plots optimization history, parameter importance, and trajectory comparisons.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
import matplotlib.pyplot as plt

from controllers import Waypoint
from sim_runner import SimRunner, SimConfig, ControllerParams
from cost_function import EpisodeResult
import paths as path_generators


def plot_optimization_history(trials_file: str, output_dir: str):
    """Plot cost vs trial number from saved study."""
    with open(trials_file, 'r') as f:
        trials = yaml.safe_load(f)
    
    if not trials:
        print("No completed trials found")
        return
    
    numbers = [t['number'] for t in trials]
    values = [t['value'] for t in trials]
    
    best_so_far = []
    current_best = float('inf')
    for v in values:
        current_best = min(current_best, v)
        best_so_far.append(current_best)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.scatter(numbers, values, alpha=0.5, s=20)
    ax1.plot(numbers, best_so_far, 'r-', linewidth=2, label='Best so far')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Cost')
    ax1.set_title('Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(numbers, best_so_far, 'g-', linewidth=2)
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Best Cost')
    ax2.set_title('Convergence')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'optimization_history.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved optimization history to {output_path}")


def plot_parameter_distributions(trials_file: str, output_dir: str):
    """Plot parameter value distributions across trials."""
    with open(trials_file, 'r') as f:
        trials = yaml.safe_load(f)
    
    if not trials:
        return
    
    param_names = list(trials[0]['params'].keys())
    n_params = len(param_names)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        if i >= len(axes):
            break
        
        values = [t['params'][param] for t in trials]
        costs = [t['value'] for t in trials]
        
        scatter = axes[i].scatter(values, costs, c=range(len(values)), 
                                  cmap='viridis', alpha=0.6, s=20)
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('Cost')
        axes[i].set_title(f'{param} vs Cost')
        axes[i].grid(True, alpha=0.3)
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'parameter_distributions.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved parameter distributions to {output_path}")


def plot_trajectory(result: EpisodeResult, title: str, output_path: str):
    """Plot a single trajectory against its reference path."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    path_x = [wp.x for wp in result.path]
    path_y = [wp.y for wp in result.path]
    traj_x = [tp.pose.x for tp in result.trajectory]
    traj_y = [tp.pose.y for tp in result.trajectory]
    
    axes[0, 0].plot(path_x, path_y, 'b--', linewidth=2, label='Reference Path')
    axes[0, 0].plot(traj_x, traj_y, 'r-', linewidth=1.5, label='Actual Trajectory')
    axes[0, 0].scatter([traj_x[0]], [traj_y[0]], c='green', s=100, marker='o', label='Start')
    axes[0, 0].scatter([traj_x[-1]], [traj_y[-1]], c='red', s=100, marker='x', label='End')
    axes[0, 0].set_xlabel('X [m]')
    axes[0, 0].set_ylabel('Y [m]')
    axes[0, 0].set_title('Path Tracking')
    axes[0, 0].legend()
    axes[0, 0].axis('equal')
    axes[0, 0].grid(True, alpha=0.3)
    
    times = [tp.timestamp for tp in result.trajectory]
    velocities = [tp.velocity for tp in result.trajectory]
    
    axes[0, 1].plot(times, velocities, 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Velocity [m/s]')
    axes[0, 1].set_title('Velocity Profile')
    axes[0, 1].grid(True, alpha=0.3)
    
    steering = [tp.steering for tp in result.trajectory]
    
    axes[1, 0].plot(times, steering, 'g-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Steering [rad]')
    axes[1, 0].set_title('Steering Commands')
    axes[1, 0].grid(True, alpha=0.3)
    
    cross_track_errors = []
    for tp in result.trajectory:
        min_dist = float('inf')
        for wp in result.path:
            dist = np.sqrt((tp.pose.x - wp.x)**2 + (tp.pose.y - wp.y)**2)
            min_dist = min(min_dist, dist)
        cross_track_errors.append(min_dist)
    
    axes[1, 1].plot(times, cross_track_errors, 'm-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Cross-Track Error [m]')
    axes[1, 1].set_title('Tracking Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    status = "COMPLETED" if result.completed else "TIMEOUT"
    if result.collision:
        status = "COLLISION"
    
    fig.suptitle(f'{title} - {status} (Time: {result.total_time:.1f}s)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_and_plot_best_params(params_file: str, output_dir: str):
    """Load best params, run simulation, and plot results."""
    with open(params_file, 'r') as f:
        config = yaml.safe_load(f)
    
    ros_params = config['control_node']['ros__parameters']
    pid = ros_params['pid']
    
    params = ControllerParams(
        kp=pid['kp'],
        ki=pid['ki'],
        kd=pid['kd'],
        kff=pid['kff'],
        lookahead_min=ros_params['lookahead_min'],
        k_lookahead=ros_params['k_lookahead'],
        wheelbase=ros_params['wheelbase']
    )
    
    sim_config = SimConfig(
        use_gui=False,
        use_obstacles=False,  # Match tuning config - no obstacles
        use_heightmap=False,
        max_episode_time=100.0,  # Match tuning config
        target_velocity=ros_params['max_velocity'],
        goal_tolerance=ros_params['goal_tolerance']
    )
    
    runner = SimRunner(sim_config)
    test_paths = path_generators.get_all_test_paths()
    
    output_dir = Path(output_dir)
    
    try:
        for path_name, path in test_paths.items():
            print(f"Running {path_name}...")
            result = runner.run_episode(params, path)
            
            output_path = output_dir / f'trajectory_{path_name}.png'
            plot_trajectory(result, path_name.replace('_', ' ').title(), str(output_path))
            print(f"  Completed: {result.completed}, Collision: {result.collision}, "
                  f"Time: {result.total_time:.1f}s")
    finally:
        runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Visualize tuning results')
    parser.add_argument('--trials', type=str, help='Path to study_trials YAML file')
    parser.add_argument('--params', type=str, help='Path to best_params YAML file')
    parser.add_argument('--output', type=str, default='../results',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.trials:
        print("Plotting optimization history...")
        plot_optimization_history(args.trials, str(output_dir))
        plot_parameter_distributions(args.trials, str(output_dir))
    
    if args.params:
        print("Running best parameters and plotting trajectories...")
        run_and_plot_best_params(args.params, str(output_dir))
    
    if not args.trials and not args.params:
        print("No input files specified. Use --trials and/or --params")
        print("Example: python visualize.py --trials ../results/study_trials_*.yaml "
              "--params ../results/best_params_*.yaml")


if __name__ == '__main__':
    main()
