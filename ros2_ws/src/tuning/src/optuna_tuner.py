"""
Optuna-based Bayesian optimization for controller tuning.
Runs headless simulations to find optimal PID and Pure Pursuit parameters.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from controllers import Waypoint
from sim_runner import SimRunner, SimConfig, ControllerParams
from cost_function import compute_cost, CostWeights, EpisodeResult
import paths as path_generators


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ControllerTuner:
    """Optuna-based controller parameter tuner."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.sim_config = self._create_sim_config()
        self.cost_weights = self._create_cost_weights()
        self.test_paths = self._load_test_paths()
        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _create_sim_config(self) -> SimConfig:
        sim_cfg = self.config['simulation']
        return SimConfig(
            use_gui=sim_cfg['use_gui'],
            use_obstacles=sim_cfg['use_obstacles'],
            use_heightmap=sim_cfg['use_heightmap'],
            max_episode_time=sim_cfg['max_episode_time'],
            dt=sim_cfg['dt'],
            control_dt=sim_cfg['control_dt'],
            target_velocity=self.config['fixed']['target_velocity'],
            goal_tolerance=self.config['fixed']['goal_tolerance']
        )

    def _create_cost_weights(self) -> CostWeights:
        w = self.config['cost_weights']
        return CostWeights(
            cross_track=w['cross_track'],
            heading=w['heading'],
            control_effort=w['control_effort'],
            control_smoothness=w['control_smoothness'],
            collision=w['collision'],
            timeout=w['timeout'],
            completion_bonus=w['completion_bonus']
        )
    
    def _load_test_paths(self) -> Dict[str, list]:
        all_paths = path_generators.get_all_test_paths()
        selected = self.config['test_paths']
        return {name: all_paths[name] for name in selected if name in all_paths}
    
    def _sample_params(self, trial: optuna.Trial) -> ControllerParams:
        space = self.config['search_space']
        fixed = self.config['fixed']
        
        return ControllerParams(
            kp=trial.suggest_float('kp', space['kp']['low'], space['kp']['high']),
            ki=trial.suggest_float('ki', space['ki']['low'], space['ki']['high']),
            kd=trial.suggest_float('kd', space['kd']['low'], space['kd']['high']),
            kff=trial.suggest_float('kff', space['kff']['low'], space['kff']['high']),
            lookahead_min=trial.suggest_float('lookahead_min', 
                                              space['lookahead_min']['low'],
                                              space['lookahead_min']['high']),
            k_lookahead=trial.suggest_float('k_lookahead',
                                            space['k_lookahead']['low'],
                                            space['k_lookahead']['high']),
            wheelbase=fixed['wheelbase']
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function. Runs all test paths and returns average cost."""
        params = self._sample_params(trial)
        runner = SimRunner(self.sim_config)
        
        total_cost = 0.0
        num_paths = len(self.test_paths)
        
        try:
            for i, (path_name, path) in enumerate(self.test_paths.items()):
                result = runner.run_episode(params, path)
                cost = compute_cost(result, self.cost_weights)
                total_cost += cost

                trial.report(total_cost / (i + 1), i)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()

                logger.debug(f"Trial {trial.number}, {path_name}: cost={cost:.2f}, "
                           f"collision={result.collision}, completed={result.completed}")
        finally:
            runner.cleanup()

        avg_cost = total_cost / num_paths
        return avg_cost
    
    def run(self, n_trials: Optional[int] = None, n_jobs: Optional[int] = None) -> Dict[str, float]:
        """Run the optimization study."""
        study_cfg = self.config['study']
        pruning_cfg = self.config['pruning']
        
        n_trials = n_trials or study_cfg['n_trials']
        n_jobs = n_jobs or study_cfg['n_jobs']
        
        sampler = TPESampler(seed=42)
        
        pruner = None
        if pruning_cfg['enabled']:
            pruner = MedianPruner(
                n_startup_trials=pruning_cfg['n_startup_trials'],
                n_warmup_steps=pruning_cfg['n_warmup_steps'],
                interval_steps=pruning_cfg['interval_steps']
            )
        
        study = optuna.create_study(
            study_name=study_cfg['name'],
            direction=study_cfg['direction'],
            sampler=sampler,
            pruner=pruner
        )
        
        logger.info(f"Starting optimization: {n_trials} trials, {n_jobs} parallel jobs")
        logger.info(f"Test paths: {list(self.test_paths.keys())}")
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=study_cfg['timeout'],
            show_progress_bar=True
        )
        
        best_params = study.best_params
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best cost: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        if self.config['output']['save_best_params']:
            self._save_best_params(best_params, study.best_value)
        
        if self.config['output']['save_study']:
            self._save_study(study)
        
        return best_params
    
    def _save_best_params(self, params: Dict[str, float], cost: float):
        """Save best parameters in ROS2 YAML format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ros2_params = {
            'control_node': {
                'ros__parameters': {
                    'lookahead_min': params['lookahead_min'],
                    'k_lookahead': params['k_lookahead'],
                    'wheelbase': self.config['fixed']['wheelbase'],
                    'goal_tolerance': self.config['fixed']['goal_tolerance'],
                    'max_velocity': self.config['fixed']['target_velocity'],
                    'control_rate': 1.0 / self.config['simulation']['control_dt'],
                    'pid': {
                        'kp': params['kp'],
                        'ki': params['ki'],
                        'kd': params['kd'],
                        'kff': params['kff']
                    }
                }
            }
        }
        
        output_path = self.results_dir / f"best_params_{timestamp}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(ros2_params, f, default_flow_style=False)
        
        summary_path = self.results_dir / f"tuning_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Tuning Results - {timestamp}\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Best Cost: {cost:.4f}\n\n")
            f.write("Parameters:\n")
            for k, v in params.items():
                f.write(f"  {k}: {v:.6f}\n")
        
        logger.info(f"Saved best parameters to {output_path}")
    
    def _save_study(self, study: optuna.Study):
        """Save study for later analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials_data.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                })
        
        output_path = self.results_dir / f"study_trials_{timestamp}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(trials_data, f, default_flow_style=False)
        
        logger.info(f"Saved study data to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Optuna controller tuning')
    parser.add_argument('--config', type=str, 
                        default='../config/tuning_config.yaml',
                        help='Path to tuning config YAML')
    parser.add_argument('--trials', type=int, default=None,
                        help='Number of trials (overrides config)')
    parser.add_argument('--jobs', type=int, default=None,
                        help='Number of parallel jobs (overrides config)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    tuner = ControllerTuner(str(config_path))
    best_params = tuner.run(n_trials=args.trials, n_jobs=args.jobs)
    
    print("\n" + "="*50)
    print("TUNING COMPLETE")
    print("="*50)
    print("\nBest parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()
