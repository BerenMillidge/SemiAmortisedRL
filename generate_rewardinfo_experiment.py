import os
import sys
generated_name = str(sys.argv[1])
results_save_base = str(sys.argv[2])
save_path = str(sys.argv[3])
env_name = str(sys.argv[4])
base_call = (f"python main.py --env_name {env_name} --env_std 0.0 --action_repeat 3 --expl_scale 0.1 --save_every 10 --plan_horizon 30 --action_noise 0.0 --max_episode_len 500")
output_file = open(generated_name, "w")

seeds = 5
for s in range(seeds):
    log_path = results_save_base +"/reward_only"
    spath = save_path +"/reward_only/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration False"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/exploration_only"
    spath = save_path +"/exploration_only/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward False"
    f" --use_exploration True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/exploration_reward"
    spath = save_path +"/exploration_reward/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/reward_only_ensemble_model"
    spath = save_path +"/reward_only_ensemble_model/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration False"
    f" --use_ensemble_reward_model True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/exploration_only_ensemble_model"
    spath = save_path +"/exploration_only_ensemble_model/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward False"
    f" --use_exploration True"
    f" --use_ensemble_reward_model True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/exploration_reward_ensemble_model"
    spath = save_path +"/exploration_reward_ensemble_model/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration True"
    f" --use_ensemble_reward_model True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/reward_info_gain_only"
    spath = save_path +"/reward_info_gain_only/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward False"
    f" --use_exploration False"
    f" --use_reward_info_gain True"
    f" --use_ensemble_reward_model True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/reward_info_gain_plus_exploration"
    spath = save_path +"/reward_info_gain_plus_exploration/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward False"
    f" --use_exploration True"
    f" --use_reward_info_gain True"
    f" --use_ensemble_reward_model True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/reward_plus_reward_info_gain"
    spath = save_path +"/reward_plus_reward_info_gain/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration False"
    f" --use_reward_info_gain True"
    f" --use_ensemble_reward_model True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)

for s in range(seeds):
    log_path = results_save_base +"/reward_plus_exploration_plus_reward_info_gain"
    spath = save_path +"/reward_plus_exploration_plus_reward_info_gain/" + str(s)
    final_call = (
    f"{base_call} "
    f" --use_reward True"
    f" --use_exploration True"
    f" --use_reward_info_gain True"
    f" --use_ensemble_reward_model True"
    f" --save_path {spath} "
    f"--logdir {log_path} "
    )
    print(final_call, file=output_file)



print("Experiment file genereated")
output_file.close()
