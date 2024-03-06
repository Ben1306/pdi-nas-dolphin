from nni.experiment import Experiment
import torch

if __name__ == "__main__":
    # Out channels needs to be a multiple of 8 (cf Keras article about EfficientNet params tuning)
    search_space = {
        'k_mult': {'_type': 'choice', '_value': [1, 2, 3, 4, 5, 6]},
        'o_c_1': {'_type': 'choice', '_value': [8, 16]},
        'o_c_2': {'_type': 'choice', '_value': [16, 24]},
        'o_c_3': {'_type': 'choice', '_value': [24, 32, 40]},
        'o_c_4': {'_type': 'choice', '_value': [64, 72, 80]},
        'o_c_5': {'_type': 'choice', '_value': [88, 96, 104, 112]},
        'o_c_6': {'_type': 'choice', '_value': [128, 144, 160, 184]},
        'o_c_7': {'_type': 'choice', '_value': [200, 264, 280]},
        'resolution' : {'_type': 'choice', '_value': [120, 160, 200, 224]},
        'dropout': {'_type': 'uniform', '_value': [0.1, 0.4]},
        #'phi': {'_type': 'uniform', '_value': [0, 0.5]},
    }

    experiment = Experiment('local')
    experiment.config.experiment_name = "EfficientNet NAS"
    experiment.config.trial_command = 'python3 nni_experiment_training.py'
    experiment.config.trial_code_directory = '.'
    if torch.cuda.is_available():
        experiment.config.trial_gpu_number = 1


    experiment.config.search_space = search_space

    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

    #experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 1
    experiment.config.max_experiment_duration = '24h'

    if torch.cuda.is_available():
        experiment.config.training_service.use_active_gpu = True

    experiment.run(8001)

    input('Press enter to quit')
    experiment.stop()