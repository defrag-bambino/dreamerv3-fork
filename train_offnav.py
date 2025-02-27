def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  config = config.update({
      'logdir': 'logdir/run1',
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'run.steps': 1000000,
      'batch_size': 32,
      'jax.prealloc': False,
      'encoder.mlp_keys': '.*',
      'decoder.mlp_keys': '.*',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      'jax.platform': 'cpu',
      'encoder.minres': 10,
      'decoder.minres': 10,
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  from mlagents_envs.environment import UnityEnvironment
  from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
  from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
  from stable_baselines3.common.monitor import Monitor
  from stable_baselines3.common.vec_env import DummyVecEnv
  from stable_baselines3.common.vec_env import VecTransposeImage
  from ObsAsDictWrapper import ObsAsDictWrapper
  from embodied.envs import from_gym

  channel = EngineConfigurationChannel()
  channel.set_configuration_parameters(time_scale = 3.0)
  unity_env = UnityEnvironment("/home/fabian/Unity/Projects/OffroadNavigation/Build/offnav1.x86_64", worker_id=0, no_graphics=False, side_channels=[channel])
  env = UnityToGymWrapper(unity_env, 0, allow_multiple_obs=True)
  env = ObsAsDictWrapper(env)
  #env = DummyVecEnv([lambda: env])
  #env = VecTransposeImage(env)
 
  env = from_gym.FromGym(env, obs_key='image')
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)


  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
  main()
