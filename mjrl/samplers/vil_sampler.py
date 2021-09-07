import numpy as np
from mjrl.utils import tensor_utils
from tqdm import tqdm

def trajectory_generator(N,
    expert_policy,
    viz_policy,
    beta,
    seed_offset,
    env,
    use_tactile,
    camera_name,
    use_cuda,
    frame_size=(128, 128),
    device_id = 0,
    pegasus_seed=None):
    """
    params:
    N               : number of trajectories
    policy          : policy to be used to sample the data
    env             : env object to sample from
    pegasus_seed    : seed for environment (numpy speed must be set externally)
    seed_offset     : number to offset all seeds, for each trajectory (offset by seed_offset * N)
    """
    T = env.horizon

    paths = []
    print('Generating trajectories')
    for ep in tqdm(range(N)):
        if pegasus_seed is not None:
            seed = pegasus_seed + ep + seed_offset * N
            env.env.env._seed(seed)
            np.random.seed(seed)
        else:
            np.random.seed()
        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []
        path_image_pixels = []
        all_robot_info = []

        o = env.reset()
        robot_info = env.env.env.get_proprioception(use_tactile=use_tactile)
        done = False
        t = 0
        while t < T and done != True:
            r = np.random.random()
            image_pix = env.env.env.get_pixels(frame_size=frame_size, camera_name=camera_name,
                                            device_id=device_id)

            a_expert, agent_info_expert = expert_policy.get_action(o)

            img = image_pix
            prev_img = image_pix
            prev_prev_img = image_pix
            if t > 0:
                prev_img = path_image_pixels[t - 1]
            if t > 1:
                prev_prev_img = path_image_pixels[t - 2]

            prev_prev_img = np.expand_dims(prev_prev_img, axis=0)
            prev_img = np.expand_dims(prev_img, axis=0)
            img = np.expand_dims(img, axis=0)

            o_img = np.concatenate((prev_prev_img, prev_img, img), axis=0)

            if beta < 1:
                a_viz, agent_info_viz = viz_policy.get_action(o_img, use_cuda=use_cuda,
                                                                   robot_info=robot_info)

            if r <= beta:
                a = agent_info_expert['evaluation']
                agent_info = agent_info_expert
            else:
                a = a_viz
                agent_info = agent_info_viz

            next_o, r, done, env_info = env.step(a)
            observations.append(o)
            actions.append(agent_info_expert['evaluation'])
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_image_pixels.append(image_pix)

            all_robot_info.append(robot_info)
            robot_info = env.env.env.get_proprioception(use_tactile=use_tactile)

            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done,
            image_pixels=np.array(path_image_pixels),
            robot_info=np.array(all_robot_info)
        )
        paths.append(path)
    return paths
