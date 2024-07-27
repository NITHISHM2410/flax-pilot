from fpilot.checkpoints import replicate, unreplicate, ocp


def new_manager(save_dir, max2keep):
    options = ocp.CheckpointManagerOptions(max_to_keep=max2keep)
    manager = ocp.CheckpointManager(save_dir, options=options)
    return manager


def save_ckpt(save_dir, save_dict, max2keep):
    save_dict = unreplicate(save_dict)

    with new_manager(save_dir, max2keep) as mngr:
        result = mngr.save(save_dict.step, args=ocp.args.StandardSave(save_dict))
        mngr.wait_until_finished()

    return result


def load_from_ckpt(step, save_path, state, max2keep):
    state = unreplicate(state)

    with new_manager(save_path, max2keep) as mngr:
        state = mngr.restore(step=step, args=ocp.args.StandardRestore(state))

    state = replicate(state)
    return state
