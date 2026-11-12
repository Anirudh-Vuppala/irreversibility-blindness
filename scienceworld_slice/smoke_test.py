"""Small API and Java smoke check for the ScienceWorld 1.2.3 install."""

from scienceworld import ScienceWorldEnv


def main() -> None:
    env = ScienceWorldEnv("", serverPath=None, envStepLimit=100)
    try:
        task_names = env.get_task_names()
        if not task_names:
            raise RuntimeError("ScienceWorld returned no task names")

        task_name = "boil" if "boil" in task_names else task_names[0]
        env.load(task_name, variationIdx=0, generateGoldPath=True)
        env.reset()

        actions = env.get_possible_actions()
        if not actions:
            raise RuntimeError(f"No actions available for task {task_name!r}")
        observation, reward, done, info = env.step(actions[0])
        object_tree = env.getObjectTree()
        gold_actions = env.get_gold_action_sequence()

        if not isinstance(object_tree, dict) or not object_tree:
            raise RuntimeError("getObjectTree() returned an empty or invalid tree")
        if not gold_actions or gold_actions[0].startswith("ERROR:"):
            raise RuntimeError(f"Gold action sequence unavailable: {gold_actions}")

        print(f"ScienceWorld task count: {len(task_names)}")
        print(f"First task names: {', '.join(task_names[:5])}")
        print(f"Loaded task: {task_name}")
        print(f"Stepped action: {actions[0]}")
        print(f"Step result: reward={reward}, done={done}, info_keys={sorted(info)}")
        print(f"Observation characters: {len(observation)}")
        print(f"Object tree top-level keys: {sorted(object_tree)}")
        print(f"Gold action count: {len(gold_actions)}")
        print("ScienceWorld API smoke test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
