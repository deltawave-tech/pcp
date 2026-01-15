# wandb_adapter.py
import json
import sys

import wandb


def main():
    # Read config from first line of stdin
    try:
        config_line = sys.stdin.readline()
        if not config_line:
            return
        config = json.loads(config_line)

        # Initialize WandB
        wandb.init(
            project=config.get("project", "pcp-distributed"),
            entity=config.get("entity", None),
            name=config.get("run_name", None),
            config=config.get("hyperparameters", {}),
        )
    except Exception as e:
        print(f"Error initializing wandb: {e}", file=sys.stderr)
        return

    # Read metrics loop
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break  # EOF

            data = json.loads(line)

            # Check for special shutdown command
            if data.get("_command") == "finish":
                break

            step = data.get("outer_step")
            if step is None:
                wandb.log(data)
            else:
                wandb.log(data, step=int(step))

        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"WandB Adapter Error: {e}", file=sys.stderr)

    wandb.finish()


if __name__ == "__main__":
    main()
