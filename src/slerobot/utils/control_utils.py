import logging
import traceback
from contextlib import nullcontext
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff
from slerobot.policies.act.modeling_act import ACTEigenCAMHelper
from slerobot.policies.pretrained import PreTrainedPolicy
from slerobot.processor import PolicyAction, PolicyProcessorPipeline
from slerobot.policies.utils import prepare_observation_for_inference


def _clone_numpy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy array entries so `prepare_observation_for_inference` can mutate safely."""
    cloned: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().numpy()
        else:
            cloned[key] = value
    return cloned


def _summarize_tensor(tensor: torch.Tensor, max_items: int = 16) -> str:
    detached = tensor.detach().to("cpu")
    flat = detached.reshape(-1)
    preview = flat[:max_items].tolist()
    suffix = "..." if flat.numel() > max_items else ""
    return f"shape={tuple(detached.shape)} values={preview}{suffix}"

@cache
def is_headless():
    """
    Detects if the Python script is running in a headless environment (e.g., without a display).

    This function attempts to import `pynput`, a library that requires a graphical environment.
    If the import fails, it assumes the environment is headless. The result is cached to avoid
    re-running the check.

    Returns:
        True if the environment is determined to be headless, False otherwise.
    """
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True
    
def init_keyboard_listener():

    """
    Initializes a non-blocking keyboard listener for real-time user interaction.

    This function sets up a listener for specific keys (right arrow, left arrow, escape) to control
    the program flow during execution, such as stopping recording or exiting loops. It gracefully
    handles headless environments where keyboard listening is not possible.

    Returns:
        A tuple containing:
        - The `pynput.keyboard.Listener` instance, or `None` if in a headless environment.
        - A dictionary of event flags (e.g., `exit_early`) that are set by key presses.
    """
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events

def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation_numpy = _clone_numpy_observation(observation)
    attention_helper = getattr(policy, "_attention_helper", None)
    if attention_helper is not None and attention_helper.image_keys_needing_roi_mask:
        ACTEigenCAMHelper.apply_warning_roi_mask_to_observation(
            observation_numpy,
            attention_helper.image_keys_needing_roi_mask,
            policy.config.grad_cam_edge_margin_px,
        )

    debug_counter = getattr(predict_action, "_debug_counter", 0)
    # Grad-CAM++ needs autograd; inference_mode disables it entirely.
    needs_grad = (
        attention_helper is not None
        and getattr(getattr(policy, "config", None), "attention_cam_method", "eigen_cam") == "grad_cam_pp"
        and attention_helper.should_refresh_attention()
    )
    with (
        nullcontext() if needs_grad else torch.inference_mode(),
        torch.autocast(device_type=device.type)
        if device.type == "cuda" and use_amp and not needs_grad
        else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(
            _clone_numpy_observation(observation_numpy), device, task, robot_type
        )
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        raw_action = action
        action = postprocessor(action)

    if debug_counter < 10:
        logging.info("[POLICY_RAW_ACTION] %s", _summarize_tensor(raw_action))
        logging.info("[POLICY_POSTPROCESSED_ACTION] %s", _summarize_tensor(action))
        setattr(predict_action, "_debug_counter", debug_counter + 1)

    return action
