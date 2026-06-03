from abc import ABC, abstractmethod

from smolagents import Tool


class BaseNoisyTool(Tool, ABC):
    """
    The base class for all custom noisy tools.

    The name, description, inputs and output_type attributes are baked into the agent’s system prompt
    upon initialization.

    If you subclass the __init__ method, you can give it no other argument than self.
    Set class attributes for anything you need to hard-code (just set your_variable=(...) directly under the class YourTool(Tool): line).

    Pass this variable to the agent's toolkit
    custom_tool = ImplementedTool()
    """

    # =============== -- TOOL DESCRIPTION --  =============================================

    @property
    @abstractmethod
    def name(self):
        """
        The name of the tool itself: it should be descriptive enough of what this tool does to help the LLM brain
        powering the agent.

        Returns: The name of the tool
        """
        pass

    @property
    @abstractmethod
    def description(self):
        """
        A description of the tool's purpose (included in the agent's system prompt): it is an instruction manual for the LLM powering your agent.

        Returns: The description of the tool.
        """
        pass

    @property
    @abstractmethod
    def output_type(self):
        """
        The output type of the tool (of execute_tool()).

        Returns: The output type of the tool.
        """
        pass

    @property
    @abstractmethod
    def inputs(self):
        """
        A description of the tool's arguments (arguments of execute_tool()).
        A dictionary where each argument name is associated with another dictionary mapping "type" and "description"
        to strings: the argument data type and a description of the argument, respectively.

        Example:
            ```
            inputs = {
                "task": {
                    "type": "string",
                    "description": "the task category (such as text-classification, depth-estimation, etc)",
                }
            }
            ```

        Returns: The dictionary of the tool's arguments.
        """
        pass

    # =============== -- NOISE SWITCHES DEFINITION --  =============================================

    # Probability of returning standard output (instead of noisy output) after the first call.
    # When PROBABILITY_OF_STD_OUTPUT_FIRSTCALL is "noisy", the first call while a noisy switch is active
    # is always noisy. For subsequent calls, the tool returns execute_tool() with probability
    # PROBABILITY_OF_STD_OUTPUT and the switch (noisy) output with probability 1 - PROBABILITY_OF_STD_OUTPUT.
    PROBABILITY_OF_STD_OUTPUT = 0.0
    # Controls how the first call behaves when a noisy switch is active.
    # - "noisy": preserve the current behavior and force the first active call to be noisy.
    # - "random": treat the first active call like any other call, so it can return standard output with
    #   probability PROBABILITY_OF_STD_OUTPUT and noisy output with probability 1 - PROBABILITY_OF_STD_OUTPUT.
    PROBABILITY_OF_STD_OUTPUT_FIRSTCALL = 'random'

    # Internal scope for the legacy noisy-first-call behavior. When the first-call mode is "noisy", this
    # determines whether the first call state is tracked per instance or globally across tool instances.
    PROBABILITY_OF_STD_OUTPUT_BEHAVIOR = 'local'

    import threading
    _thread_local = threading.local()

    # If True, the tool will return a general error message instead of executing the logic in execute_tool().
    SWITCH_RETURN_GENERAL_ERROR_MESSAGE = False

    @property
    def general_error_message(self) -> str:
        """
        If SWITCH_RETURN_GENERAL_ERROR_MESSAGE is True, instead of executing the logic in execute_tool() the tool will return this value as general error message.
        """
        import random
        return random.choice(["The tool could not perform the request. Try again.",
                              "We are unable to process this action at this time. Please try again shortly.",
                              "The requested operation could not be completed. Please retry.",
                              "Your request could not be fulfilled due to a temporary system interruption.",
                              "An error occurred while communicating with the tool. Please try again.",
                              "We apologize for the inconvenience; please attempt to resubmit your request.",
                              "The tool is currently unable to fulfill this command. Please retry."])

    # ======

    @abstractmethod
    def execute_tool(self, *args, **kwargs):
        """
        The method containing the logic of the tool to be executed.
        All imports should be defined directly within the tool’s functions.
        """
        pass

    def forward(self, *args, **kwargs):
        """
        The function that will be called by the agent.
        If none of the noise switches is currently active, this function only executes execute_tool() and returns its
        output. Otherwise, it applies the logic defined by the noise switches and returns the output accordingly.
        """
        import random

        if self.PROBABILITY_OF_STD_OUTPUT_FIRSTCALL == 'noisy':
            if self.PROBABILITY_OF_STD_OUTPUT_BEHAVIOR == 'global':
                is_first_call = not getattr(BaseNoisyTool._thread_local, 'global_called', False)
                BaseNoisyTool._thread_local.global_called = True
            else:
                is_first_call = not hasattr(self, '_noisy_forward_called')
                self._noisy_forward_called = True
        else:
            is_first_call = False

        if self.SWITCH_RETURN_GENERAL_ERROR_MESSAGE:
            if not is_first_call and random.random() < self.PROBABILITY_OF_STD_OUTPUT:
                return self.execute_tool(*args, **kwargs)

            return self.general_error_message
        else:
            return self.execute_tool(*args, **kwargs)

    @classmethod
    def __init_subclass__(cls, **kwargs):
        """
        Once self.execute_tool() is implemented in the child class, this method dynamically injects the same signature
        into the forward() method.
        """
        super().__init_subclass__(**kwargs)
        import inspect
        from functools import wraps

        # Grab the signature from the subclass' implementation of execute_tool()
        signature_execute_tool = inspect.signature(cls.execute_tool)
        # Create a proxy for forward() that carries the same signature as execute_tool()
        original_forward = cls.forward

        @wraps(cls.execute_tool)  # This copies name, docstring, and annotations
        def temp_forward(self, *args, **kwargs):
            return original_forward(self, *args, **kwargs)

        # Inject the signature explicitly
        temp_forward.__signature__ = signature_execute_tool
        # Overwrite forward() in the child class
        setattr(cls, 'forward', temp_forward)
