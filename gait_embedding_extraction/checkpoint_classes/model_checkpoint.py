class ModelCheckpoint:
    """
    Implements model checkpointing by saving the model state when a monitored metric improves.

    Attributes
    ----------
    monitor : str, default 'loss'
        The name of the metric to monitor for improvement.
    min_delta : float, default 0.0
        Minimum change in the monitored metric to qualify as an improvement, ensuring meaningful model updates.
    verbose : bool, default False
        If True, prints a message whenever the model weights are saved.
    lower_is_better : bool, default True
        Set to True if a decrease in the metric is considered an improvement, False otherwise.
    best_score : float or None
        The best score achieved so far with respect to the monitored metric.
    best_model_state : dict or None
        The best model state dictionary saved when the monitored metric was improved.

    Methods
    -------
    __call__(val_metrics, model)
        Called during training to possibly update the model checkpoint based on the current metric.
    _should_stop(current_score)
        Determines if the checkpoint update should occur, based on the comparison of current and best scores.
    """

    def __init__(self, monitor='loss', min_delta=0.000000000, verbose=False, lower_is_better=True, initial_value_threshold=None):
        """
        Initializes the ModelCheckpoint instance with the provided parameters and sets up initial state.

        Parameters
        ----------
        monitor : str, default 'loss'
            The name of the metric to monitor for improvement.
        min_delta : float, default 0.0
            Minimum change in the monitored metric to qualify as an improvement, ensuring meaningful model updates.
        verbose : bool, default False
            If True, prints a message whenever the model weights are saved.
        lower_is_better : bool, default True
            Set to True if a decrease in the metric is considered an improvement, False otherwise.
        initial_value_threshold : float, optional
            The initial value to set as the best score. If None, the first monitored value is used as the baseline.
        """

        self.monitor = monitor
        self.min_delta = min_delta
        self.verbose = verbose
        self.lower_is_better = lower_is_better

        self.best_score = initial_value_threshold
        self.best_model_state = None

    def __call__(self, val_metrics, model):
        """
        Evaluate current model based on monitored metrics and update checkpoint if improvement is detected.

        Parameters
        ----------
        val_metrics : dict
            A dictionary containing the values for monitored metrics.
        model : any
            The model instance from which the state_dict will be saved if an improvement is detected.

        Returns
        -------
        None

        Example
        -------
        >>> checkpoint = ModelCheckpoint(verbose=True)
        >>> metrics = {'loss': 0.025}
        >>> model = YourModelClass()
        >>> checkpoint(metrics, model)
        """

        current_score = val_metrics[self.monitor]

        if self.best_score is None:
            if self.verbose:
                print("Saved model weights. \n")
            self.best_score = current_score
            self.best_model_state = model.state_dict()
        elif not self._should_stop(current_score):
            if self.verbose:
                print("Saved model weights. \n")
            self.best_score = current_score
            self.best_model_state = model.state_dict()

    def _should_stop(self, current_score):
        """
        Determine whether the checkpoint update should not occur.

        Parameters
        ----------
        current_score : float
            The current value of the monitored metric.

        Returns
        -------
        bool
            Returns True if the checkpoint should not be updated (no improvement), False otherwise.
        """
        
        if self.lower_is_better:
            return current_score > self.best_score + self.min_delta
        else:
            return current_score < self.best_score - self.min_delta