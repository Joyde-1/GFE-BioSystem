class EarlyStopping:
    """
    Implements early stopping to terminate training when a monitored metric has stopped improving.

    Attributes
    ----------
    monitor : str, default 'loss'
        The name of the metric to monitor.
    min_delta : float, default 0.0
        Minimum change in the monitored metric to qualify as an improvement.
    patience : int, default 1
        Number of epochs with no improvement after which training will be stopped.
    verbose : bool, default False
        If True, prints detailed logging of early stopping decisions.
    lower_is_better : bool, default True
        Set to True if a decrease in the metric is considered an improvement, False otherwise.
    start_from_epoch : int, default 0
        The starting epoch number from which the monitoring should begin.
    best_score : float or None
        The best score achieved so far with respect to the monitored metric.
    epochs_no_improve : int
        Number of epochs with no improvement since last improvement or start.
    early_stop : bool
        If True, an early stopping condition was triggered.

    Methods
    -------
    __call__(val_metrics, epoch)
        Called every epoch to check whether early stopping criteria are met.
    _should_stop(current_score)
        Determines if early stopping criteria are met, based on the current score.
    _log_improvement(current_score)
        Logs improvement in the metric if verbose is True.
    """

    def __init__(self, monitor='loss', min_delta=0.000000000, patience=1, verbose=False, lower_is_better=True, baseline=None, start_from_epoch=0):
        """
        Initializes the EarlyStopping instance with the provided parameters.

        Parameters
        ----------
        monitor : str, default 'loss'
            The name of the metric to monitor.
        min_delta : float, default 0.0
            Minimum change in the monitored metric to qualify as an improvement.
        patience : int, default 1
            Number of epochs with no improvement after which training will be stopped.
        verbose : bool, default False
            If True, prints detailed logging of early stopping decisions.
        lower_is_better : bool, default True
            Set to True if a decrease in the metric is considered an improvement, False otherwise.
        baseline : float, optional
            The baseline value for the metric. Training will stop if this value is not reached.
        start_from_epoch : int, default 0
            The starting epoch number from which the monitoring should begin.
        """

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.lower_is_better = lower_is_better
        self.start_from_epoch = start_from_epoch

        self.best_score = baseline
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_metrics, epoch):
        """
        Evaluate whether the training process should be stopped early.

        Parameters
        ----------
        val_metrics : dict
            A dictionary containing the values for monitored metrics.
        epoch : int
            The current epoch number.

        Returns
        -------
        None

        Example
        -------
        >>> es = EarlyStopping(verbose=True)
        >>> metrics = {'loss': 0.1}
        >>> for epoch in range(10):
        ...     es(metrics, epoch)
        ...     if es.early_stop:
        ...         print("Stopping training...")
        ...         break
        """

        if epoch > self.start_from_epoch:

            current_score = val_metrics[self.monitor]

            if self.best_score is None:
                if self.verbose:
                    self._log_improvement(current_score)
                self.best_score = current_score
            elif self._should_stop(current_score):
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        print("Early stopping triggered. \n")
            else:
                if self.verbose:
                    self._log_improvement(current_score)
                self.epochs_no_improve = 0
                self.best_score = current_score
        
    def _should_stop(self, current_score):
        """
        Determine whether the early stopping condition is met.

        Parameters
        ----------
        current_score : float
            The current value of the monitored metric.

        Returns
        -------
        bool
            Returns True if the stopping condition is met, False otherwise.
        """

        if self.lower_is_better:
            return current_score > self.best_score + self.min_delta
        else:
            return current_score < self.best_score - self.min_delta

    def _log_improvement(self, current_score):
        """
        Log the improvement or worsening of the metric.

        Parameters
        ----------
        current_score : float
            The current value of the monitored metric.

        Returns
        -------
        None
        """

        direction = 'decreased' if self.lower_is_better else 'increased'

        if self.best_score == None:
            print(f'New best validation {self.monitor}: {current_score:.4f}. \n')
        else:
            print(f'Validation {self.monitor} {direction} ({self.best_score:.4f} --> {current_score:.4f}). \n')