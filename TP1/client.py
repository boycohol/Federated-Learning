class Client:
    r"""Represents a client participating in the learning process

    Attributes
    ----------
    client_id:

    client_id: int

    learner: Learner

    device: str or torch.device

    train_loader: torch.utils.data.DataLoader

    val_loader: torch.utils.data.DataLoader

    test_loader: torch.utils.data.DataLoader

    train_iterator:

    local_steps: int

    metadata: dict

    logger: torch.utils.tensorboard.SummaryWriter

    """
    def __init__(
            self,
            client_id,
            local_steps,
            logger,
            learner=None,
            train_loader=None,
            val_loader=None,
            test_loader=None,
    ):

        self.client_id = client_id

        self.learner = learner

        self.device = self.learner.device

        if train_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

            self.num_samples = len(self.train_loader.dataset)

            self.train_iterator = iter(self.train_loader)

            self.is_ready = True

        else:
            self.is_ready = False

        self.local_steps = local_steps

        self.logger = logger

        self.metadata = dict()

        self.counter = 0

    def step(self):
        """perform one local round

        """
        self.counter += 1

        # TODO: Perform one local step of training.
        #  Train for 'self.local_steps' epochs using 'trainer.fit_epochs'.
        self.learner.fit_epochs(self.train_loader, self.local_steps)            

    def write_logs(self, counter=None):
        if counter is None:
            counter = self.counter

        # TODO: Implement the logging of training and testing metrics. Evaluate the model on
        #  'self.train_loader' and 'self.test_loader' using 'self.learner'.
        self.step()
        train_loss, train_metric = self.learner.evaluate_loader(self.train_loader)
        test_loss, test_metric = self.learner.evaluate_loader(self.test_loader)


        self.logger.add_scalar("Train/Loss", train_loss, counter)
        self.logger.add_scalar("Train/Metric", train_metric, counter)
        self.logger.add_scalar("Test/Loss", test_loss, counter)
        self.logger.add_scalar("Test/Metric", test_metric, counter)
        self.logger.flush()

        return train_loss, train_metric, test_loss, test_metric
