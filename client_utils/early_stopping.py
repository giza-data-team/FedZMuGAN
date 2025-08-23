class EarlyStopping:
    def __init__(self, patience, min_delta):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement before stopping.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')  
        self.counter = 0
        self.early_stop = False
        self.save_weights = False

    def __call__(self, val_loss):
        """
        Check if validation loss has improved. If not, increment counter and check if patience is exceeded.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_weights = True
            self.counter = 0  
        else:
            self.counter += 1  
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")  

            if self.counter >= self.patience:
                self.early_stop = True  
                return True  

        return False