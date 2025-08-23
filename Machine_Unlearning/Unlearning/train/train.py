import torch
import torch.nn as nn
from client_utils.general_utils import set_seed

set_seed()
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from Machine_Unlearning.Unlearning.train.loss import DistillKL
from copy import deepcopy
import os


class UnlearningProcessor:
    def __init__(
        self, model, model_name, dataset_name, image_size, img_channels, device, configs
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.ref_model = deepcopy(model).to(device)
        self.image_size = image_size
        self.img_channels = img_channels
        self.device = device
        self.configs = configs

        os.makedirs(self.configs["RESULTS_PATH_UNLEARN"], exist_ok=True)

    def our_method(self, forget_dataloader, retain_dataloader, train_with_retain=True):
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.configs["LR_UNLEARN"]
        )
        criterion = DistillKL(4.0)

        ref_model = deepcopy(self.model)
        ref_model.eval()

        self.model.train()
        torch.cuda.empty_cache()

        print("forget  step...")
        for batch in forget_dataloader:
            images, labels = batch
            random_input = torch.rand(images.shape, device=self.device)

            out_r = ref_model(random_input)
            out = self.model(images)

            loss = criterion(out, out_r)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if train_with_retain:
            print("retain  step...")
            for remember_batch in retain_dataloader:
                images, labels = remember_batch
                out_r = ref_model(images)
                out = self.model(images)
                loss = criterion(out, out_r)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        return self.model
