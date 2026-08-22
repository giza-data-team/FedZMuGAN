import os
from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import torchvision
import torchvision.transforms as transforms
from config_manager import ConfigurationManager

config = ConfigurationManager()
img_size = config.get_image_size()


class DatasetLoader(ABC):
    def __init__(self, image_size=img_size, root="./data", transform_for="classifier"):
        self.root = root
        self.image_size = image_size
        self.transform_for = transform_for
        self.num_classes = None
        self.img_channels = None

    def gan_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @abstractmethod
    def get_train_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def get_test_dataset(self) -> Dataset:
        pass


class MNISTLoader(DatasetLoader):
    def __init__(self, root="./data", image_size=img_size, transform_for="classifier"):
        super().__init__(image_size=image_size, root=root, transform_for=transform_for)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std for MNIST
            ]
        )

    def get_train_dataset(self) -> Dataset:
        return torchvision.datasets.MNIST(
            root=self.root, train=True, download=True, transform=self.transform
        )

    def get_test_dataset(self) -> Dataset:
        return torchvision.datasets.MNIST(
            root=self.root, train=False, download=True, transform=self.transform
        )


class CIFAR10Loader(DatasetLoader):
    def get_train_dataset(self) -> Dataset:
        transform_train = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.Pad(padding=2),
                transforms.RandomCrop(size=(self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784),
                ),
            ]
        )

        return torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
            transform=transform_train
            if self.transform_for == "classifier"
            else self.gan_transform(),
        )

    def get_test_dataset(self) -> Dataset:
        transform_test = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784),
                ),
            ]
        )

        return torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
            transform=transform_test
            if self.transform_for == "classifier"
            else self.gan_transform(),
        )


# class CIFAR100Loader(DatasetLoader):
#     """
#     CIFAR100 loader. Mirrors CIFAR10Loader's transform pipeline but with
#     CIFAR100's own normalization statistics and CIFAR100's 100 classes.
#     """
#
#     mean = (0.5071, 0.4865, 0.4409)
#     std = (0.2673, 0.2564, 0.2762)
#
#     def get_raw_dataset(self) -> Dataset:
#         """
#         Canonical-size, unaugmented, unnormalized [0,1] tensors (Resize + ToTensor
#         only). Used by DataSplitter to build the per-client cache so augmentation
#         can be re-sampled every epoch instead of being baked in once, and so the
#         cached tensors stay directly reusable for real-image GAN training later.
#         """
#         raw_transform = transforms.Compose(
#             [transforms.Resize(self.image_size), transforms.ToTensor()]
#         )
#         return torchvision.datasets.CIFAR100(
#             root=self.root, train=True, download=True, transform=raw_transform
#         )
#
#     def get_train_augment_transform(self):
#         """
#         Stochastic augmentation + normalization, applied on-the-fly at train
#         time to a raw [0,1] CHW tensor produced by get_raw_dataset().
#         """
#         return transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.Pad(padding=2),
#                 transforms.RandomCrop(size=(self.image_size, self.image_size)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(
#                     brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
#                 ),
#                 transforms.ToTensor(),
#                 transforms.Normalize(self.mean, self.std),
#             ]
#         )
#
#     def get_eval_normalize_transform(self):
#         """
#         Deterministic normalization only, applied on-the-fly at eval time to a
#         raw [0,1] CHW tensor produced by get_raw_dataset().
#         """
#         return transforms.Normalize(self.mean, self.std)
#
#     def get_train_dataset(self) -> Dataset:
#         transform_train = transforms.Compose(
#             [
#                 transforms.Resize(self.image_size),
#                 transforms.Pad(padding=2),
#                 transforms.RandomCrop(size=(self.image_size, self.image_size)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(
#                     brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
#                 ),
#                 transforms.ToTensor(),
#                 transforms.Normalize(self.mean, self.std),
#             ]
#         )
#
#         return torchvision.datasets.CIFAR100(
#             root=self.root,
#             train=True,
#             download=True,
#             transform=transform_train
#             if self.transform_for == "classifier"
#             else self.gan_transform(),
#         )
#
#     def get_test_dataset(self) -> Dataset:
#         transform_test = transforms.Compose(
#             [
#                 transforms.Resize(self.image_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize(self.mean, self.std),
#             ]
#         )
#
#         return torchvision.datasets.CIFAR100(
#             root=self.root,
#             train=False,
#             download=True,
#             transform=transform_test
#             if self.transform_for == "classifier"
#             else self.gan_transform(),
#         )
#

class SVHNLoader(DatasetLoader):
    def get_train_dataset(self) -> Dataset:
        transform_train = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.Pad(padding=2),
                transforms.RandomCrop(size=(self.image_size, self.image_size)),
                transforms.ColorJitter(
                    brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4376821, 0.4437697, 0.47280442),
                    (0.19803012, 0.20101562, 0.19703614),
                ),
            ]
        )

        return torchvision.datasets.SVHN(
            root=self.root,
            split="train",
            download=True,
            transform=transform_train
            if self.transform_for == "classifier"
            else self.gan_transform(),
        )

    def get_test_dataset(self) -> Dataset:
        transform_test = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4376821, 0.4437697, 0.47280442),
                    (0.19803012, 0.20101562, 0.19703614),
                ),
            ]
        )

        return torchvision.datasets.SVHN(
            root=self.root,
            split="test",
            download=True,
            transform=transform_test
            if self.transform_for == "classifier"
            else self.gan_transform(),
        )


class EuroSATLoader(DatasetLoader):
    """
    EuroSAT RGB: 27,000 64x64 Sentinel-2 land-use images, 10 classes (including
    "Residential"). torchvision ships it as a single dataset with no official
    train/test split, so one is built here via a fixed stratified split
    (random_state=42) that get_train_dataset()/get_test_dataset() each
    recompute identically, so the two calls stay complementary and reproducible.
    """

    # Commonly-used approximate EuroSAT RGB channel statistics.
    mean = (0.3444, 0.3803, 0.4078)
    std = (0.0929, 0.0651, 0.0552)
    test_size = 0.2

    def _split_indices(self, dataset):
        return train_test_split(
            range(len(dataset)),
            test_size=self.test_size,
            stratify=dataset.targets,
            random_state=42,
        )

    def get_train_dataset(self) -> Dataset:
        transform_train = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        full_dataset = torchvision.datasets.EuroSAT(
            root=self.root,
            download=True,
            transform=transform_train
            if self.transform_for == "classifier"
            else self.gan_transform(),
        )
        train_indices, _ = self._split_indices(full_dataset)
        return Subset(full_dataset, train_indices)

    def get_test_dataset(self) -> Dataset:
        transform_test = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        full_dataset = torchvision.datasets.EuroSAT(
            root=self.root,
            download=True,
            transform=transform_test
            if self.transform_for == "classifier"
            else self.gan_transform(),
        )
        _, test_indices = self._split_indices(full_dataset)
        return Subset(full_dataset, test_indices)


class _ImageListDataset(Dataset):
    """A plain (path, label) list dataset - used where images can't be laid out
    as a single torchvision ImageFolder-compatible root (e.g. after filtering to
    only the top-N identities of a larger on-disk dataset)."""

    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class VGGFace2Loader(DatasetLoader):
    """
    VGGFace2 (https://github.com/ox-vgg/vgg_face2). Registration-gated, so this
    does NOT auto-download - the dataset must already be extracted locally at
    <root>/vggface2/, either as <identity_id>/*.jpg directly, or split across
    subfolders named any of train/val/test/<identity_id>/*.jpg (all present
    ones are auto-detected and merged).

    Common VGGFace2 mirrors (including the official release) use an
    identity-DISJOINT train/test split for face verification - i.e. train/ and
    test/ contain entirely different identities, by design. That's unusable
    for closed-set classification (the same classes need to exist in both
    train and test here), so that split is deliberately ignored: all images
    from every detected subfolder are pooled together first, the
    NUM_TOP_IDENTITIES identities with the most images (combined across
    wherever their images live) are kept, and THOSE images are re-split into a
    fresh, stratified train/test partition (random_state=42, deterministic
    across the two get_*_dataset() calls).
    Remember to set NUM_CLASSES=50 (or NUM_TOP_IDENTITIES, if changed) in .env -
    it isn't inferred automatically from the dataset elsewhere in this codebase.
    """

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    test_size = 0.2
    num_top_identities = 50

    def __init__(self, root="./data", image_size=img_size, transform_for="classifier"):
        super().__init__(
            image_size=image_size,
            root=os.path.join(root, "vggface2"),
            transform_for=transform_for,
        )

    def _identity_files(self):
        """{identity_id: [file paths]}, merging any of root/train, root/val,
        root/test that exist (else falling back to root/ directly)."""
        search_roots = [
            os.path.join(self.root, split)
            for split in ("train", "val", "test")
            if os.path.isdir(os.path.join(self.root, split))
        ] or [self.root]

        if not os.path.isdir(search_roots[0]):
            raise FileNotFoundError(
                f"VGGFace2 not found at {self.root}. Download/extract it first, as "
                f"<root>/<identity_id>/*.jpg or <root>/train|val|test/<identity_id>/*.jpg."
            )

        identity_files = {}
        for search_root in search_roots:
            for identity_id in sorted(os.listdir(search_root)):
                identity_path = os.path.join(search_root, identity_id)
                if not os.path.isdir(identity_path):
                    continue
                files = [
                    os.path.join(identity_path, f)
                    for f in os.listdir(identity_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if files:
                    identity_files.setdefault(identity_id, []).extend(files)

        if not identity_files:
            preview = {}
            for search_root in search_roots:
                try:
                    preview[search_root] = os.listdir(search_root)[:10]
                except OSError as e:
                    preview[search_root] = f"<error listing: {e}>"
            raise FileNotFoundError(
                f"No images found under {search_roots}. Expected "
                f"<search_root>/<identity_id>/*.jpg|jpeg|png. First entries actually "
                f"found: {preview}"
            )

        return identity_files

    def _top_identity_samples(self):
        """Paths/labels for the num_top_identities identities with the most
        images, pooled across every detected subfolder."""
        identity_files = self._identity_files()
        top_identities = sorted(
            identity_files.items(), key=lambda kv: len(kv[1]), reverse=True
        )[: self.num_top_identities]

        self.classes = [identity_id for identity_id, _ in top_identities]
        class_to_idx = {identity_id: i for i, identity_id in enumerate(self.classes)}

        paths, labels = [], []
        for identity_id, files in top_identities:
            paths.extend(files)
            labels.extend([class_to_idx[identity_id]] * len(files))
        return paths, labels

    def _split_dataset(self, transform):
        paths, labels = self._top_identity_samples()
        train_indices, test_indices = train_test_split(
            range(len(paths)),
            test_size=self.test_size,
            stratify=labels,
            random_state=42,
        )
        full_dataset = _ImageListDataset(paths, labels, transform)
        return Subset(full_dataset, train_indices), Subset(full_dataset, test_indices)

    def get_train_dataset(self) -> Dataset:
        transform_train = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        train_subset, _ = self._split_dataset(
            transform_train if self.transform_for == "classifier" else self.gan_transform()
        )
        return train_subset

    def get_test_dataset(self) -> Dataset:
        transform_test = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        _, test_subset = self._split_dataset(
            transform_test if self.transform_for == "classifier" else self.gan_transform()
        )
        return test_subset


class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_name: str, **kwargs) -> DatasetLoader:
        providers = {
            "cifar10": CIFAR10Loader,
            "cifar100": CIFAR100Loader,
            "svhn": SVHNLoader,
            "mnist": MNISTLoader,
            "eurosat": EuroSATLoader,
            "vggface2": VGGFace2Loader,
        }
        if dataset_name.lower() not in providers:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return providers[dataset_name.lower()](**kwargs)
