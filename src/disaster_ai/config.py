from pathlib import Path

CLASS_NAMES = ("destroyed", "major_damage", "minor_damage", "no_damage")
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}

DEFAULT_DATA_DIR = Path("data/images")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_MODEL_PATH = DEFAULT_ARTIFACT_DIR / "disaster_cnn.pt"
DEFAULT_IMAGE_SIZE = 64
DEFAULT_SEED = 42

TRAIN_EVENTS = {
    "hurricane-florence",
    "hurricane-harvey",
    "hurricane-matthew",
    "midwest-flooding",
    "palu-tsunami",
    "socal-fire",
}
VAL_EVENTS = {"guatemala-volcano", "hurricane-michael"}
TEST_EVENTS = {"mexico-earthquake", "santa-rosa-wildfire"}

EVENT_TO_SPLIT = {
    **{event: "train" for event in TRAIN_EVENTS},
    **{event: "val" for event in VAL_EVENTS},
    **{event: "test" for event in TEST_EVENTS},
}

NORMALIZATION = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}
